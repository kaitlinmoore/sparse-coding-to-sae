"""Load van Hateren natural images and extract training patches.

Library code. Reads the linear ``.iml`` set (1536x1024, 12-bit big-endian, no
header) and cuts random PxP patches. Linear-intensity images are the point: they
are what make whitening-from-scratch a meaningful step rather than a no-op on
already-processed data.

Pipeline order (locked at Stage 1): load -> log-luminance -> whiten the whole
image (see whitening.py) -> extract patches. Whitening the full image rather
than each patch keeps full frequency resolution and avoids patch-boundary
artifacts; patches are cut from the already-whitened image.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

# van Hateren .iml geometry: 1536 wide x 1024 tall, so a C-order flat buffer
# reshapes to (rows=height, cols=width) = (1024, 1536).
IML_SHAPE = (1024, 1536)


def load_iml(path: str | Path, *, shape: tuple[int, int] = IML_SHAPE) -> np.ndarray:
    """Read one van Hateren linear ``.iml`` image.

    The format is headerless: unsigned 12-bit samples stored as 16-bit
    big-endian integers (dtype ``>u2``), row-major. Returns a float64 2D array
    of linear luminance counts.
    """
    data = np.fromfile(path, dtype=">u2")
    expected = shape[0] * shape[1]
    if data.size != expected:
        raise ValueError(
            f"{path}: read {data.size} samples, expected {expected} for shape {shape}. "
            "Wrong file, wrong shape, or a truncated download."
        )
    return data.reshape(shape).astype(np.float64)


def load_images(
    source: str | Path | Iterable[str | Path], *, shape: tuple[int, int] = IML_SHAPE
) -> list[np.ndarray]:
    """Load all ``.iml`` under a directory, or an explicit iterable of paths."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        paths = sorted(p.glob("*.iml")) if p.is_dir() else [p]
    else:
        paths = [Path(x) for x in source]
    if not paths:
        raise FileNotFoundError(
            f"No .iml images found at {source!r}. Download the van Hateren linear "
            "set into data/ (see README > Data sources)."
        )
    return [load_iml(pp, shape=shape) for pp in paths]


def preprocess_image(image: np.ndarray, *, log_luminance: bool = True) -> np.ndarray:
    """Prepare a raw linear image for whitening.

    Linear calibrated luminance has a very large dynamic range, so a handful of
    bright pixels otherwise dominate the spectrum. Taking log10 compresses that
    range and is the standard treatment for van Hateren linear images (van
    Hateren & van der Schaaf 1998). We floor at the smallest positive value in
    the image so log10 is defined without assuming a particular intensity scale.
    """
    img = np.asarray(image, dtype=np.float64)
    if log_luminance:
        positive = img[img > 0]
        floor = positive.min() if positive.size else 1.0
        img = np.log10(np.maximum(img, floor))
    return img


def extract_patches(
    images: np.ndarray | Iterable[np.ndarray],
    n_patches: int,
    patch_size: int,
    *,
    rng: np.random.Generator,
    remove_dc: bool = True,
) -> np.ndarray:
    """Cut ``n_patches`` random PxP patches from one or more images.

    ``rng`` is a ``np.random.Generator``; the draw sequence is deterministic
    given the generator's seed, so calling this twice with identically-seeded
    generators over the same-shaped image list yields patches at the *same*
    locations. That is how the notebook compares raw vs whitened patches fairly.

    Returns ``(n_patches, patch_size, patch_size)`` float64. With ``remove_dc``
    the per-patch mean is subtracted (the whitening filter also removes DC, but
    patches cut from a raw image still carry it).
    """
    if isinstance(images, np.ndarray) and images.ndim == 2:
        images = [images]
    images = [np.asarray(im, dtype=np.float64) for im in images]
    if not images:
        raise ValueError("No images provided to extract_patches.")

    p = patch_size
    patches = np.empty((n_patches, p, p), dtype=np.float64)
    n_img = len(images)
    for i in range(n_patches):
        im = images[rng.integers(n_img)]
        h, w = im.shape
        if h < p or w < p:
            raise ValueError(f"Image {im.shape} smaller than patch size {p}.")
        r = rng.integers(0, h - p + 1)
        c = rng.integers(0, w - p + 1)
        patch = im[r : r + p, c : c + p]
        if remove_dc:
            patch = patch - patch.mean()
        patches[i] = patch
    return patches
