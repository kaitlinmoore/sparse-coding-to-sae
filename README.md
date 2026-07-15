# sparse-coding-to-saes

Sparse coding of natural images → emergent V1 simple-cell receptive fields → the sparse-autoencoder lineage.

A from-scratch reproduction of Olshausen & Field (1996), extended with quantitative receptive-field analysis against published macaque V1 statistics and an explicit experimental bridge to modern sparse autoencoders. Results sections below are placeholders until the corresponding build stage lands. Nothing here is a claimed finding yet.

---

## Problem / question

Can the receptive-field properties of V1 simple cells — localized, oriented, bandpass spatial filters — emerge purely from an efficient-coding objective applied to natural images, with no supervision and no built-in notion of orientation? Olshausen & Field (1996) showed they can: minimizing reconstruction error under a sparsity penalty over an overcomplete dictionary yields Gabor-like basis functions.

This repo reproduces that result from scratch and then asks two questions the bare reproduction does not answer:

1. **How well do the learned fields match real cortex, quantitatively?** Not "do they look Gabor-like" but: what are their orientation, spatial-frequency, and envelope distributions, and how do those compare to measured macaque V1 simple-cell statistics (Ringach 2002)?
2. **How does classical iterative sparse inference relate to modern sparse autoencoders?** An SAE trained on the same patches uses an *amortized* feed-forward encoder instead of O&F's iterative per-example inference. Does it recover the same receptive-field statistics, or does amortization change the learned dictionary?

The through-line — sparse coding → dictionary learning → sparse autoencoders → superposition — is the same lineage that underpins current mechanistic-interpretability work. Here it is made literal with an experiment rather than asserted.

---

## Repository structure

```
sparse-coding-to-sae/
  data/                    # gitignored: raw .iml/.imc images, patch caches, learned dictionaries
  src/sparse_coding/       # installed as a package (`uv sync`), so imports work from any cwd
    data_loading.py        # load van Hateren images, extract patches
    whitening.py           # OWNED: frequency-domain whitening filter
    inference.py           # OWNED: LCA + ISTA/FISTA sparse solvers
    dictionary.py          # OWNED: dictionary-learning loop, update, column renorm
    gabor.py               # 2D Gabor model + fitting (owned model, scipy optimizer)
    analysis.py            # tuning distributions, Ringach comparison
    sae.py                 # OWNED (small): amortized SAE + dictionary comparison
  notebooks/
    01_whitening.ipynb     # raw vs whitened patch spectra
    02_learning.ipynb      # the emergent-filter grid (reproduction milestone)
    03_tuning_analysis.ipynb  # Gabor fits, tuning distributions, nx-ny vs Ringach 2002
    04_sae_bridge.ipynb    # amortized-vs-iterative dictionary comparison
  tests/
    test_whitening.py      # whitening flattens a synthetic 1/f spectrum
    test_inference.py      # solvers match sklearn LASSO on a planted-sparse-code problem
    test_dictionary.py     # reconstruction error decreases monotonically on a toy problem
    test_gabor.py          # Gabor fit round-trips known parameters
    fixtures/              # small synthetic fixtures (committed)
  docs/
    methodology.md         # approach, choices + rejected alternatives, limitations
    figures/               # committed writeup figures (filter grid, nx-ny scatter)
  README.md
  CLAUDE.md
  .gitignore
  pyproject.toml           # uv-managed deps + metadata
  uv.lock                  # pinned resolve, committed
  LICENSE                  # MIT
```

**Owned vs. library at a glance:** the whitening filter, both sparse solvers (LCA, ISTA/FISTA), the dictionary-learning loop, and the small SAE are implemented from scratch. `sklearn`'s LASSO is used only to *validate* the owned solvers. The Gabor *model* and fitting objective are owned; `scipy.optimize` supplies the numerics.

---

## Data sources

Data is **gitignored** — download locally before running. The van Hateren linear (`.iml`) set is the primary source; use the linear images so the from-scratch whitening step is meaningful.

| Source | URL | Set | Access date |
|---|---|---|---|
| van Hateren natural images (mirror, Paul Ivanov) | `pirsquared.org/research` (`#van-hateren-database`) | Linear `.iml` (primary) / deblurred `.imc` (alt) | 2026-07-14 |
| Convenience fetcher | `github.com/hunse/vanhateren` | auto-download + patch builder | 2026-07-14 |
| Fallback (pre-whitened) | Olshausen `sparsenet` `IMAGES.mat` | 10× 512×512, **pre-whitened** | — |

**Notes.**
- The original host (`hlab.phys.rug.nl`) is defunct; the pirsquared page is the maintained mirror.
- The `hunse/vanhateren` package is fine as a *fetcher*; whitening and patching are done in `src/` so that logic stays owned. If its download URL 404s, fall back to the manual `.tgz` from the mirror.
- The `IMAGES.mat` fallback is **pre-whitened** — using it removes the from-scratch whitening component. Use only if both van Hateren mirrors are down, and if so, exercise the whitening code on a separate raw source so the owned component still exists.
- **Required citation** (dataset terms — scientific, non-commercial use): van Hateren, J.H. & van der Schaaf, A. (1998). *Independent component filters of natural images compared with simple cells in primary visual cortex.* Proc. R. Soc. Lond. B, 265, 359–366.

---

## Environment & setup

Runs under WSL2 (Ubuntu), Python 3.11, uv-managed.

```bash
# from the repo root
uv sync                    # creates .venv and installs pinned deps from uv.lock
```

Key dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-learn` (LASSO validation), `torch` (SAE), `pytest`.

---

## How to run

In order. Each step assumes `uv run` (or an activated `.venv`).

```bash
# 1. fetch data (see Data sources; manual download or the convenience fetcher)
#    place van Hateren .iml images under data/

# 2. whitening — inspect raw vs whitened spectra
uv run jupyter lab notebooks/01_whitening.ipynb

# 3. validate the sparse solvers BEFORE any real-data run
uv run pytest tests/test_inference.py    # gate: must pass before Stage 3

# 4. learn the dictionary — produces the emergent-filter grid
uv run jupyter lab notebooks/02_learning.ipynb

# 5. quantitative tuning analysis vs Ringach 2002
uv run jupyter lab notebooks/03_tuning_analysis.ipynb

# 6. SAE bridge — amortized vs iterative dictionary comparison
uv run jupyter lab notebooks/04_sae_bridge.ipynb

# full test suite
uv run pytest
```

---

## Results / EDA summary

_Placeholders — populated as each stage lands. No numbers here are claimed findings yet._

- **Whitening** _(Stage 1)_ — raw patch spectra fall as ~1/f; the whitening filter flattens them up to the low-pass cutoff `f₀`. Figure: `01_whitening.ipynb`. _(f₀ value + one sensitivity sweep to be recorded here.)_
- **Emergent receptive fields** _(Stage 3, reproduction)_ — the learned overcomplete dictionary converges to localized, oriented, bandpass filters. Figure: filter grid in `02_learning.ipynb`. _(This is a reproduction of Olshausen & Field 1996, labeled as such — not a novel result.)_
- **Tuning distributions** _(Stage 4, contribution)_ — orientation, spatial-frequency, and phase distributions across the dictionary; the nx–ny envelope scatter overlaid on Ringach (2002) macaque V1 values. _(Well-fit Gabor fraction + the expected sparse-coding-vs-real-V1 discrepancy to be reported honestly here.)_
- **SAE bridge** _(Stage 5, contribution)_ — dictionary learned by the amortized SAE vs the iterative sparse-coding dictionary, compared by Gabor-parameter distributions and greedy cosine matching. _(Answer to the amortized-vs-iterative question to be stated here.)_

---

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for the full approach: every locked design choice with the alternatives it beat (LCA vs ISTA vs library LASSO; L1 vs O&F's log-sparsity cost; `.iml` linear vs `.imc` deblurred images), the evaluation strategy, and limitations.

---

## Limitations & honest scope

- **The emergent-filter grid is a reproduction, not a contribution.** Sparse-coding reproductions exist as tutorials. What is owned-and-novel here is the from-scratch inference, the quantitative V1 comparison (Stage 4), and the amortized-vs-iterative SAE experiment (Stage 5).
- **Known discrepancy with real V1.** Sparse-coding dictionaries tend to come out blobbier and more symmetric than measured V1 simple-cell fields, which skew more elongated. This is expected and is reported, not tuned away.
- **Whitening sensitivity.** Results depend on the whitening cutoff `f₀`; it is locked, documented, and shown with one sensitivity sweep rather than cherry-picked.
- **Gabor-fit failure fraction.** Not every dictionary element is Gabor-like (DC blobs, edge artifacts). The well-fit fraction is reported; poorly-fit elements are not silently dropped from the distributions.
- **Superposition is an analogy here, not an identity.** The overcomplete-dictionary ↔ superposition connection is illustrative of the lineage; this repo does not claim to establish properties of superposition in trained networks.

---

## AI assistance disclosure

- **Owned and re-derivable** (rebuildable and explainable at a whiteboard): the frequency-domain whitening filter, the LCA and ISTA/FISTA sparse solvers, the dictionary-learning loop and update, the small sparse autoencoder, and the 2D Gabor model.
- **Library, used deliberately:** `sklearn` LASSO (solver validation only), `scipy.optimize` (Gabor-fit numerics), `torch` (SAE training).
- **AI-assisted:** Gabor-fit numerics wiring, plotting/visualization code, and repo scaffolding. Every non-trivial owned component carries a plain-language explanation in comments or `docs/methodology.md`.
