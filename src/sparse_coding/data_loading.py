"""Load van Hateren natural images and extract training patches.

Library code. Reads the linear ``.iml`` set (1536x1024, 12-bit big-endian, no
header) and cuts random PxP patches. Linear-intensity images are the point: they
are what make whitening-from-scratch a meaningful step rather than a no-op on
already-processed data.

Stage 1. Unimplemented.
"""
