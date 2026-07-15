"""2D Gabor model and fitting.

SPLIT OWNERSHIP, stated honestly: the 2D Gabor model and the least-squares
objective are owned; scipy.optimize.least_squares supplies the numerics. The
model parameterization (center, orientation, spatial frequency, phase, and the
envelope widths nx, ny) is what the downstream V1 comparison depends on, so it
is written here rather than pulled from a library.

Every learned dictionary element gets a fit. Fit quality is reported, and
poorly-fit elements are NOT silently dropped from the tuning distributions
(CLAUDE.md).

Stage 4. Unimplemented.
"""
