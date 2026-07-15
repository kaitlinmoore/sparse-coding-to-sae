"""Sparse inference: solve for codes `a` given a fixed dictionary and a patch.

OWNED (whiteboard-defensible). Two from-scratch solvers for
    min_a ||x - Phi a||^2 + lambda * S(a)

- LCA (Rozell et al. 2008): headline solver. Continuous-time thresholding
  dynamics over leaky integrator units -- the biologically-motivated one, and the
  one that connects to the neural story the project is telling.
- ISTA/FISTA: second owned implementation and independent cross-check. A
  well-understood proximal-gradient method with a known convergence rate.

sklearn's LASSO is the third leg of the triangulation -- a VALIDATION reference
only, never part of the owned path. Gate (CLAUDE.md): neither solver touches real
patches until both match sklearn LASSO within tolerance on a planted-sparse-code
synthetic problem.

Stage 2. Unimplemented.
"""
