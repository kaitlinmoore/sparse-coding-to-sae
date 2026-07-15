"""Dictionary learning: the Olshausen & Field outer loop.

OWNED (whiteboard-defensible). Alternates the two halves of the O&F objective:

1. Infer codes `a` for a batch of patches with the dictionary held fixed
   (delegated to inference.py).
2. Update the dictionary by gradient on reconstruction error,
   dPhi ~ <(x - Phi a) a^T>, then renormalize columns to unit norm.

Column renormalization is load-bearing, not cosmetic: without it the objective
is trivially minimized by scaling Phi up and `a` down, and the sparsity penalty
stops biting.

Produces the emergent Gabor-like filter grid -- a REPRODUCTION of Olshausen &
Field (1996), labeled as such, not a contribution of this repo.

Stage 3. Unimplemented.
"""
