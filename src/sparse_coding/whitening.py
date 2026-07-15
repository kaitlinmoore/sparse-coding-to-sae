"""Frequency-domain whitening filter.

OWNED (whiteboard-defensible). Implements R(f) = f * exp(-(f / f0) ** 4): a
whitening ramp that flattens the ~1/f amplitude spectrum of natural images,
multiplied by a low-pass envelope that suppresses the high-frequency noise the
ramp would otherwise amplify. Whitening alone is not enough -- the ramp boosts
exactly the band where sensor noise dominates signal, so the low-pass is what
keeps the learned dictionary from fitting noise.

f0 is a locked hyperparameter (CLAUDE.md): document the value, show one
sensitivity sweep, do not tune it to prettify the filters.

Stage 1. Unimplemented.
"""
