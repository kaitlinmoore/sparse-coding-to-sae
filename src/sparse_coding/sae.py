"""Small sparse autoencoder: the amortized-vs-iterative bridge.

OWNED (small). Linear encoder -> ReLU -> linear decoder, L1 penalty on the codes,
trained in torch on the same whitened patches the iterative solver sees.

The point is the contrast, not the architecture. O&F infer codes by running an
optimization per example; an SAE amortizes that inference into a single
feed-forward pass. The sharp question: does the amortized encoder recover the
same receptive-field statistics, or does amortization change the dictionary?
Compared via Gabor-parameter distributions and greedy cosine matching against the
Stage 3 dictionary.

Stage 5. Unimplemented.
"""
