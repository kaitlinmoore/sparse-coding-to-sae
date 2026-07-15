# CLAUDE.md — sparse-coding-v1

## What this is
A from-scratch reproduction of Olshausen & Field (1996) sparse coding on whitened
natural-image patches, extended with quantitative receptive-field analysis against
published V1 statistics and an explicit experimental bridge to sparse autoencoders.

## Standing principles (do not violate)
- The whitening filter, sparse-inference dynamics (LCA + ISTA), and dictionary-update 
  loop are core owned logic: I must be able to rebuild and explain each at a whiteboard.
  AI-assisted code is fine, but every   non-trivial piece gets a plain-language explanation
  in comments or methodology.md.
- Explain as you build. For each solver and metric, record the alternatives considered
  (LCA vs ISTA vs library LASSO; L1 vs log-sparsity) and why the choice won.
- Honest framing. Separate reproduction (emergent Gabor grid) from contribution
  (quantitative analysis, SAE comparison). Report the known sparse-coding-vs-real-V1
  discrepancy rather than hiding it. State limitations plainly.
- Plans before execution. Propose a plan and wait for approval before large changes,
  restructures, or deleting work. I decide.
- Tests where logic is testable. Whitening, inference, dictionary update, and Gabor
  fitting each get pytest tests against synthetic problems with known answers.

## Repo conventions
- Keep raw images, patch caches, and learned dictionaries out of git. Commit code,
  configs, small synthetic fixtures, docs.
- README carries exact data-source URL (van Hateren mirror), image set (.iml linear),
  and access date.
- docs/methodology.md explains approach, choices, evaluation, and limitations.

## Validation discipline
- No sparse solver touches real patches until it matches sklearn LASSO within tolerance
  on a planted-sparse-code synthetic problem.
- The emergent-filter result is not "compartment recovery"-style over-claiming: it is a
  known reproduction. Novelty claims attach only to the quantitative analysis and the
  amortized-vs-iterative SAE comparison.

## Environment
Run under WSL2 (Ubuntu). uv-managed env (pyproject.toml + uv.lock), Python 3.11.
Key deps: numpy, scipy, matplotlib, scikit-learn (LASSO validation), torch (SAE),
pytest. Data fetch: requests (or manual download per README).

## Project-specific instructions
- Whitening cutoff f0 is a locked hyperparameter; document its value and show one
  sensitivity sweep. Do not cherry-pick f0 to prettify filters.
- Report Gabor-fit success fraction; never drop poorly-fit dictionary elements silently
  from tuning distributions.
- Headline V1 comparison is the Ringach (2002) nx-ny envelope scatter.