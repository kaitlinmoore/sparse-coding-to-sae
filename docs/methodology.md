# Methodology

_Stage 0 scaffold — section headers only. Each section is filled as its stage lands
(see the build plan). Nothing below is a claimed result yet._

---

## 1. Approach

_What is being reproduced (Olshausen & Field 1996), and what is being added on top._

## 2. Data

_van Hateren linear `.iml` vs deblurred `.imc`: the choice and why. Patch extraction._

## 3. Locked decisions and the alternatives they beat

_Every non-obvious choice gets its rejected alternatives recorded here, per CLAUDE.md._

### 3.1 Whitening: `R(f) = f · exp(−(f/f₀)⁴)`

_Why whiten × low-pass rather than whiten alone. The locked `f₀` value, plus the
one sensitivity sweep — `f₀` is not tuned to prettify filters._

### 3.2 Sparsity cost: L1 vs O&F's `log(1 + a²)`

_L1 for the headline runs (it matches the SAE objective and the LASSO validation
cleanly); the log-sparsity variant run as a documented check on faithfulness to
the paper._

### 3.3 Sparse inference: LCA vs ISTA/FISTA vs library LASSO

_Why LCA is the headline solver, why ISTA/FISTA exists as a second owned
implementation, and why sklearn LASSO is a validation reference only._

### 3.4 Overcompleteness

_~2× (256-dim patches → ~512 elements), and the explicit link to superposition._

## 4. Evaluation

_Gabor fits, tuning distributions, the Ringach (2002) nx–ny envelope scatter,
greedy cosine matching between dictionaries._

## 5. Reproduction vs. contribution

_The emergent-filter grid is a reproduction. The quantitative V1 comparison and
the amortized-vs-iterative SAE experiment are the contribution. Stated plainly._

## 6. Limitations

_Whitening `f₀` sensitivity. Gabor-fit failure fraction. The known
blobbier-than-real-V1 discrepancy. Superposition as analogy, not identity._

## 7. AI assistance

_Owned: whitening, LCA/ISTA, dictionary loop, SAE, Gabor model. Assisted:
Gabor-fit numerics, plotting, scaffolding._
