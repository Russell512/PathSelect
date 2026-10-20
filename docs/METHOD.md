# Method and evaluation protocol

## Representations

For raw CONCH patch features `Z` of shape `[N,512]`, the task router uses
`u = concat(L2(mean(Z)), 1)`. Its ridge gamma is always `1e-3`.
FINAL-B scores each patch by the within-slide z-score of its maximum cosine
similarity to the **two texts of the specified task**. `one_shot(score, 64)`
uses the original `top_k_select` behavior, including ties and selecting all
patches when `N <= 64`. Formal evidence is `L2(mean(Z[indices]))`: selected
**raw** features are equally weighted before normalization.

At training time the supplied task is the true training task, never the true
subtype as a selector. At CIL inference the task comes from the router.
Subtype prediction is restricted to that predicted task's two classes.
Cosine ablations reuse exactly the same evidence and change only the readout.
They still have a supervised router and are not a zero-shot pipeline.

The subtype ridge retains cumulative `S = sum(v v^T)` and
`R = sum(v one_hot(y)^T)`, where `v = concat(evidence, 1)`. It solves
`W = solve(S + gamma I, R)`. The bias is regularized like the other dimensions.
Statistics and solves use CPU float64; feature/head arithmetic uses float32.
Previously seen classes retain their statistics; only current-task training
slides are added. Validation and test never enter these statistics.

## Residual-head ablation and controls

A adds `I6Expert(r=2)` to B's patch score (1,033 trainable parameters per task).
A uses five seeds `[42,43,44,45,46]`, initialized before model construction:
5 epochs, Adam, lr `5e-4`, weight decay `1e-4`, one train WSI per update,
`seed + epoch` shuffle. Training uses detached top-64 indices and **selected-score
softmax pooling**, followed by the original classification cross-entropy.
After freezing, official evidence uses equal pooling. The same trained head is
shared across orders, readouts and selection strategies.

Four-round controls select 16 patches per round, four rounds, lambda `1.5`.
They build their own training evidence and readout weights. Selection-only
comparisons fix both gammas at `1e-3`; delta means top-64 minus four-round.

Random-64 uses SHA256 of compact JSON `[seed, fold, slide_id]`, the first eight
bytes interpreted little-endian, then NumPy `default_rng.choice` without
replacement. Its subset is independent of task predictions, stages and orders.
Train/val/test each use random evidence and its own subtype ridge. Five seeds
are averaged within folds. This control is distinct from Table 2(b).

## Orders, labels and validation

| Task ID | Dataset | Global class columns |
|---|---|---|
| 0 | ESCA | 0 ESAD, 1 ESCC |
| 1 | RCC | 2 CCRCC, 3 PRCC |
| 2 | BRCA | 4 IDC, 5 ILC |
| 3 | LUNG | 6 LUAD, 7 LUSC |

Original (`paper`) order: LUNG → BRCA → RCC → ESCA. Reverse:
ESCA → RCC → BRCA → LUNG. All four stages are evaluated, with only seen tasks
and classes eligible. There are 10 existing folds, not newly sampled splits.

Subtype gamma is selected separately for MEAN, B64 and A64 from
`[1e-5,1e-4,1e-3,1e-2,1e-1]`. The objective is validation oracle within-task
accuracy, equal-weighted by task within stage, then by order/stage/fold; A
also averages seeds. Exact ties choose the larger gamma. One gamma is shared
by the whole representation across folds, seeds and orders. The selected
values were MEAN `1e-2`, B64 `1e-3`, A64 `1e-3`. LIN8 and ROUTED_MEAN share the
same subtype weights and oracle predictions. Selection is frozen before
formal test analysis. The test gamma sweep is explicitly descriptive/post hoc.

## Metrics and uncertainty

- ACC: stage-4 equal-task accuracy, then fold mean and sample SD (`ddof=1`).
- Masked_ORACLE / WP: true task chooses both evidence and the final class pair.
- Masked_ROUTED_EVIDENCE: evidence uses predicted task; only the final pair
  uses true task. This legacy measure is secondary and must not replace WP.
- `R[t,j]`: CIL accuracy for task j after stage t.
- Forgetting: average for the first three tasks of their maximum accuracy
  **before the final stage** minus final accuracy. No clamp to zero.
- BWT: final accuracy minus the accuracy at each task's arrival, first three
  tasks averaged. AIA: mean of the four stage ACC values.
- Seeded methods average metrics within fold; no prediction ensemble and no
  50-sample significance test. Paired comparisons have `n=10` folds per order.
- Two-sided exact binomial sign tests omit ties at tolerance `1e-12`. Raw p
  and supplemental Holm-adjusted p are saved separately.
- Slide CI: paired `(fold,task)`-stratified bootstrap, 1,000 draws,
  `torch.Generator(seed=0)`. All methods/seeds share resampled indices.
- Patient CI: paired patient-cluster bootstrap within `(fold,task)`, 10,000
  draws, NumPy seed `20261014`; all slides of a sampled patient are retained.
  Accuracy is slide-weighted within the stratum, then tasks/folds are equal.

The two orders are not independent. Cross-validation training sets overlap.
Bootstrap uncertainty is conditional on the fitted models and does not cover
all training variability. A p value above 0.05 or a CI containing zero does
not demonstrate equivalence or noninferiority.
