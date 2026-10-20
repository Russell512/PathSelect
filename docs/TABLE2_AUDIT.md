# Table 2(b): whole-slide mean, not random-64

The audited public upstream at
`f5813cd81c845ff11549d69615c32ede6d32c971` defines the row as follows:

1. `scripts/lad1_cells.py`: `MV_RDG` uses `kind="mv"`.
2. `scripts/moe3_common.py`: `mv` resolves to `mean_vec`.
3. `scripts/nc8_batch.py`: `mean_vec` is produced by `mean_norm(Z)` without
   selection indices, for both training and evaluation.
4. `selector/cil_ops.py`: `mean_norm(..., idx=None)` averages all patches.
5. PREREG-24 requires LIN8 and MV_RDG to share the mean-vector subtype ridge;
   their decision rule differs only in whether task routing restricts classes.

The completed server audit traced cached vectors and compared full-mean
representations on slides with more than 64 patches. No qualifying random-64
implementation was found for this row. The claim about a different local
version could not be established merely from the name “task routing.”

This release therefore maps Table 2(b) to `ROUTED_MEAN`. Its source path is
`Run.vector()` in `scripts/paper64_run.py`, which assigns
`vals['MEAN'] = mean_norm(r.Z)[None]`, and the shared subtype fit/evaluation in
the same module. This scientific file is unchanged from the accepted study.

`RANDOM64_RIDGE_FIXED` is separately generated with stable per-slide random
subsets for train/val/test and its own ridge. It is not substituted into the
mean baseline caption or cell. The two mean rows have identical subtype
weights and oracle predictions; this is checked by the experiment pilot.
