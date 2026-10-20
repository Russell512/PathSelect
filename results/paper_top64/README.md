# Accepted aggregate results

These 31 files are byte-identical aggregate exports from the accepted
`paper_top64_20261016` study. Their SHA256 values are recorded in
[PROVENANCE.json](PROVENANCE.json) and tested. There are no slide/patient IDs,
model weights or machine paths in these exports. `literature_rows.json` is the
separate cited input for reported rows.

| Result | File |
|---|---|
| Table 1 | [Markdown](table1_preview.md), [CSV](table1.csv), [LaTeX](table1.tex) |
| Table 2 ladder | [Markdown](table2_ladder.md), [CSV](table2_ladder.csv), [LaTeX](table2_ladder.tex) |
| Table 2 2×2 factorial | [Markdown](table2_factorial.md), [CSV](table2_factorial.csv), [LaTeX](table2_factorial.tex) |
| Random-64 control | [CSV](random64_control.csv) |
| Top-64 vs four rounds | [Five seeds](selection_four16_vs_top64_5seed.csv), [three-seed subset](selection_four16_vs_top64_3seed.csv) |
| Budget and gamma | [Budget](budget_sweep.csv), [validation](gamma_validation.csv), [descriptive test sweep](gamma_test_sensitivity.csv) |
| Paired statistics | [CSV](paired_comparisons.csv) |
| Fold/seed/stage results | [Folds](per_fold.csv), [seed-folds](per_seed_fold.csv), [stages](stage_curves.csv) |
| Task/class diagnostics | [Tasks](per_task.csv), [classes](per_class.csv), [confusion](confusion.csv), [routing](router_errors.csv) |
| Costs | [Timing](timing_summary.csv), [speedup](selection_speedup.csv), [retained state bytes](model_state_bytes.csv) |
| Metric definitions | [Mapping](metrics_mapping.txt), [Masked audit](masked_definition_audit.md) |

Original / reverse order is preserved. Accuracy is equal-task macro, not
pooled slide accuracy. Seed averaging precedes fold summaries; fold SD uses
`ddof=1`. Fractions, percentages and percentage-point differences are distinct
columns. `Masked_ORACLE` is the paper-ready definition; legacy routed-evidence
Masked remains separately named.

The ladder ACC values are **85.66 → 90.45 → 91.99%**. The 2×2 factorial ACC is
B/cosine **88.10%**, B/ridge **91.99%**, A/cosine **91.29%**, A/ridge **92.38%**.
At fixed gamma `1e-3`, random-64 gives **90.15%**, full mean **90.30%**, and B
top-64 **91.99%**. B budget K=16/32/64/128/256 gives approximately
**91.72/92.01/91.99/92.06/92.09%**; test sweeps did not change the selected model.

B top-64 minus four-round is −0.147 pp; A (five seeds) is −0.143 pp. Patient
95% CIs are respectively [−0.367, 0.061] and [−0.272, −0.027] pp. These
results are retained even though top-64 is slightly less accurate.

Timing and statistical qualifications are in
[PROVENANCE.md](../../docs/PROVENANCE.md) and [METHOD.md](../../docs/METHOD.md).
Do not interpret a CI covering zero as equivalence. This was a follow-up after
prior test exploration. Internal model computation used CPU; the external
rerun used its separately documented GPU environment; starred literature
rows originate from their cited publication.
