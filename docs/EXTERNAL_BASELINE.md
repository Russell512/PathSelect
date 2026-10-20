# Independent external baseline

The complete Table 1 includes literature-reported rows from
[Gou et al., AAAI 2025](https://arxiv.org/html/2410.10573v3), clearly marked `*`,
and a distinct **QPMIL-VL (our rerun)** row. Historical imported CSVs were not
treated as evidence of a server rerun. The accepted study completed 20 fresh
official runs and validated all 80 stages.

The external implementation is **not bundled or imported** into PathSelect.
Use an independent checkout and environment:

```bash
git clone https://github.com/can-can-ya/QPMIL-VL.git
cd QPMIL-VL
git checkout 3a7a7698582dec866d43eb748f8c3599f7be4391
```

| Setting | Accepted rerun |
|---|---|
| Folds / orders / stages | 10 / 2 / 4 |
| Seed | Fold ID 1–10, one run per fold/order |
| Epoch cap | 12 per task; official validation/early stopping retained |
| Learning rate / weight decay | `1e-3` / `5e-4` |
| Pool / prompt / match | 20 / 24 / 5 |
| `bp_every_batch` | 16 original, 8 reverse |
| Features and splits | Same original CONCH features and 10 folds |
| Environment | Separate GPU environment, torch 2.6.0+cu126 |
| Checkpoint/scheduler | Official workflow, not a simplified replacement trainer |

This method needs authorized CONCH weights for its learnable text module;
PathSelect's small frozen caches are not a replacement. Obtain access from
the model provider. The publication export contains aggregate results and the
configuration/provenance above, not the external checkout, compatibility patch,
checkpoints, private paths or per-slide predictions. Reproducing that separate
method requires setting up its official environment and data adapters.
Do not claim a new external rerun from these aggregate CSVs alone.

Evaluation uses binary fixed-0.5 or multiclass argmax accuracy, never a
threshold chosen from test ROC. Masked restricts argmax to the true task pair.
Metrics are task-macro and reconstructed at all stages.

To include a separately completed rerun in new PathSelect reports, supply
`OUT/manifests/external_results.json` with `pass`, `method`, `provenance` and
80 `stages` keyed `paper/fold1/stage1` … `reverse/fold10/stage4`.
Each stage entry needs a local `path` and `sha256` of JSON with `rows`.
Each row must include `slide_id`, `patient_id`, `true_task`, `true_label`,
`pred_task`, `pred_label`, `oracle_pred` and `legacy_pred`. Prediction and
patient/label IDs are checked against the same dataset manifest. Do not
fabricate oracle/CIL predictions or per-slide bootstrap intervals from
fold-level CSVs. Do not publish that private manifest with the repository.
