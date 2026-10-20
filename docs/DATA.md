# Data and frozen text inputs

The full paper reproduction requires the original, lawfully obtained CONCH
patch feature dataset and existing 10-fold splits. This repository does not
provide or download raw WSIs, feature files or patient identifiers.

Set `PATHSELECT_DATA_ROOT` to the directory containing:

```text
can_dataset/
  tcga_esca/                 # also tcga_rcc, tcga_brca, tcga_lung
    datasplit/fold_1.npz     # through fold_10.npz
    table/TCGA_ESCA_path_subtype_x10_processed.csv
    feats-l1-s256_CONCH/pt_files/<slide_id>.pt
```

Each feature file stores a finite, nonempty `[N,512]` tensor. The loader casts
to float32. Each task table has `patient_id`, `pathology_id`, `label` and
`subtype` columns; local label is 0 or 1, with the mapping in
[METHOD.md](METHOD.md). Split NPZ files contain `train_patients`,
`val_patients`, `test_patients`; this legacy format may contain NumPy object
arrays, so only load trusted split files. Train/val/test patients must be
disjoint within a fold, and test patients must not repeat across folds.
Overlap between different folds' training sets is expected.

The benchmark expects 120 split cells, 2,835 test slides and 2,645 test
patients: ESCA 150, RCC 768, BRCA 952, LUNG 965 test slides. Preflight checks
tables/splits/text hashes, slide and patient set hashes, labels, coverage and
every feature's shape/finiteness/checksum. It fails on missing files; it does
not drop samples or create replacement splits. Hash references in
`configs/benchmark_hashes.json` carry no patient or slide IDs.

The four `cache/text/f_txt_tcga_*.pt` files are the original frozen text
embeddings. Each contains `f_txt` `[2,512]`, `logit_scale`, `class_names` and
task metadata. Their hashes and class order are checked. `data/class_prompts.json`
records the original prompt ensemble. The release uses these caches without
loading a CONCH text tower or redistributing foundation-model weights.

The small `PathSelect` API can be used with other features/texts of the same
dimension and compatible semantics. Such runs are new experiments, not
reproductions of the TCGA results. The strict paper runner deliberately keeps
the original split/hash checks.
