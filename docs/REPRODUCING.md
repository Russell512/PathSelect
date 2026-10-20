# Reproduce the accepted study

Run the README install/test/demo steps first. The benchmark requires the
original data described in [DATA.md](DATA.md). Do not use the accepted
`results/` directory as an output target. The original accepted numerical
exports are already available without access to the private dataset.

## Slurm configuration

The full runner requires Slurm, Linux and four CPU threads. It does not train
on a login node. Supply your site's partition/account; none are hardcoded.
`--gpus-per-node 1` is available if your site's partition requires a GPU
allocation for admission, but PathSelect still uses CPU and records that it
uses zero GPUs. GPU allocation is not otherwise needed.

```bash
export PATHSELECT_DATA_ROOT=/path/to/can_dataset
export PATHSELECT_SLURM_PARTITION=your_partition
export PATHSELECT_SLURM_ACCOUNT=your_account
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONNOUSERSITE=1
OUT="$PWD/outputs/navcil/local/paper_top64_reproduction"

# Source must be a clean tested Git commit. Each submission prints its job ID.
python scripts/submit_paper64.py --output "$OUT" --phase tests
python scripts/submit_paper64.py --output "$OUT" --phase preflight
```

Wait for the preflight receipt to pass before submitting smoke. A source
snapshot, fingerprint, environment metadata, checksummed data manifest and
expected 8,560 physical validation/test cells are recorded. Use one unchanged
source snapshot throughout the study. Local machine YAML files, if needed,
are ignored by Git; `PATHSELECT_DATA_ROOT` is sufficient for the standard layout.
The source and active environment must remain unchanged throughout a run.

```bash
# Independent namespaces, excluded from formal results.
python scripts/submit_paper64.py --output "$OUT" --phase smoke
# After smoke passes: full fold-1, five real training epochs, A seed42 and B,
# both orders, four stages, both readouts, mean ladder and top64/four16.
python scripts/submit_paper64.py --output "$OUT" --phase pilot
```

Smoke/pilot use sibling output directories. Inspect each receipt and log;
Slurm COMPLETED alone does not establish output validity. Pilot completion
registers `manifests/pilot_receipt.json`. The complete study trains A seeds
42–46 afresh; it does not count pilot observations as extra samples.

```bash
# After the full pilot passes; submit folds serially or with site-appropriate
# concurrency. Subsequent jobs are pinned to the established physical node.
# For multiple dependent batches, --dependency JOB_ID uses afterok semantics.
python scripts/submit_paper64.py --output "$OUT" --phase prepare --folds 1
# Repeat for folds 2,...,10, or pass --folds 1,2,3,4,5,6,7,8,9,10 if walltime permits.
# Every fold automatically includes shared deterministic evidence and A seeds42–46.

# After all folds finish: select using validation only and freeze aliases.
python scripts/submit_paper64.py --output "$OUT" --phase select
# After selection receipt passes: formal test, again for all ten folds.
python scripts/submit_paper64.py --output "$OUT" --phase test --folds 1
# Repeat for folds 2,...,10.

# After all prediction workers finish, time in isolation on the same node.
python scripts/submit_paper64.py --output "$OUT" --phase timing
# Reconstruct tables and paired statistics from the completed predictions.
python scripts/submit_paper64.py --output "$OUT" --phase report
```

`--source /path/to/immutable/snapshot` pins a recorded snapshot explicitly.
Use `--node NODE` on the first preparation job if matching a chosen compute
node is required. `--minutes` and `--memory-gb` control resource requests;
they must not be used to alter scientific settings. The historical timing
experiment used one physical CPU node, four threads and a 32 GiB allocation.

The report runner reconstructs every metric from predictions, verifies IDs,
class columns, shared evidence and order invariance, and writes mean/sample
SD, paired comparisons and both CI definitions. It can produce the internal
results without external baseline predictions, but then records
`external_complete=false` and does not declare the full paper accepted.

## Monitoring and recovery

```bash
mkdir -p "$OUT/logs"
tmux new-session -d -s pathselect-reproduction \
  "python scripts/monitor_paper64.py --output '$OUT' --interval 90 >> '$OUT/logs/monitor.log' 2>&1"
cat "$OUT/slurm/heartbeat.json"
# Verify sequence/time advances after a poll; inspect status.json and logs.
```

The monitor validates receipts/checksums and allows at most two automatic
NODE_FAIL/PREEMPTED retries per unit. Python errors, OOM, timeout and invalid
outputs become `NEEDS_AGENT`; they require diagnosis, not silent retries.
The monitor does not independently repair code or submit later scientific
phases. It stays active until a separately audited `DELIVERY.json` declares
acceptance; stop only this monitor session when your independent run is done.

Resume submits only missing/incomplete folds. Completed heads and artifacts
are reused only after provenance, IDs and checksum validation. Incomplete
heads restart all five epochs; no partial-epoch optimizer/RNG resume is
claimed. Do not launch duplicate concurrent copies of a unit. If submission
acknowledgement times out, query Slurm before resubmitting.

## Full archived-delivery audit

The original study also used `paper64_costs.py`, `paper64_accept.py` and
`paper64_delivery.py`. They retain the stricter full-delivery requirements:
external predictions, full deterministic pilot-repeat evidence, helper-source
parity records and Slurm accounting. These advanced audit tools are preserved
for readers with the complete original artifact set; the quick-start demo is
not a substitute for these requirements.

The source release does not redistribute the original 70,186-file delivery,
patient-level predictions, external checkpoints or machine-specific scheduler
ledger. See [PROVENANCE.md](PROVENANCE.md) for original source identities and
[EXTERNAL_BASELINE.md](EXTERNAL_BASELINE.md) for the independent rerun protocol.
