# Provenance and release boundary

## Accepted experiment

| Item | Identity |
|---|---|
| Study | `paper_top64_20261016` |
| Upstream project | https://github.com/nsysu-visualAI/navrouter-cil |
| Verified upstream base | `f5813cd81c845ff11549d69615c32ede6d32c971` |
| Formal training implementation | `52c92fac996b0f72e121674b977869204bff42c1` |
| Formal source SHA256 fingerprint | `a388cffd63e1f63cc66efaf5c02efbb736eb46f79451916886a21903f69072af` |
| Final analysis/export source | `c390ed16f56b1ccbbd250e6f51fd85d08e59777f` |
| Original sealed DELIVERY.json SHA256 | `f559aa74188bdf17aec8f75684ccc757a180c023aa1ddae4578f3b2d923ab6f4` |
| Acceptance | 200 I6 heads; 720 evidence artifacts; 3,680 validation cells; 4,880 test cells; 80 external stages |
| Dataset | 10 folds; 2 orders; 4 stages; 2,835 test slides; 2,645 test patients |
| Original final test suite | 193 passed, 2 skipped (historical Mac-cache-only comparisons) |

The original sealed experiment remains separate and unchanged. This repository
exports aggregate reports byte-for-byte, with per-file hashes in
`results/paper_top64/PROVENANCE.json`. Those are measured results, not the
output of the release's synthetic demonstration. The original source commits
identify the authors' research checkout; they are not necessarily reachable
from the public upstream main branch. This new repository provides the curated
code and its own independent public history.

`docs/exported_source.json` records each source file's hash before publication
changes. `docs/scientific_core.json` additionally fixes 13 complete scientific
files plus the common selection/readout functions. Tests verify those source
bytes against the accepted implementation. Function-source hashes are used
instead of Python-version-dependent AST serialization.

## Compute and inference timing

Internal experiments used one physical server node, CPU PyTorch 2.11.0,
four OMP/MKL/OpenBLAS threads and 32 GiB requested memory. A GPU was allocated
only to satisfy the site's partition admission rule; internal methods did not
use it. The independent external baseline used a separate GPU environment,
explicitly identified in its protocol.

Timing used all 279 fold-1 test slides, A seed42, final-stage reverse order,
fixed gamma `1e-3`, three interleaved repetitions per slide and three discarded
warm-ups per variant/task. Each timing pass computed routing, all-patch scores,
selection, pooling and readout from raw patch features. Cached evidence was
used only for post-timer parity. File reads were shared across interleaved
variants, the OS cache was not flushed, and compute order rotated.

Read + compute quantiles are calculated from per-observation sums, then
per-slide repeat medians; component P95s are never added. Timings exclude
CONCH feature extraction, integrity scanning and oracle diagnostics. Shared
process peak memory must not be attributed to individual variants.

## What is intentionally absent

The publication export excludes raw/patient-level data, per-slide predictions,
training checkpoints, private machine paths and configs, authentication files,
historical Mac result caches, paper drafts, external implementation code and
CONCH model weights. Hash references and aggregate tables are retained.
Full-data reproduction therefore requires the original authorized dataset.
The public source is useful independently through its tested synthetic demo
and minimal API.
