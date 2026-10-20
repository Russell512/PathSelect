# Publication validation (2026-10-20)

The release was checked on a compute node using the recorded CPU environment.
This is validation of the code export and portable entry points, not a new
training batch substituted into the accepted paper tables.

- **78 tests passed**. The suite covers original one-shot/four-round selection,
  raw-feature pooling, deterministic random subsets, incremental ridge against
  a batch reference, seen-class mapping, separation of routed/oracle evidence,
  seed initialization, sign tests, both bootstrap definitions, source parity,
  result checksums, API evidence parity and save/load/continued-training behavior.
- **13 scientific source files** and **7 common scientific functions** match
  the accepted implementation, verified by SHA256. All 31 exported aggregate
  report files match their original bytes.
- All **8,560 physical validation/test configurations** match the accepted
  experiment. The release represents absent cosine gamma as JSON null instead
  of the historical string `"none"`; prediction paths/configuration IDs match.
- The new preflight successfully checked **120 split cells, 2,835 feature files,
  2,835 test slides and 2,645 patients**, with zero issues. Data were read only.
- A separate **real-data smoke** completed both orders/four stages, A/B,
  ridge/cosine, mean ladder and top-64/four-round paths. Smoke uses one epoch
  and two training slides per task; it is excluded from formal results.
- The synthetic quick-start completed both orders and all four stages.
- Public-file checks found no actual TCGA patient IDs, private server/account
  paths or credential patterns; all relative documentation links resolved.
- Original upstream worktree and sealed study were preserved.

The original accepted full study's 193-pass/two-skip test record is documented
separately in [PROVENANCE.md](PROVENANCE.md). The release smoke is not claimed
to reproduce its 200 fully trained heads or all formal predictions.

One initial release check failed because AST serialization differs between
Python versions. The check was corrected to hash exact function source text,
while full-file hashes remained unchanged. Scientific code and expected
accuracies were not changed to pass this check.
