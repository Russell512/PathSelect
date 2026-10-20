# Publication changes

Release `v1.0.0` packages the accepted one-shot top-64 study for public review.

- New independent Git history; no changes to the upstream repository or sealed study.
- Retained original scientific selection, residual-head training, incremental
  ridge, prediction, gamma selection, statistics and timing implementations.
  File/function checksums enforce parity in the test suite.
- Added `pathselect.PathSelect` and a data-free, four-stage synthetic demo.
  The API adds CPU conversion, input validation, frozen inference state and
  checkpoint round-tripping around the same numerical primitives. Its
  all-selected shortcut avoids an undefined score standard deviation for a
  single-patch WSI; benchmark data and original scientific functions are unchanged.
- Replaced the eager CONCH dependency with a frozen-cache reader. No foundation
  model, tokenizer, weights, transformers or external baseline code is vendored.
- Replaced historical Mac cache/ID references in preflight with the same
  aggregate split/ID/text hashes; added direct table/split byte-hash checks.
- Replaced site-specific account, partition, Python and machine settings with
  CLI/environment configuration. Formal runs still require compute allocations,
  immutable snapshots and one physical CPU backend.
- Added explicit preflight/manifest/freeze phases and registered pilot receipts.
- Kept report/audit tools separate from small inference usage, and documented
  which advanced delivery checks require the full original private artifacts.
- Exported 31 aggregate report files unchanged; retained literature row labels
  and references. No numerical result was recalculated or selected for this release.
- Removed obsolete dependency-test exceptions for code that is not shipped;
  expanded the scan to the new public API. Added a check that foundation-model
  code is not vendored. The prohibited dependency policy was not relaxed.
- Added source/result checksum tests, physical-manifest checks, API parity,
  serialization/resume tests and GitHub CI configuration.

The original study passed 193 tests with two historical Mac-cache skips.
The curated release has its own smaller, relevant suite; its pass count must
not be described as rerunning all 193 original tests. It does not rerun the
entire accepted training study merely to reformat the repository.
