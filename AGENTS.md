# Contribution rules

- Preserve the one-shot top-64 FINAL-B main method and the documented metric definitions.
- Run `python -m pytest -q` before every commit. Investigate failures; never change expected scientific results to pass.
- Keep data, patient/slide identifiers, checkpoints, credentials, private site settings and generated run directories out of Git.
- Keep external baseline implementations in separate checkouts. `tests/test_no_banned_deps.py` must pass.
- Use float32 features/heads and CPU float64 ridge statistics/solves.
- Treat `results/paper_top64/` as immutable exports from the accepted study. New runs use a fresh output directory; never overwrite those results.
- Use compute allocations for long real-data work on shared clusters. Do not terminate other users' jobs.
- Record changes to scientific functions and run parity checks against `docs/scientific_core.json`.
- Source availability does not grant a new license over upstream work; retain `NOTICE.md`.
