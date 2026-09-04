# Changelog

All notable changes to this fork of `rliable` are documented here.

## 2.0.0 (2026-09-04)

### Removed
- The `arch` dependency (the bootstrap no longer uses it).
- The `absl` dependency (`absl-py`, `absltest`).
- `setup.py` (packaging is now `pyproject.toml` + setuptools build backend).
- The PyPI publish workflow (`python-publish.yml`) — the `rliable` name on PyPI belongs to the
  archived upstream, so this fork is distributed by git tag only.
- Keyword data arrays (`y=`, `z=`, ...) on the bootstrap classes. Pass extra score arrays
  positionally instead.

### Changed
- The bootstrap is implemented in pure numpy/scipy, with the same resampling semantics as the
  previous `arch`-based implementation.
- `random_state` now actually seeds the bootstrap. Previously it was accepted but silently
  ignored, so results were not reproducible. It accepts an `int`, a `numpy.random.Generator`, a
  `numpy.random.RandomState`, or `None`.
- `probability_of_improvement` is vectorized.
- Performance profiles (`create_performance_profile`) are vectorized.
- Minimum supported Python is now 3.10 (was 3.7).
- Packaging moved to `pyproject.toml`.
- Test suite runs under CI across the supported Python matrix.

### Added
- `metrics.batched`, a decorator marking a statistic function as accepting a whole chunk of
  bootstrap replications at once, plus batched counterparts of the aggregate metrics
  (`aggregate_mean_batched`, `aggregate_median_batched`, `aggregate_iqm_batched`,
  `aggregate_optimality_gap_batched`, `probability_of_improvement_batched`).
- `rliable.data`, with `load_scores_from_dataframe` and `load_score_dict_from_dataframe` for
  building score arrays/dicts out of long-format DataFrames.
- `random_state` argument on `create_performance_profile`.
- `chunk_size` and `batched` arguments on `get_interval_estimates`, controlling how many bootstrap
  replications are computed per call.
- `rliable.__version__`.

### Fixed
- `method='bca'` with several statistics at once: `arch` computed the jackknife acceleration
  from the mean over *all* statistics instead of per statistic, biasing every column whenever the
  statistics have different scales. 2.0.0 uses the standard per-statistic acceleration; for a
  single statistic the two agree. Percentile, basic and bc intervals are unchanged (verified
  against arch 7.2 within 1% of the CI width at 50k reps).
- `plot_performance_profiles` popped `linewidth` from `kwargs` inside the per-method loop, so only
  the first method received a custom line width.
- Various numpy/matplotlib API usages tightened for compatibility with numpy 2 and current
  matplotlib.

## Performance

Scores of shape 10 runs x 26 tasks, median of 3 runs on one CPU core. "before" is upstream HEAD
with arch 7.2; "after" is 2.0.0 with the same per-rep `func`; "after, batched" uses
`metrics.batched` with the `*_batched` metrics.

| configuration | 4 aggregates, 50k reps | same, BCa | performance profile, 81 thresholds, 2k reps | probability of improvement, 2k reps |
|---|---|---|---|---|
| before (arch 7.2) | 10.85 s | 10.83 s | 1.10 s | 23.7 s |
| after | 8.98 s | 9.42 s | 0.04 s | 0.06 s |
| after, batched | 0.39 s | 0.40 s | 0.04 s | 0.06 s |

The unbatched path is still dominated by the per-rep `func` call; the batched path removes it.
