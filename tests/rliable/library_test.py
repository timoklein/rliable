# coding=utf-8
# Copyright 2021 The Rliable Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests functionality of library functions.

Dimension key:
  R: bootstrap reps    N: runs    M: tasks    L: training steps
  T: profile thresholds           K: statistics returned by `func`
"""

import numpy as np
import pytest

import rliable.library as rly
import rliable.metrics as metrics

NUM_RUNS, NUM_TASKS = 5, 16


@pytest.fixture
def x() -> np.ndarray:
  """Deterministic `(N, M)` score matrix; entry `[n, m]` is `n * M + m`."""
  return np.arange(NUM_RUNS * NUM_TASKS).reshape(NUM_RUNS, NUM_TASKS)


@pytest.fixture
def y(x: np.ndarray) -> np.ndarray:
  """`x` with the run order reversed, used as a second algorithm."""
  return np.flip(x, axis=0)


@pytest.fixture
def z(x: np.ndarray) -> np.ndarray:
  """`(N, M, 2)` scores, used to check the per-cell 3-D semantics."""
  return np.stack([x, 2 * x], axis=-1)


def _metric_func(scores: np.ndarray) -> np.ndarray:
  """The four aggregate metrics, as a `(4,)` array."""
  return np.array(
    [
      metrics.aggregate_iqm(scores),
      metrics.aggregate_mean(scores),
      metrics.aggregate_median(scores),
      metrics.aggregate_optimality_gap(scores),
    ]
  )


def _iqm(flat_scores: np.ndarray) -> float:
  """25% trimmed mean of a flat array, matching `scipy.stats.trim_mean`."""
  ordered = np.sort(flat_scores)
  cut = int(0.25 * ordered.size)
  return ordered[cut : ordered.size - cut].mean()


def _mean_and_iqm(scores: np.ndarray) -> np.ndarray:
  """Per-replication `(2,)` statistic used by the batched-vs-unbatched test."""
  flat = np.asarray(scores).reshape(-1)
  return np.array([flat.mean(), _iqm(flat)])


def _mean_and_iqm_batched(scores: np.ndarray) -> np.ndarray:
  """Batched `_mean_and_iqm`; takes `(R, N, M)` and returns `(R, 2)`."""
  flat = np.asarray(scores).reshape(scores.shape[0], -1)
  ordered = np.sort(flat, axis=1)
  cut = int(0.25 * flat.shape[1])
  trimmed = ordered[:, cut : flat.shape[1] - cut]
  return np.stack([flat.mean(axis=1), trimmed.mean(axis=1)], axis=1)


_mean_and_iqm_batched._rliable_batched = True


####################### Preserved tests #######################
@pytest.mark.parametrize(
  "method, task_bootstrap",
  [("percentile", False), ("basic", True), ("bc", False), ("bca", False)],
)
def test_interval_estimation(x, y, method, task_bootstrap):
  """Point estimates are exact and the CIs bracket them."""
  score_dict = {"method": x, "baseline": y}
  # Use a small number of bootstrap samples for testing.
  point_estimates, interval_estimates = rly.get_interval_estimates(
    score_dict,
    _metric_func,
    method=method,
    task_bootstrap=task_bootstrap,
    reps=100,
    random_state=0,
  )
  for key, scores in score_dict.items():
    lower_ci_estimates, upper_ci_estimates = interval_estimates[key]
    np.testing.assert_array_equal(point_estimates[key], _metric_func(scores))
    np.testing.assert_array_less(lower_ci_estimates, point_estimates[key])
    np.testing.assert_array_less(point_estimates[key], upper_ci_estimates)


@pytest.mark.parametrize("use_score_distribution", [True, False])
def test_performance_profiles(x, y, use_score_distribution):
  """Profiles are decreasing in tau and their CIs have shape `(2, T)`."""
  tau_list = [0.0, 0.25, 0.5, 1.0]
  score_dict = {"x": x, "y": y}
  profiles, profile_cis = rly.create_performance_profile(
    score_dict,
    tau_list,
    use_score_distribution=use_score_distribution,
    reps=100,
    random_state=0,
  )
  for key in score_dict:
    assert isinstance(profiles[key], np.ndarray)
    np.testing.assert_array_equal(profiles[key], sorted(profiles[key], reverse=True))
    assert len(profiles[key]) == len(tau_list)
    assert profile_cis[key].shape == (2, len(tau_list))


def test_performance_profiles_match_scalar_helpers(x):
  """The vectorized profiles agree with the scalar per-tau helpers."""
  tau_list = np.linspace(0, 60, 11)
  np.testing.assert_allclose(
    rly.score_distributions(x, tau_list),
    [rly.run_score_deviation(x, tau) for tau in tau_list],
  )
  np.testing.assert_allclose(
    rly.average_score_distributions(x, tau_list),
    [rly.mean_score_deviation(x, tau) for tau in tau_list],
  )
  # The batched variants operate on a leading reps axis.
  batch = np.stack([x, 2 * x])
  np.testing.assert_allclose(
    rly.score_distributions_batched(batch, tau_list),
    [rly.score_distributions(scores, tau_list) for scores in batch],
  )
  np.testing.assert_allclose(
    rly.average_score_distributions_batched(batch, tau_list),
    [rly.average_score_distributions(scores, tau_list) for scores in batch],
  )


def test_improvement_probability_cis(x, y):
  """Metrics taking two score arrays route through the independent bootstrap."""
  score_dict = {"x,y": [x, y]}
  point_estimates, interval_estimates = rly.get_interval_estimates(
    score_dict, metrics.probability_of_improvement, reps=100, random_state=0
  )
  for key, scores in score_dict.items():
    lower_ci_estimates, upper_ci_estimates = interval_estimates[key]
    np.testing.assert_array_equal(point_estimates[key], metrics.probability_of_improvement(*scores))
    np.testing.assert_array_less(lower_ci_estimates, point_estimates[key])
    np.testing.assert_array_less(point_estimates[key], upper_ci_estimates)


@pytest.mark.parametrize("task_bootstrap", [False, True])
def test_stratified_bootstrap_indices(x, y, task_bootstrap):
  """`bootstrap` yields exactly `array[bs.index]` for every positional array."""
  bs = rly.StratifiedBootstrap(x, y, task_bootstrap=task_bootstrap, random_state=0)
  num_yielded = 0
  for data, kw_data in bs.bootstrap(5):
    index = bs.index
    assert len(data) == 2
    assert kw_data == {}
    np.testing.assert_array_equal(x[index], data[0])
    np.testing.assert_array_equal(y[index], data[1])
    assert data[0].shape == x.shape
    num_yielded += 1
  assert num_yielded == 5


def test_stratified_independent_bootstrap_indices(x, y):
  """The independent bootstrap keeps one index tuple per positional array."""
  bs = rly.StratifiedIndependentBootstrap(x, y, random_state=0)
  for data, kw_data in bs.bootstrap(2):
    assert isinstance(bs.index, list)
    index_x, index_y = bs.index
    assert len(data) == 2
    assert kw_data == {}
    np.testing.assert_array_equal(x[index_x], data[0])
    np.testing.assert_array_equal(y[index_y], data[1])


####################### Reference-loop equivalence #######################
def _naive_percentile_ci(scores: np.ndarray, indices, func, size: float = 0.95) -> np.ndarray:
  """Reference CI: a plain Python loop over already-drawn indices."""
  batch = scores[indices]
  results = np.array([np.asarray(func(rep)).reshape(-1) for rep in batch])
  alpha = (1.0 - size) / 2.0
  lower, upper = np.percentile(results, [100 * alpha, 100 * (1 - alpha)], axis=0)
  return np.vstack((lower, upper))


def test_percentile_ci_matches_reference_loop_exactly(x):
  """Same drawn indices => bit-identical percentile interval."""
  reps = 200
  bs_reference = rly.StratifiedBootstrap(x, random_state=np.random.default_rng(123))
  indices = bs_reference.sample_indices(reps)
  reference = _naive_percentile_ci(x, indices, _metric_func)

  bs = rly.StratifiedBootstrap(x, random_state=np.random.default_rng(123))
  interval = bs.conf_int(_metric_func, reps=reps, method="percentile")
  np.testing.assert_allclose(interval, reference, rtol=0, atol=0)


def test_percentile_ci_matches_reference_loop_statistically(x):
  """Independent seeds at reps=20000 agree within 3% of the CI width."""
  reps = 20000
  bs = rly.StratifiedBootstrap(x, random_state=np.random.default_rng(1))
  interval = bs.conf_int(_metric_func, reps=reps, method="percentile")

  rng = np.random.default_rng(2)
  strata = np.arange(NUM_TASKS)[None, :]
  results = []
  for _ in range(reps):
    run_indices = rng.integers(0, NUM_RUNS, size=x.shape)
    results.append(_metric_func(x[run_indices, strata]))
  results = np.array(results)
  reference = np.vstack(np.percentile(results, [2.5, 97.5], axis=0))

  width = reference[1] - reference[0]
  tolerance = np.broadcast_to(0.03 * width, interval.shape)
  np.testing.assert_array_less(np.abs(interval - reference), tolerance)


####################### Batched vs unbatched #######################
@pytest.mark.parametrize("method", ["percentile", "bc", "bca"])
def test_batched_matches_unbatched(method):
  """A batched `func` gives identical intervals to its per-rep twin."""
  scores = np.random.default_rng(0).normal(size=(6, 9))
  unbatched = rly.StratifiedBootstrap(scores, random_state=7).conf_int(_mean_and_iqm, reps=300, method=method)
  batched = rly.StratifiedBootstrap(scores, random_state=7).conf_int(_mean_and_iqm_batched, reps=300, method=method)
  np.testing.assert_allclose(batched, unbatched, rtol=0, atol=0)


def test_batched_point_estimate_matches_unbatched():
  """`get_interval_estimates` unwraps the batch axis for the point estimate."""
  scores = np.random.default_rng(0).normal(size=(6, 9))
  point_batched, _ = rly.get_interval_estimates({"a": scores}, _mean_and_iqm_batched, reps=50, random_state=0)
  np.testing.assert_allclose(point_batched["a"], _mean_and_iqm(scores))


####################### Random state handling #######################
def test_as_generator_accepts_supported_types():
  assert isinstance(rly._as_generator(None), np.random.Generator)
  assert isinstance(rly._as_generator(3), np.random.Generator)
  assert isinstance(rly._as_generator(np.int64(3)), np.random.Generator)
  generator = np.random.default_rng(0)
  assert rly._as_generator(generator) is generator
  legacy = np.random.RandomState(0)
  assert rly._as_generator(legacy) is legacy


def test_as_generator_rejects_other_types():
  with pytest.raises(TypeError):
    rly._as_generator("abc")
  with pytest.raises(TypeError):
    rly.get_interval_estimates({"a": np.ones((3, 4))}, np.mean, reps=10, random_state="abc")


@pytest.mark.parametrize("random_state", [0, np.random.default_rng(0), np.random.RandomState(0)])
def test_random_state_flavours_run(x, random_state):
  _, interval_estimates = rly.get_interval_estimates({"a": x}, _metric_func, reps=50, random_state=random_state)
  assert interval_estimates["a"].shape == (2, 4)


def test_same_seed_is_reproducible(x):
  kwargs = dict(reps=200, random_state=1234)
  _, first = rly.get_interval_estimates({"a": x}, _metric_func, **kwargs)
  _, second = rly.get_interval_estimates({"a": x}, _metric_func, **kwargs)
  np.testing.assert_allclose(first["a"], second["a"], rtol=0, atol=0)


def test_one_seed_gives_different_draws_per_key(x):
  """The generator is shared across keys, so identical data still differs."""
  _, intervals = rly.get_interval_estimates({"a": x, "b": x.copy()}, _metric_func, reps=200, random_state=1234)
  assert not np.allclose(intervals["a"], intervals["b"])


####################### Stratification semantics #######################
def test_resample_keeps_each_column_in_its_task(x):
  """Column `m` of a resample only ever draws from column `m` of the source."""
  bs = rly.StratifiedBootstrap(x, random_state=0)
  (resampled,) = bs.resample(7)
  assert resampled.shape == (7, NUM_RUNS, NUM_TASKS)
  tasks = np.broadcast_to(np.arange(NUM_TASKS), resampled.shape)
  np.testing.assert_array_equal(resampled % NUM_TASKS, tasks)


def test_task_bootstrap_draws_whole_columns(x):
  """With `task_bootstrap`, every column of a resample is a source column."""
  bs = rly.StratifiedBootstrap(x, task_bootstrap=True, random_state=0)
  (resampled,) = bs.resample(20)
  # A column drawn from source task `m` has every entry congruent to `m`.
  source_tasks = resampled % NUM_TASKS
  for rep in range(resampled.shape[0]):
    for task in range(NUM_TASKS):
      column = source_tasks[rep, :, task]
      assert len(np.unique(column)) == 1
  # Runs are still resampled, so at least one column repeats a source task.
  assert not np.array_equal(source_tasks[:, 0, :], np.broadcast_to(np.arange(NUM_TASKS), (20, NUM_TASKS)))


def test_three_dimensional_scores_keep_per_cell_semantics():
  """For `(N, M, L)` scores each `(task, step)` cell draws its own run."""
  num_runs, num_tasks, num_steps = 4, 3, 2
  scores = np.arange(num_runs * num_tasks * num_steps).reshape(num_runs, num_tasks, num_steps)
  bs = rly.StratifiedBootstrap(scores, random_state=0)
  (resampled,) = bs.resample(5)
  assert resampled.shape == (5, num_runs, num_tasks, num_steps)
  steps = np.broadcast_to(np.arange(num_steps), resampled.shape)
  tasks = np.broadcast_to(np.arange(num_tasks)[:, None], resampled.shape)
  np.testing.assert_array_equal(resampled % num_steps, steps)
  np.testing.assert_array_equal((resampled // num_steps) % num_tasks, tasks)


####################### Degenerate and edge cases #######################
@pytest.mark.parametrize("method", ["bc", "bca"])
def test_constant_scores_warn_and_stay_finite(method):
  """Constant scores make the bias correction degenerate; warn, don't raise."""
  scores = np.ones((5, 4))
  with pytest.warns(RuntimeWarning):
    _, interval_estimates = rly.get_interval_estimates({"a": scores}, np.mean, method=method, reps=100, random_state=0)
  assert np.all(np.isfinite(interval_estimates["a"]))
  np.testing.assert_allclose(interval_estimates["a"], np.ones((2, 1)))


def test_scalar_func_yields_two_by_one_interval(x):
  point_estimates, interval_estimates = rly.get_interval_estimates({"a": x}, np.mean, reps=100, random_state=0)
  assert np.ndim(point_estimates["a"]) == 0
  assert interval_estimates["a"].shape == (2, 1)


def test_per_step_metric_on_three_dimensional_scores():
  """`(N, M, L)` scores with a per-step statistic give a `(2, L)` interval."""
  num_runs, num_tasks, num_steps = 6, 5, 3
  scores = np.random.default_rng(0).normal(size=(num_runs, num_tasks, num_steps))

  def per_step_iqm(step_scores):
    return np.array([_iqm(step_scores[..., step].reshape(-1)) for step in range(num_steps)])

  point_estimates, interval_estimates = rly.get_interval_estimates({"a": scores}, per_step_iqm, reps=100, random_state=0)
  assert point_estimates["a"].shape == (num_steps,)
  assert interval_estimates["a"].shape == (2, num_steps)


def test_independent_bootstrap_with_unequal_run_counts():
  """Unequal run counts resample fine but cannot be jackknifed."""
  rng = np.random.default_rng(0)
  scores_x = rng.normal(size=(7, 4))
  scores_y = rng.normal(size=(3, 4))
  bs = rly.StratifiedIndependentBootstrap(scores_x, scores_y, random_state=0)
  interval = bs.conf_int(metrics.probability_of_improvement, reps=100, method="percentile")
  assert interval.shape == (2, 1)

  bs = rly.StratifiedIndependentBootstrap(scores_x, scores_y, random_state=0)
  with pytest.raises(ValueError):
    bs.conf_int(metrics.probability_of_improvement, reps=100, method="bca")


def test_invalid_method_and_size(x):
  bs = rly.StratifiedBootstrap(x, random_state=0)
  with pytest.raises(ValueError):
    bs.conf_int(np.mean, reps=10, method="not-a-method")
  with pytest.raises(ValueError):
    bs.conf_int(np.mean, reps=10, method="percentile", size=1.5)
