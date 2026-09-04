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
"""Tests various aggregate metrics."""

import numpy as np
import pytest
import scipy.stats

import rliable.metrics as metrics

# Score matrices are of the form num_runs x num_tasks.
_X = np.array([[1, 2], [2, 2], [1, 1], [2, 1]])
_Y = np.array([[1, 1], [2, 2], [3, 3]])


def _old_probability_of_improvement(scores_x, scores_y):
  """The pre-vectorization implementation, kept here as the reference."""
  num_tasks = scores_x.shape[1]
  task_improvement_probabilities = []
  num_runs_x, num_runs_y = scores_x.shape[0], scores_y.shape[0]
  for task in range(num_tasks):
    if np.array_equal(scores_x[:, task], scores_y[:, task]):
      task_improvement_prob = 0.5
    else:
      task_improvement_prob, _ = scipy.stats.mannwhitneyu(scores_x[:, task], scores_y[:, task], alternative="greater")
      task_improvement_prob /= num_runs_x * num_runs_y
    task_improvement_probabilities.append(task_improvement_prob)
  return np.mean(task_improvement_probabilities)


@pytest.mark.parametrize(
  "metric_fn",
  [metrics.aggregate_median, metrics.aggregate_mean, metrics.aggregate_iqm],
)
@pytest.mark.parametrize("scores, expected", [(_X, 1.5), (_Y, 2)])
def test_aggregate_metrics(metric_fn, scores, expected):
  assert metric_fn(scores) == expected


def test_probability_of_improvement_reference_value():
  assert metrics.probability_of_improvement(_X, _Y) == pytest.approx(1 / 3)


def test_optimality_gap_reference_value():
  # gamma - mean(min(scores, gamma)); all entries of _X are >= 1.
  assert metrics.aggregate_optimality_gap(_X, gamma=1) == 0.0
  assert metrics.aggregate_optimality_gap(_Y, gamma=2) == pytest.approx(2 - np.mean([1, 1, 2, 2, 2, 2]))


# (num_runs, num_tasks) shapes; (3, 5) makes N*M odd for the IQM trim check.
_BATCH_SHAPES = [(4, 6), (3, 5), (5, 3), (2, 2)]


@pytest.mark.parametrize("shape", _BATCH_SHAPES)
@pytest.mark.parametrize(
  "batched_fn, scalar_fn",
  [
    (metrics.aggregate_mean_batched, metrics.aggregate_mean),
    (metrics.aggregate_median_batched, metrics.aggregate_median),
    (metrics.aggregate_iqm_batched, metrics.aggregate_iqm),
    (metrics.aggregate_optimality_gap_batched, metrics.aggregate_optimality_gap),
  ],
)
def test_batched_metric_matches_scalar_loop(batched_fn, scalar_fn, shape):
  rng = np.random.default_rng(0)
  scores_RNM = rng.normal(size=(7,) + shape)
  expected = np.array([scalar_fn(scores_RNM[rep]) for rep in range(scores_RNM.shape[0])])
  np.testing.assert_allclose(batched_fn(scores_RNM), expected)


@pytest.mark.parametrize(
  "batched_fn, scalar_fn",
  [
    (metrics.aggregate_mean_batched, metrics.aggregate_mean),
    (metrics.aggregate_median_batched, metrics.aggregate_median),
  ],
)
def test_batched_mean_median_keep_trailing_axes(batched_fn, scalar_fn):
  # (R, N, M, L): the scalar form reduces over runs then tasks, so a per-step
  # axis survives and the batched form must return (R, L), not (R,).
  rng = np.random.default_rng(5)
  num_reps, num_steps = 6, 3
  scores_RNML = rng.normal(size=(num_reps, 4, 5, num_steps))
  expected = np.stack([scalar_fn(scores_RNML[rep]) for rep in range(num_reps)])
  assert expected.shape == (num_reps, num_steps)
  result = batched_fn(scores_RNML)
  assert result.shape == (num_reps, num_steps)
  np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_optimality_gap_batched_gamma(gamma):
  rng = np.random.default_rng(1)
  scores_RNM = rng.normal(loc=1.0, size=(6, 4, 5))
  expected = np.array([metrics.aggregate_optimality_gap(s, gamma=gamma) for s in scores_RNM])
  np.testing.assert_allclose(metrics.aggregate_optimality_gap_batched(scores_RNM, gamma=gamma), expected)


@pytest.mark.parametrize(
  "fn",
  [
    metrics.aggregate_mean_batched,
    metrics.aggregate_median_batched,
    metrics.aggregate_iqm_batched,
    metrics.aggregate_optimality_gap_batched,
    metrics.probability_of_improvement_batched,
  ],
)
def test_batched_functions_are_marked(fn):
  assert getattr(fn, "_rliable_batched", False) is True


def test_unbatched_functions_are_not_marked():
  assert not hasattr(metrics.aggregate_mean, "_rliable_batched")
  assert not hasattr(metrics.probability_of_improvement, "_rliable_batched")


@pytest.mark.parametrize("num_runs_x, num_runs_y, num_tasks", [(5, 5, 4), (3, 7, 6), (4, 4, 1)])
def test_probability_of_improvement_matches_old_loop(num_runs_x, num_runs_y, num_tasks):
  rng = np.random.default_rng(2)
  # Small integer support guarantees ties, which is where the midrank
  # correction in the Mann-Whitney U statistic matters.
  scores_x = rng.integers(0, 4, size=(num_runs_x, num_tasks)).astype(float)
  scores_y = rng.integers(0, 4, size=(num_runs_y, num_tasks)).astype(float)
  np.testing.assert_allclose(
    metrics.probability_of_improvement(scores_x, scores_y),
    _old_probability_of_improvement(scores_x, scores_y),
  )


def test_probability_of_improvement_identical_inputs_is_half():
  rng = np.random.default_rng(3)
  scores = rng.normal(size=(6, 5))
  assert metrics.probability_of_improvement(scores, scores) == 0.5


def test_probability_of_improvement_constant_columns_is_half():
  # All-identical values used to make older SciPy raise; the explicit branch
  # (and the formula) return 0.5 here.
  scores = np.ones((4, 3))
  assert metrics.probability_of_improvement(scores, scores) == 0.5


@pytest.mark.parametrize("num_runs_x, num_runs_y", [(5, 5), (3, 7)])
def test_probability_of_improvement_batched_matches_loop(num_runs_x, num_runs_y):
  rng = np.random.default_rng(4)
  num_reps, num_tasks = 8, 5
  scores_x_RNxM = rng.integers(0, 4, size=(num_reps, num_runs_x, num_tasks)).astype(float)
  scores_y_RNyM = rng.integers(0, 4, size=(num_reps, num_runs_y, num_tasks)).astype(float)
  expected = np.array([metrics.probability_of_improvement(scores_x_RNxM[rep], scores_y_RNyM[rep]) for rep in range(num_reps)])
  np.testing.assert_allclose(metrics.probability_of_improvement_batched(scores_x_RNxM, scores_y_RNyM), expected)
