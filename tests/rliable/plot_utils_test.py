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
"""Smoke tests for the plotting utilities.

The plots themselves are not inspected; the tests check that each public
`plot_*` function runs end to end on realistic bootstrap output, returns the
documented matplotlib objects, and does not emit a matplotlib or seaborn
deprecation warning (the module-level filter turns those into errors).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rliable import library as rly  # noqa: E402
from rliable import (
  metrics,  # noqa: E402
  plot_utils,  # noqa: E402
)

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")

_ALGORITHMS = ["algo_a", "algo_b", "algo_c"]
_NUM_RUNS = 5
_NUM_TASKS = 8
_TAU_LIST = np.linspace(0.0, 1.0, 21)


def _score_dict():
  """Three algorithms, each a (num_runs x num_tasks) matrix of scores in [0, 1]."""
  rng = np.random.default_rng(0)
  return {name: rng.random((_NUM_RUNS, _NUM_TASKS)) for name in _ALGORITHMS}


@pytest.fixture(scope="module")
def aggregate_estimates():
  """Point estimates and CIs for the (mean, IQM) pair of aggregate metrics."""

  @metrics.batched
  def aggregate_func(scores_RNM):
    return np.stack(
      [metrics.aggregate_mean_batched(scores_RNM), metrics.aggregate_iqm_batched(scores_RNM)],
      axis=-1,
    )

  return rly.get_interval_estimates(_score_dict(), aggregate_func, reps=200, random_state=0)


@pytest.fixture(scope="module")
def performance_profiles():
  return rly.create_performance_profile(_score_dict(), _TAU_LIST, reps=100, random_state=0)


@pytest.fixture(scope="module")
def improvement_estimates():
  """Probability of improvement for two algorithm pairs."""
  scores = _score_dict()
  pairs = {
    "algo_a,algo_b": [scores["algo_a"], scores["algo_b"]],
    "algo_a,algo_c": [scores["algo_a"], scores["algo_c"]],
  }
  return rly.get_interval_estimates(pairs, metrics.probability_of_improvement, reps=100, random_state=0)


@pytest.fixture(scope="module")
def sample_efficiency_estimates():
  """Frames plus per-algorithm curves: (num_frames,) points and (2, num_frames) CIs."""
  rng = np.random.default_rng(0)
  num_frames = 6
  frames = np.arange(1, num_frames + 1)
  point_estimates, interval_estimates = {}, {}
  for name in _ALGORITHMS:
    curve = np.sort(rng.random(num_frames))
    point_estimates[name] = curve
    interval_estimates[name] = np.stack([curve - 0.05, curve + 0.05])
  return frames, point_estimates, interval_estimates


def test_plot_performance_profiles(performance_profiles):
  profiles, profile_cis = performance_profiles
  ax = plot_utils.plot_performance_profiles(profiles, _TAU_LIST, performance_profile_cis=profile_cis)
  assert isinstance(ax, plt.Axes)
  assert len(ax.get_lines()) == len(_ALGORITHMS)
  plt.close("all")


def test_plot_performance_profiles_non_linear_scaling(performance_profiles):
  profiles, profile_cis = performance_profiles
  ax = plot_utils.plot_performance_profiles(
    profiles,
    _TAU_LIST,
    performance_profile_cis=profile_cis,
    use_non_linear_scaling=True,
  )
  assert isinstance(ax, plt.Axes)
  plt.close("all")


def test_plot_performance_profiles_linewidth_applies_to_every_line(performance_profiles):
  """Regression test: `linewidth` used to be popped inside the per-method loop,
  so only the first method got the custom value."""
  profiles, _ = performance_profiles
  ax = plot_utils.plot_performance_profiles(profiles, _TAU_LIST, linewidth=5.0)
  linewidths = [line.get_linewidth() for line in ax.get_lines()]
  assert len(linewidths) == len(_ALGORITHMS)
  assert all(width == 5.0 for width in linewidths)
  plt.close("all")


def test_plot_interval_estimates(aggregate_estimates):
  point_estimates, interval_estimates = aggregate_estimates
  fig, axes = plot_utils.plot_interval_estimates(
    point_estimates, interval_estimates, metric_names=["Mean", "IQM"], algorithms=_ALGORITHMS
  )
  assert isinstance(fig, plt.Figure)
  assert len(axes) == 2
  assert all(isinstance(ax, plt.Axes) for ax in axes)
  plt.close("all")


def test_plot_interval_estimates_single_metric(aggregate_estimates):
  """With one metric `plt.subplots` returns a bare Axes rather than an array."""
  point_estimates, interval_estimates = aggregate_estimates
  single_point = {key: value[:1] for key, value in point_estimates.items()}
  single_interval = {key: value[:, :1] for key, value in interval_estimates.items()}
  fig, axes = plot_utils.plot_interval_estimates(single_point, single_interval, metric_names=["Mean"])
  assert isinstance(fig, plt.Figure)
  assert isinstance(axes, plt.Axes)
  plt.close("all")


def test_plot_sample_efficiency_curve(sample_efficiency_estimates):
  frames, point_estimates, interval_estimates = sample_efficiency_estimates
  ax = plot_utils.plot_sample_efficiency_curve(frames, point_estimates, interval_estimates, algorithms=_ALGORITHMS)
  assert isinstance(ax, plt.Axes)
  assert len(ax.get_lines()) == len(_ALGORITHMS)
  plt.close("all")


def test_plot_probability_of_improvement(improvement_estimates):
  probabilities, probability_cis = improvement_estimates
  ax = plot_utils.plot_probability_of_improvement(probabilities, probability_cis)
  assert isinstance(ax, plt.Axes)
  plt.close("all")
