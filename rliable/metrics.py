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
"""Aggregate Performance Estimators.

Dimension key:
  R: bootstrap replications   N: runs (per algorithm)   M: tasks

Every metric comes in two flavours. The plain form takes a single score matrix
of shape (N, M, ...); the `_batched` form takes a stack of R such matrices,
shape (R, N, M, ...), and returns the same thing per replication along a
leading axis of length R. Each batched form mirrors its scalar counterpart's
reduction exactly: mean and median reduce over runs and then tasks, so trailing
axes (e.g. training steps) survive and the result is (R, ...); IQM and
optimality gap reduce over everything (their scalar forms use `axis=None`), so
the result is always (R,).

The batched forms exist so the bootstrap can evaluate thousands of resamples
with a handful of numpy calls instead of a Python loop; they are marked with
the `batched` decorator so the bootstrap can detect them.
"""

from collections.abc import Callable

import numpy as np
import scipy.stats


def batched(func: Callable) -> Callable:
  """Marks `func` as accepting a leading bootstrap-replication axis.

  The bootstrap inspects this attribute to decide whether it may hand a whole
  (R, N, M, ...) chunk to `func` at once (expecting an (R,) or (R, K) result)
  or has to fall back to calling `func` once per replication. Using an
  attribute rather than a registry keeps user-defined metrics opt-in without
  importing anything from the bootstrap module.
  """
  func._rliable_batched = True  # pylint: disable=protected-access
  return func


def aggregate_mean(scores: np.ndarray):
  """Computes mean of sample mean scores per task.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
  Returns:
    Mean of sample means.
  """
  mean_task_scores = np.mean(scores, axis=0, keepdims=False)
  return np.mean(mean_task_scores, axis=0)


def aggregate_median(scores: np.ndarray):
  """Computes median of sample mean scores per task.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
  Returns:
    Median of sample means.
  """
  mean_task_scores = np.mean(scores, axis=0, keepdims=False)
  return np.median(mean_task_scores, axis=0)


def aggregate_optimality_gap(scores: np.ndarray, gamma=1):
  """Computes optimality gap across all runs and tasks.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
    gamma: Threshold for optimality gap. All scores above `gamma` are clipped
     to `gamma`.

  Returns:
    Optimality gap at threshold `gamma`.
  """
  return gamma - np.mean(np.minimum(scores, gamma))


def aggregate_iqm(scores: np.ndarray):
  """Computes the interquartile mean across runs and tasks.

  Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
      represent the score on run `n` of task `m`.
  Returns:
    IQM (25% trimmed mean) of scores.
  """
  return scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=None)


@batched
def aggregate_mean_batched(scores_RNM: np.ndarray) -> np.ndarray:
  """Batched `aggregate_mean`.

  Args:
    scores_RNM: Array of shape (`reps`, `num_runs`, `num_tasks`, ...).

  Returns:
    Array of shape (`reps`, ...), one mean of per-task sample means per rep.
    Trailing axes (e.g. training steps) are kept, matching the scalar version,
    which reduces only over runs and then over tasks.
  """
  scores_RNM = np.asarray(scores_RNM)
  mean_task_scores_RM = np.mean(scores_RNM, axis=1)
  return np.mean(mean_task_scores_RM, axis=1)


@batched
def aggregate_median_batched(scores_RNM: np.ndarray) -> np.ndarray:
  """Batched `aggregate_median`.

  Args:
    scores_RNM: Array of shape (`reps`, `num_runs`, `num_tasks`, ...).

  Returns:
    Array of shape (`reps`, ...), the median of the per-task sample means per
    rep. Trailing axes (e.g. training steps) are kept, matching the scalar
    version, which reduces only over runs and then over tasks.
  """
  scores_RNM = np.asarray(scores_RNM)
  mean_task_scores_RM = np.mean(scores_RNM, axis=1)
  return np.median(mean_task_scores_RM, axis=1)


@batched
def aggregate_optimality_gap_batched(scores_RNM: np.ndarray, gamma: float = 1) -> np.ndarray:
  """Batched `aggregate_optimality_gap`.

  Args:
    scores_RNM: Array of shape (`reps`, `num_runs`, `num_tasks`, ...).
    gamma: Threshold for optimality gap; scores above `gamma` are clipped.

  Returns:
    Array of shape (`reps`,).
  """
  scores_RNM = np.asarray(scores_RNM)
  clipped_RK = np.minimum(scores_RNM, gamma).reshape(scores_RNM.shape[0], -1)
  return gamma - np.mean(clipped_RK, axis=1)


@batched
def aggregate_iqm_batched(scores_RNM: np.ndarray) -> np.ndarray:
  """Batched `aggregate_iqm` (25% trimmed mean per replication).

  `scipy.stats.trim_mean` cuts `int(proportiontocut * n)` elements from each
  end of the sorted sample, i.e. it floors rather than rounds. The same floor
  is reproduced here so odd sample sizes agree bit-for-bit with the scalar
  version.

  Args:
    scores_RNM: Array of shape (`reps`, `num_runs`, `num_tasks`, ...).

  Returns:
    Array of shape (`reps`,).
  """
  scores_RNM = np.asarray(scores_RNM)
  flat_RK = scores_RNM.reshape(scores_RNM.shape[0], -1)
  num_elements = flat_RK.shape[1]
  lower_cut = int(0.25 * num_elements)
  upper_cut = num_elements - lower_cut
  sorted_RK = np.sort(flat_RK, axis=1)
  return np.mean(sorted_RK[:, lower_cut:upper_cut], axis=1)


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray):
  """Overall Probability of improvement of algorithm `X` over `Y`.

  For each task the Mann-Whitney U statistic with midrank tie handling is
  U_m = #{x > y} + 0.5 * #{x == y} over all run pairs, so the per-task
  probability is U_m / (num_runs_x * num_runs_y). Computing it directly from
  the pairwise comparison avoids a Python loop with a `scipy.stats.mannwhitneyu`
  call per task, which dominated the bootstrap cost.

  Memory note: the pairwise comparison materialises two (`num_runs_x`,
  `num_runs_y`, `num_tasks`) boolean arrays, which is negligible for the run
  counts this library targets (tens of runs).

  Args:
    scores_x: A matrix of size (`num_runs_x` x `num_tasks`) where scores_x[n][m]
      represent the score on run `n` of task `m` for algorithm `X`.
    scores_y: A matrix of size (`num_runs_y` x `num_tasks`) where scores_x[n][m]
      represent the score on run `n` of task `m` for algorithm `Y`.
  Returns:
      P(X_m > Y_m) averaged across tasks.
  """
  scores_x = np.asarray(scores_x)
  scores_y = np.asarray(scores_y)
  num_runs_x, num_runs_y = scores_x.shape[0], scores_y.shape[0]
  x_NxOneM = scores_x[:, None, :]
  y_OneNyM = scores_y[None, :, :]
  greater_M = np.sum(x_NxOneM > y_OneNyM, axis=(0, 1))
  ties_M = np.sum(x_NxOneM == y_OneNyM, axis=(0, 1))
  probabilities_M = (greater_M + 0.5 * ties_M) / (num_runs_x * num_runs_y)
  # Identical columns are a special case in the original implementation (older
  # SciPy raised on all-identical samples). The formula above already yields
  # 0.5 there by symmetry, but the branch is kept explicit so the behaviour is
  # documented rather than incidental.
  if num_runs_x == num_runs_y:
    for task in range(scores_x.shape[1]):
      if np.array_equal(scores_x[:, task], scores_y[:, task]):
        probabilities_M[task] = 0.5
  return np.mean(probabilities_M)


@batched
def probability_of_improvement_batched(scores_x_RNxM: np.ndarray, scores_y_RNyM: np.ndarray) -> np.ndarray:
  """Batched `probability_of_improvement`.

  Args:
    scores_x_RNxM: Array of shape (`reps`, `num_runs_x`, `num_tasks`).
    scores_y_RNyM: Array of shape (`reps`, `num_runs_y`, `num_tasks`).

  Returns:
    Array of shape (`reps`,), P(X_m > Y_m) averaged across tasks per rep.
  """
  scores_x_RNxM = np.asarray(scores_x_RNxM)
  scores_y_RNyM = np.asarray(scores_y_RNyM)
  num_runs_x, num_runs_y = scores_x_RNxM.shape[1], scores_y_RNyM.shape[1]
  # (R, Nx, 1, M) vs (R, 1, Ny, M) -> (R, Nx, Ny, M) booleans.
  x_RNxOneM = scores_x_RNxM[:, :, None, :]
  y_ROneNyM = scores_y_RNyM[:, None, :, :]
  greater_RM = np.sum(x_RNxOneM > y_ROneNyM, axis=(1, 2))
  ties_RM = np.sum(x_RNxOneM == y_ROneNyM, axis=(1, 2))
  probabilities_RM = (greater_RM + 0.5 * ties_RM) / (num_runs_x * num_runs_y)
  return np.mean(probabilities_RM, axis=1)
