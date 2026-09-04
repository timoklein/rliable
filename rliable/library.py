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
"""Main library functions for interval estimates and performance profiles.

This module implements the stratified bootstrap of Agarwal et al. (2021) in pure
numpy/scipy. Earlier versions delegated the resampling machinery to the `arch`
package; `arch` pulled in statsmodels plus a compiled build and broke on modern
pandas, while rliable only used two of its classes and `conf_int`. The
resampling semantics and the confidence-interval formulas here are kept
identical to what `arch` computed, so results are statistically equivalent.

Dimension key:
  R: bootstrap reps    N: runs        M: tasks
  T: profile thresholds (`tau_list`)  K: statistics returned by `func`
"""

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

Float = float | np.floating
RandomStateLike = int | np.integer | np.random.Generator | np.random.RandomState | None

# Roughly how many array elements a single gathered bootstrap chunk may hold.
# 8e6 float64 elements is ~64 MB, which keeps the vectorized path fast without
# risking memory pressure for large score matrices or long `tau_list`s.
_CHUNK_ELEMENT_BUDGET = 8_000_000

_METHOD_ALIASES = {
  "debiased": "bc",
  "bias-corrected": "bc",
  "bias_corrected": "bc",
}
_VALID_METHODS = ("basic", "percentile", "bc", "bca")


####################### Random state helpers #######################
def _as_generator(random_state: RandomStateLike) -> np.random.Generator | np.random.RandomState:
  """Normalizes `random_state` into a numpy random generator.

  Accepting both the modern `Generator` and the legacy `RandomState` keeps
  existing user code working (`arch` only accepted `RandomState`) while making
  `None` and integer seeds behave the way callers expect.

  Args:
    random_state: `None` for a fresh, unseeded generator, an integer seed, or an
      already constructed `np.random.Generator` / `np.random.RandomState`.

  Returns:
    A generator that `_integers` can draw from.

  Raises:
    TypeError: If `random_state` is of any other type.
  """
  if random_state is None:
    return np.random.default_rng()
  if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
    return random_state
  if isinstance(random_state, (int, np.integer)) and not isinstance(random_state, bool):
    return np.random.default_rng(int(random_state))
  raise TypeError(
    f"`random_state` must be None, an int, a numpy Generator or a numpy RandomState, got {type(random_state).__name__}."
  )


def _integers(rng: np.random.Generator | np.random.RandomState, high: int, size: tuple[int, ...]) -> np.ndarray:
  """Draws integers in `[0, high)` from either generator flavour."""
  if isinstance(rng, np.random.Generator):
    return rng.integers(0, high, size=size)
  return rng.randint(0, high, size=size)


def _is_batched(func: Callable[..., np.ndarray], batched: bool | None) -> bool:
  """Resolves whether `func` consumes a whole `(R, ...)` batch at once."""
  if batched is not None:
    return bool(batched)
  return bool(getattr(func, "_rliable_batched", False))


def _default_chunk_size(reps: int, elements_per_rep: int) -> int:
  """Number of reps to gather at once so a chunk stays inside the budget."""
  elements_per_rep = max(int(elements_per_rep), 1)
  return max(1, min(int(reps), _CHUNK_ELEMENT_BUDGET // elements_per_rep))


####################### Confidence intervals #######################
def _percentile_per_column(results: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
  """Per-column percentile with a per-column probability level.

  `np.percentile` applies the same probability to every column, but the `bc` and
  `bca` methods correct the probability separately for each statistic, so the
  columns have to be handled one at a time.

  Args:
    results: `(R, K)` array of bootstrap statistics.
    quantiles: `(K,)` array of probabilities in [0, 1].

  Returns:
    A `(K,)` array of percentiles, linearly interpolated as `arch` did.
  """
  return np.array([np.percentile(results[:, k], 100.0 * quantiles[k], method="linear") for k in range(results.shape[1])])


def _confidence_interval(
  base: np.ndarray,
  results: np.ndarray,
  method: str,
  size: Float,
  jackknife: Callable[[], np.ndarray] | None = None,
) -> np.ndarray:
  """Computes bootstrap confidence intervals from the replicate statistics.

  The formulas mirror `arch.bootstrap.IIDBootstrap.conf_int`: linear-interpolation
  percentiles, a strict `<` comparison in the bias term, and the standard BCa
  acceleration from a leave-one-out jackknife.

  Degenerate inputs (constant scores, a statistic that never varies) used to
  make `arch` raise or return NaN. Because rliable is routinely handed such data
  (e.g. a task where every run scores 0), the degenerate branches here clip to
  the nearest representable value and emit a `RuntimeWarning` naming the
  affected statistics instead of failing.

  Args:
    base: `(K,)` point estimates computed on the original data.
    results: `(R, K)` bootstrap replicates of the statistic.
    method: One of `basic`, `percentile`, `bc` (aliases `debiased`,
      `bias-corrected`) or `bca`.
    size: Coverage of the interval, in (0, 1).
    jackknife: Callable returning the `(N, K)` leave-one-run-out statistics.
      Only used, and only required, for `bca`.

  Returns:
    A `(2, K)` array with the lower bounds in row 0 and the upper bounds in
    row 1.

  Raises:
    ValueError: If `size` is outside (0, 1), if `method` is unknown, or if
      `bca` is requested without a jackknife.
  """
  if not 0.0 < float(size) < 1.0:
    raise ValueError(f"`size` must lie strictly between 0 and 1, got {size}.")
  method = _METHOD_ALIASES.get(str(method).lower(), str(method).lower())
  if method not in _VALID_METHODS:
    raise ValueError(f"Unknown method {method!r}; expected one of {_VALID_METHODS}.")

  base = np.asarray(base, dtype=float).reshape(-1)
  results = np.asarray(results, dtype=float)
  reps, num_stats = results.shape
  alpha = (1.0 - float(size)) / 2.0

  if method in ("percentile", "basic"):
    lower, upper = np.percentile(results, [100.0 * alpha, 100.0 * (1.0 - alpha)], axis=0, method="linear")
    if method == "basic":
      # Reflect the bootstrap distribution around the point estimate.
      lower, upper = 2.0 * base - upper, 2.0 * base - lower
    return np.vstack((lower, upper))

  # `bc` and `bca` both start from the bias-correction term z0.
  z_lo, z_hi = norm.ppf(alpha), norm.ppf(1.0 - alpha)
  prob = np.mean(results < base, axis=0)
  lo_clip, hi_clip = 1.0 / (2.0 * reps), 1.0 - 1.0 / (2.0 * reps)
  degenerate = (prob <= 0.0) | (prob >= 1.0)
  if np.any(degenerate):
    warnings.warn(
      "Bootstrap replicates never straddle the point estimate for statistic(s) "
      f"{np.flatnonzero(degenerate).tolist()}; the bias correction was clipped to "
      f"[{lo_clip:g}, {hi_clip:g}]. This usually means the scores are constant.",
      RuntimeWarning,
      stacklevel=2,
    )
  z0 = norm.ppf(np.clip(prob, lo_clip, hi_clip))

  accel = np.zeros(num_stats)
  if method == "bca":
    if jackknife is None:
      raise ValueError("The `bca` method requires a jackknife callable.")
    jk = np.asarray(jackknife(), dtype=float)
    deviations = jk.mean(axis=0) - jk  # (N, K)
    numerator = np.sum(deviations**3, axis=0)
    denominator = 6.0 * np.sum(deviations**2, axis=0) ** 1.5
    zero_denominator = denominator <= 0.0
    if np.any(zero_denominator):
      warnings.warn(
        "The jackknife statistic is constant for statistic(s) "
        f"{np.flatnonzero(zero_denominator).tolist()}; the BCa acceleration was set "
        "to 0, which reduces those columns to the bias-corrected interval.",
        RuntimeWarning,
        stacklevel=2,
      )
    accel = np.where(zero_denominator, 0.0, numerator / np.where(zero_denominator, 1.0, denominator))

  quantiles = []
  for z in (z_lo, z_hi):
    shift = z0 + z
    denominator = 1.0 - accel * shift
    invalid = denominator <= 0.0
    if np.any(invalid):
      warnings.warn(
        "The BCa acceleration flips the quantile mapping for statistic(s) "
        f"{np.flatnonzero(invalid).tolist()}; those columns fall back to the "
        "bias-corrected quantile and are clipped to [0, 1].",
        RuntimeWarning,
        stacklevel=2,
      )
    safe_denominator = np.where(invalid, 1.0, denominator)
    quantiles.append(np.clip(norm.cdf(z0 + shift / safe_denominator), 0.0, 1.0))

  lower = _percentile_per_column(results, quantiles[0])
  upper = _percentile_per_column(results, quantiles[1])
  return np.vstack((lower, upper))


####################### Stratified Bootstrap #######################
class _BaseStratifiedBootstrap:
  """Shared chunked-evaluation machinery for the two stratified bootstraps."""

  _name = "Base Stratified Bootstrap"

  def __init__(self, *args: np.ndarray, random_state: RandomStateLike = None) -> None:
    if not args:
      raise ValueError("At least one positional score array is required.")
    self.pos_data = tuple(np.asarray(arg) for arg in args)
    for arg in self.pos_data:
      if arg.ndim < 2:
        raise ValueError(
          f"Score arrays must have shape (num_runs, num_tasks, ...); got an array with {arg.ndim} dimension(s)."
        )
    self.data = (self.pos_data, {})
    self._generator = _as_generator(random_state)
    self.index: tuple[np.ndarray, ...] | list[tuple[np.ndarray, ...]] | None = None

  # --- to be provided by the subclasses -------------------------------------
  def sample_indices(self, reps: int = 1):
    raise NotImplementedError

  def _gather(self, indices) -> tuple[np.ndarray, ...]:
    raise NotImplementedError

  def _rep_index(self, indices, rep: int):
    raise NotImplementedError

  def _jackknife_args(self) -> list[tuple[np.ndarray, ...]]:
    raise NotImplementedError

  @property
  def _elements_per_rep(self) -> int:
    return int(sum(arg.size for arg in self.pos_data))

  # --- public API -----------------------------------------------------------
  def resample(self, reps: int = 1) -> tuple[np.ndarray, ...]:
    """Draws `reps` bootstrap resamples of every positional array.

    Args:
      reps: Number of bootstrap replications to draw at once.

    Returns:
      One `(R, N, M, ...)` array per positional argument.
    """
    indices = self.sample_indices(reps)
    self.index = indices
    return self._gather(indices)

  def bootstrap(self, reps: int):
    """Yields one bootstrap resample at a time, `arch`-style.

    Each iteration sets `self.index` to the index tuple of that replication, so
    `array[bs.index]` reproduces the yielded data.

    Args:
      reps: Number of bootstrap replications to yield.

    Yields:
      `(pos_data, kw_data)` where `pos_data` is a tuple with one resampled array
      per positional argument and `kw_data` is always an empty dict (keyword
      data arrays were dropped along with the `arch` dependency).
    """
    for _ in range(reps):
      indices = self.sample_indices(1)
      rep_index = self._rep_index(indices, 0)
      self.index = rep_index
      yield tuple(arg[idx] for arg, idx in zip(self.pos_data, self._as_per_arg(rep_index))), {}

  def _as_per_arg(self, rep_index):
    """Maps one replication's index onto the positional arguments."""
    return [rep_index] * len(self.pos_data)

  def conf_int(
    self,
    func: Callable[..., np.ndarray],
    reps: int = 1000,
    method: str = "basic",
    size: Float = 0.95,
    batched: bool | None = None,
    chunk_size: int | None = None,
  ) -> np.ndarray:
    """Computes bootstrap confidence intervals for `func`.

    Args:
      func: Statistic to bootstrap. Called as `func(*resampled_args)`. If it is
        marked batched (attribute `_rliable_batched`, or `batched=True`) it is
        instead called once per chunk with `(R, N, M, ...)` arrays and must
        return something reshapable to `(R, K)`.
      reps: Number of bootstrap replications.
      method: One of `basic`, `percentile`, `bc` or `bca`.
      size: Coverage of the confidence interval.
      batched: Overrides the `_rliable_batched` marker on `func`.
      chunk_size: Number of replications gathered at once. Defaults to a size
        that keeps a chunk near `_CHUNK_ELEMENT_BUDGET` elements.

    Returns:
      A `(2, K)` array of lower and upper bounds.
    """
    is_batched = _is_batched(func, batched)
    base, results = self._bootstrap_statistics(func, reps, is_batched, chunk_size)
    return _confidence_interval(base, results, method, size, jackknife=lambda: self._jackknife(func, is_batched))

  # --- internals ------------------------------------------------------------
  def _base_statistic(self, func: Callable[..., np.ndarray], is_batched: bool) -> np.ndarray:
    """Applies `func` to the original data, flattened to `(K,)`."""
    if is_batched:
      value = np.asarray(func(*(arg[None, ...] for arg in self.pos_data)))
      return value.reshape(-1)
    return np.asarray(func(*self.pos_data)).reshape(-1)

  def _bootstrap_statistics(
    self,
    func: Callable[..., np.ndarray],
    reps: int,
    is_batched: bool,
    chunk_size: int | None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates `func` on `reps` resamples, gathering one chunk at a time.

    Gathering the full `(R, N, M, ...)` resample at once would be the fastest
    option but needs R times the memory of the score matrix, which is
    prohibitive at the default `reps=50000`. Chunking keeps the vectorized
    gather while bounding peak memory.

    Args:
      func: The statistic; see `conf_int`.
      reps: Number of bootstrap replications.
      is_batched: Whether `func` consumes a whole chunk at once.
      chunk_size: Replications per chunk, or None for the default budget.

    Returns:
      `base` of shape `(K,)` and `results` of shape `(R, K)`.
    """
    base = self._base_statistic(func, is_batched)
    num_stats = 1 if base.ndim == 0 else base.size
    base = base.reshape(num_stats)
    if chunk_size is None:
      chunk_size = _default_chunk_size(reps, self._elements_per_rep)
    chunk_size = max(1, int(chunk_size))

    results = np.empty((reps, num_stats), dtype=float)
    done = 0
    while done < reps:
      chunk = min(chunk_size, reps - done)
      indices = self.sample_indices(chunk)
      batch = self._gather(indices)
      if is_batched:
        results[done : done + chunk] = np.asarray(func(*batch)).reshape(chunk, -1)
      else:
        for rep in range(chunk):
          # `arr[rep]` is a view, so no extra copy per replication.
          results[done + rep] = np.asarray(func(*(arr[rep] for arr in batch))).reshape(-1)
      done += chunk
    return base, results

  def _jackknife(self, func: Callable[..., np.ndarray], is_batched: bool) -> np.ndarray:
    """Leave-one-run-out jackknife statistics, shape `(N, K)`.

    The bootstrap resamples runs, so the BCa acceleration is estimated by
    dropping one run at a time (never a task or a training step).
    """
    values = []
    for args in self._jackknife_args():
      if is_batched:
        value = np.asarray(func(*(arg[None, ...] for arg in args))).reshape(-1)
      else:
        value = np.asarray(func(*args)).reshape(-1)
      values.append(value)
    return np.array(values, dtype=float)


class StratifiedBootstrap(_BaseStratifiedBootstrap):
  """Bootstrap using stratified resampling over runs (and optionally tasks).

  Every `(task, step, ...)` cell draws its own run index, so the resample keeps
  each column inside its own task -- that is the stratification. With
  `task_bootstrap=True` the task axis is resampled as well, with a single task
  permutation shared by all runs and steps within a replication, which captures
  the sensitivity of the aggregate to the particular task suite.

  Examples:
    >>> import numpy as np
    >>> from rliable.library import StratifiedBootstrap
    >>> x = np.random.default_rng(0).standard_normal((5, 50))
    >>> bs = StratifiedBootstrap(x, random_state=0)
    >>> for data, _ in bs.bootstrap(100):
    ...   bs_x = data[0]
    >>> ci = bs.conf_int(np.mean, method='percentile', reps=50000)  # 95% CI

  Attributes:
    pos_data: Tuple of the positional score arrays, in the order entered.
    data: Two-element tuple `(pos_data, {})`, kept for backwards compatibility.
    index: Index tuple of the most recent draw.
  """

  _name = "Stratified Bootstrap"

  def __init__(
    self,
    *args: np.ndarray,
    random_state: RandomStateLike = None,
    task_bootstrap: bool = False,
  ) -> None:
    """Initializes StratifiedBootstrap.

    Args:
      *args: Score arrays of shape `(num_runs, num_tasks, ...)`. All arrays must
        agree on the leading dimensions of the first one; trailing dimensions
        may differ.
      random_state: `None`, an int seed, a `np.random.Generator` or a
        `np.random.RandomState`. Ensures reproducibility when set.
      task_bootstrap: Whether to bootstrap (a) over runs only or (b) over both
        runs and tasks. Defaults to False, i.e. (a). (a) captures the
        statistical uncertainty in the aggregate performance if the experiment
        were repeated with a different set of runs on the same tasks. (b) also
        captures the sensitivity to the particular task suite.
    """
    super().__init__(*args, random_state=random_state)
    self._args_shape = self.pos_data[0].shape
    for arg in self.pos_data[1:]:
      if arg.shape[: len(self._args_shape)] != self._args_shape:
        raise ValueError(f"All positional arrays must share the leading shape {self._args_shape}; got {arg.shape}.")
    # `_num_items` is the number of runs, mirroring the old `arch` attribute.
    self._num_items = self._args_shape[0]
    self._num_tasks = self._args_shape[1]
    self._task_bootstrap = task_bootstrap
    self._parameters = [self._num_tasks, task_bootstrap]
    self._strata_indices = self._get_strata_indices()

  def _get_strata_indices(self) -> list[np.ndarray]:
    """Static per-axis indices that pin every cell to its own task/step.

    Returns:
      A list of arrays of shape `1 x M x 1 x ..`, `1 x 1 x L x ..` and so on for
      an `args_shape` of `N x M x L x ..`. They broadcast against the drawn run
      indices, which is what keeps column `m` of a resample drawn from column
      `m` of the source data.
    """
    ogrid_indices = tuple(slice(x) for x in (0, *self._args_shape[1:]))
    strata_indices = np.ogrid[ogrid_indices]
    return strata_indices[1:]

  def sample_indices(self, reps: int = 1) -> tuple[np.ndarray, ...]:
    """Draws the index tuple for `reps` bootstrap replications.

    Args:
      reps: Number of replications to draw.

    Returns:
      A tuple that indexes a source array into shape `(R, N, M, ...)`: the run
      indices of shape `(R, N, M, ...)` first, then either the static strata
      indices or, with `task_bootstrap=True`, a drawn task index of shape
      `(R, 1, M, 1, ...)` in place of the task axis.
    """
    run_indices = _integers(self._generator, self._num_items, (reps, *self._args_shape))
    if self._task_bootstrap:
      task_shape = (reps, *self._strata_indices[0].shape)
      task_indices = _integers(self._generator, self._num_tasks, task_shape)
      return (run_indices, task_indices, *self._strata_indices[1:])
    return (run_indices, *self._strata_indices)

  @property
  def _num_drawn(self) -> int:
    """How many leading entries of the index tuple carry the reps axis."""
    return 2 if self._task_bootstrap else 1

  def _rep_index(self, indices: tuple[np.ndarray, ...], rep: int) -> tuple[np.ndarray, ...]:
    drawn = tuple(idx[rep] for idx in indices[: self._num_drawn])
    return (*drawn, *indices[self._num_drawn :])

  def _gather(self, indices: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(arg[indices] for arg in self.pos_data)

  def _jackknife_args(self) -> list[tuple[np.ndarray, ...]]:
    return [tuple(np.delete(arg, run, axis=0) for arg in self.pos_data) for run in range(self._num_items)]


class StratifiedIndependentBootstrap(_BaseStratifiedBootstrap):
  """Stratified bootstrap where each input array is resampled independently.

  Useful for metrics taking several score arrays with possibly different numbers
  of runs, such as the probability of improvement. `task_bootstrap` is not
  supported here.

  Attributes:
    pos_data: Tuple of the positional score arrays, in the order entered.
    data: Two-element tuple `(pos_data, {})`, kept for backwards compatibility.
    index: List with one index tuple per positional argument, from the most
      recent draw.
  """

  _name = "Stratified Independent Bootstrap"

  def __init__(self, *args: np.ndarray, random_state: RandomStateLike = None) -> None:
    """Initializes StratifiedIndependentBootstrap.

    Args:
      *args: Score arrays of shape `(num_runs, num_tasks, ...)`. The number of
        runs may differ between arrays; the number of tasks may not.
      random_state: `None`, an int seed, a `np.random.Generator` or a
        `np.random.RandomState`.
    """
    super().__init__(*args, random_state=random_state)
    self._args_shapes = [arg.shape for arg in self.pos_data]
    self._num_arg_items = [shape[0] for shape in self._args_shapes]
    self._args_shape = self._args_shapes[0]
    self._num_items = self._num_arg_items[0]
    self._num_tasks = self._args_shape[1]
    self._task_bootstrap = False
    self._args_strata_indices = [self._get_strata_indices(shape) for shape in self._args_shapes]
    self._strata_indices = self._args_strata_indices[0]

  def _get_strata_indices(self, array_shape: tuple[int, ...]) -> list[np.ndarray]:
    """Static per-axis indices for one array; see `StratifiedBootstrap`."""
    ogrid_indices = tuple(slice(x) for x in (0, *array_shape[1:]))
    strata_indices = np.ogrid[ogrid_indices]
    return strata_indices[1:]

  def sample_indices(self, reps: int = 1) -> list[tuple[np.ndarray, ...]]:
    """Draws one independent index tuple per positional argument.

    Args:
      reps: Number of replications to draw.

    Returns:
      A list with, for each argument, a tuple indexing it into `(R, N, M, ...)`.
    """
    indices = []
    for num_runs, shape, strata in zip(self._num_arg_items, self._args_shapes, self._args_strata_indices):
      run_indices = _integers(self._generator, num_runs, (reps, *shape))
      indices.append((run_indices, *strata))
    return indices

  def _rep_index(self, indices, rep: int) -> list[tuple[np.ndarray, ...]]:
    return [(idx[0][rep], *idx[1:]) for idx in indices]

  def _as_per_arg(self, rep_index):
    return rep_index

  def _gather(self, indices) -> tuple[np.ndarray, ...]:
    return tuple(arg[idx] for arg, idx in zip(self.pos_data, indices))

  def _jackknife_args(self) -> list[tuple[np.ndarray, ...]]:
    """Leave-one-run-out over all arrays at once.

    This only makes sense when every array has the same number of runs; `arch`
    raised in the unequal case and so do we, rather than silently pairing up
    runs that are not comparable.

    Raises:
      ValueError: If the arrays have different numbers of runs.
    """
    if len(set(self._num_arg_items)) > 1:
      raise ValueError(
        "The jackknife (and therefore the `bca` method) requires every score array to "
        f"have the same number of runs; got {self._num_arg_items}."
      )
    return [tuple(np.delete(arg, run, axis=0) for arg in self.pos_data) for run in range(self._num_items)]


####################### Interval Estimates #######################
def get_interval_estimates(
  score_dict: Mapping[str, np.ndarray] | Mapping[str, Sequence[np.ndarray]],
  func: Callable[..., np.ndarray],
  method: str = "percentile",
  task_bootstrap: bool = False,
  reps: int = 50000,
  confidence_interval_size: Float = 0.95,
  random_state: RandomStateLike = None,
  batched: bool | None = None,
  chunk_size: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
  """Computes interval estimates via stratified bootstrap confidence intervals.

  Args:
    score_dict: A dictionary of scores for each method where scores are arranged
      as a matrix of the shape (`num_runs` x `num_tasks` x ..). For example, the
      scores could be a 2D matrix containing final scores of the algorithm or a
      3D matrix containing evaluation scores at multiple points during training.
      A list or tuple of arrays is routed to `StratifiedIndependentBootstrap`
      and passed to `func` as separate arguments.
    func: Function that computes the aggregate performance, which outputs a 1D
      numpy array. For example, if computing estimates for the interquartile
      mean across all runs, pass `lambda x: np.array([metrics.aggregate_iqm(x)])`.
    method:  One of `basic`, `percentile`, `bc` (identical to `debiased`,
      `bias-corrected`), or `bca`.
    task_bootstrap:  Whether to perform bootstrapping over tasks in addition to
      runs. Defaults to False. See `StratifiedBootstrap` for more details.
    reps: Number of bootstrap replications.
    confidence_interval_size: Coverage of confidence interval. Defaults to 95%.
    random_state: `None`, an int seed, a `np.random.Generator` or a
      `np.random.RandomState`. A single generator is created here and shared
      across keys, so one int seed makes the whole call reproducible while still
      giving each algorithm its own draws.
    batched: Set to True if `func` accepts a whole `(R, N, M, ...)` batch and
      returns `(R, K)`. Defaults to None, i.e. read the `_rliable_batched`
      marker off `func`.
    chunk_size: Replications gathered per chunk; None picks a memory budget.

  Returns:
    point_estimates: A dictionary of point estimates obtained by applying `func`
      on score data corresponding to each key in `score_dict`.
    interval_estimates: Confidence intervals~(CIs) for point estimates. Default
      is to return 95% CIs. Returns a np array of size (2 x ..) where the first
      row contains the lower bounds while the second row contains the upper
      bound of the 95% CIs.

  Notes:
    The bootstraps are:

    * 'basic' - Basic confidence using the estimated parameter and
      difference between the estimated parameter and the bootstrap
      parameters.
    * 'percentile' - Direct use of bootstrap percentiles.
    * 'bc' - Bias corrected using estimate bootstrap bias correction.
    * 'bca' - Bias corrected and accelerated, adding acceleration parameter
      to 'bc' method.
  """
  generator = _as_generator(random_state)
  is_batched = _is_batched(func, batched)
  interval_estimates, point_estimates = {}, {}
  for key, scores in score_dict.items():
    logger.info("Calculating estimates for %s ...", key)
    if isinstance(scores, np.ndarray):
      stratified_bs = StratifiedBootstrap(scores, task_bootstrap=task_bootstrap, random_state=generator)
      args = (scores,)
    else:
      # Pass arrays as separate arguments; `task_bootstrap` is not supported.
      stratified_bs = StratifiedIndependentBootstrap(*scores, random_state=generator)
      args = tuple(scores)
    if is_batched:
      # A batched `func` reads axis 0 as the reps axis, so the point estimate is
      # computed on a batch of one.
      point_estimates[key] = np.asarray(func(*(np.asarray(a)[None, ...] for a in args)))[0]
    else:
      point_estimates[key] = func(*args)
    interval_estimates[key] = stratified_bs.conf_int(
      func, reps=reps, size=confidence_interval_size, method=method, batched=is_batched, chunk_size=chunk_size
    )
  return point_estimates, interval_estimates


####################### Performance Profiles #######################
def run_score_deviation(scores: np.ndarray, tau: Float) -> Float:
  """Evaluates how many `scores` are above `tau` averaged across all runs."""
  return np.mean(scores > tau)


def mean_score_deviation(scores: np.ndarray, tau: Float) -> Float:
  """Evaluates how many average task `scores` are above `tau`."""
  return np.mean(np.mean(scores, axis=0) > tau)


def score_distributions(scores: np.ndarray, tau_list: Sequence[Float] | np.ndarray) -> np.ndarray:
  """Fraction of all run/task scores above each threshold.

  Args:
    scores: Scores of shape `(num_runs, num_tasks, ...)`.
    tau_list: `(T,)` thresholds.

  Returns:
    A `(T,)` array; entry `t` equals `run_score_deviation(scores, tau_list[t])`.
  """
  taus = np.asarray(tau_list, dtype=float).reshape(-1)
  flat_scores = np.asarray(scores).reshape(1, -1)
  return np.mean(flat_scores > taus[:, None], axis=1)


def average_score_distributions(scores: np.ndarray, tau_list: Sequence[Float] | np.ndarray) -> np.ndarray:
  """Fraction of run-averaged task scores above each threshold.

  Args:
    scores: Scores of shape `(num_runs, num_tasks, ...)`.
    tau_list: `(T,)` thresholds.

  Returns:
    A `(T,)` array; entry `t` equals `mean_score_deviation(scores, tau_list[t])`.
  """
  taus = np.asarray(tau_list, dtype=float).reshape(-1)
  mean_scores = np.mean(np.asarray(scores), axis=0).reshape(1, -1)
  return np.mean(mean_scores > taus[:, None], axis=1)


def score_distributions_batched(scores: np.ndarray, tau_list: Sequence[Float] | np.ndarray) -> np.ndarray:
  """Batched `score_distributions` over a whole `(R, N, M, ...)` resample.

  Args:
    scores: Scores of shape `(R, num_runs, num_tasks, ...)`.
    tau_list: `(T,)` thresholds.

  Returns:
    An `(R, T)` array.
  """
  taus = np.asarray(tau_list, dtype=float).reshape(-1)
  scores = np.asarray(scores)
  flat_scores = scores.reshape(scores.shape[0], 1, -1)
  return np.mean(flat_scores > taus[None, :, None], axis=2)


def average_score_distributions_batched(scores: np.ndarray, tau_list: Sequence[Float] | np.ndarray) -> np.ndarray:
  """Batched `average_score_distributions` over a `(R, N, M, ...)` resample.

  Args:
    scores: Scores of shape `(R, num_runs, num_tasks, ...)`.
    tau_list: `(T,)` thresholds.

  Returns:
    An `(R, T)` array.
  """
  taus = np.asarray(tau_list, dtype=float).reshape(-1)
  scores = np.asarray(scores)
  mean_scores = np.mean(scores, axis=1).reshape(scores.shape[0], 1, -1)
  return np.mean(mean_scores > taus[None, :, None], axis=2)


score_distributions_batched._rliable_batched = True
average_score_distributions_batched._rliable_batched = True


def create_performance_profile(
  score_dict: Mapping[str, np.ndarray],
  tau_list: Sequence[Float] | np.ndarray,
  use_score_distribution: bool = True,
  custom_profile_func: Callable[..., np.ndarray] | None = None,
  method: str = "percentile",
  task_bootstrap: bool = False,
  reps: int = 2000,
  confidence_interval_size: Float = 0.95,
  random_state: RandomStateLike = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
  """Function for calculating performance profiles.

  Args:
    score_dict: A dictionary of scores for each method where scores are arranged
      as a matrix of the shape (`num_runs` x `num_tasks` x ..).
    tau_list: List or 1D numpy array of threshold values on which the profile is
      evaluated.
    use_score_distribution: Whether to report score distributions or average
      score distributions. Defaults to score distributions for smaller
      uncertainty in reported results with unbiased profiles.
    custom_profile_func: Custom performance profile function. Can be used to
      compute performance profiles other than score distributions. It is called
      once per replication as `custom_profile_func(scores, tau_list)` unless it
      carries a truthy `_rliable_batched` attribute, in which case it receives a
      whole `(R, N, M, ...)` batch and must return `(R, T)`.
    method: Bootstrap method for `StratifiedBootstrap`, defaults to percentile.
    task_bootstrap:  Whether to perform bootstrapping over tasks in addition to
      runs. Defaults to False. See `StratifiedBootstrap` for more details.
    reps: Number of bootstrap replications.
    confidence_interval_size: Coverage of confidence interval. Defaults to 95%.
    random_state: `None`, an int seed, a `np.random.Generator` or a
      `np.random.RandomState`; shared across keys.

  Returns:
    profiles: A dictionary of performance profiles for each key in `score_dict`.
      Each profile is a 1D np array of same size as `tau_list`.
    profile_cis: The 95% confidence intervals of profiles evaluated at
      all thresholds in `tau_list`.
  """
  taus = np.asarray(tau_list, dtype=float).reshape(-1)
  if custom_profile_func is None:
    # The built-in profiles are cheap to vectorize, so always take the batched
    # path: one comparison per chunk instead of one Python call per replication.
    batched_func = score_distributions_batched if use_score_distribution else average_score_distributions_batched

    def profile_function(scores):
      return batched_func(scores, taus)

    profile_function._rliable_batched = True
  else:

    def profile_function(scores):
      return custom_profile_func(scores, taus)

    profile_function._rliable_batched = bool(getattr(custom_profile_func, "_rliable_batched", False))

  # A profile chunk holds `chunk x T x scores.size` comparisons, so `T` has to
  # enter the memory budget.
  largest = max(np.asarray(scores).size for scores in score_dict.values())
  chunk_size = _default_chunk_size(reps, largest * taus.size)

  profiles, profile_cis = get_interval_estimates(
    score_dict,
    func=profile_function,
    task_bootstrap=task_bootstrap,
    method=method,
    reps=reps,
    confidence_interval_size=confidence_interval_size,
    random_state=random_state,
    chunk_size=chunk_size,
  )
  return profiles, profile_cis
