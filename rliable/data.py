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
"""Loading score matrices from long-format (tidy) DataFrames.

Most experiment trackers export one row per (algorithm, run, task[, step]),
while rliable expects dense arrays of shape (`num_runs`, `num_tasks`) or
(`num_runs`, `num_tasks`, `num_steps`). Reshaping that by hand is where the
alignment bugs live: a silently dropped run, tasks in a different order for two
algorithms, or a duplicated row that pandas quietly keeps. These loaders do the
pivot once and refuse to return an array that has holes or ambiguity in it.

pandas is imported inside the functions on purpose: the rest of the package
depends only on numpy and scipy, and users who never touch a DataFrame should
not need pandas installed.

Dimension key:
  N: runs   M: tasks   S: steps
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


def _check_columns_exist(df, columns: dict[str, str]) -> None:
  """Raises KeyError naming every requested column that the frame lacks."""
  missing = {name: col for name, col in columns.items() if col not in df.columns}
  if missing:
    described = ", ".join(f"{name}={col!r}" for name, col in missing.items())
    raise KeyError(f"Column(s) not found in DataFrame: {described}. Available columns: {list(df.columns)}")


def _resolve_tasks(present_tasks: list[Any], task_order: Sequence | None) -> list[Any]:
  """Returns the task ordering to use, validating a user-supplied one.

  `task_order` must be a permutation of the tasks present: a subset would drop
  data silently, a superset would produce all-missing columns.
  """
  if task_order is None:
    return list(present_tasks)
  task_order = list(task_order)
  if sorted(map(str, task_order)) != sorted(map(str, present_tasks)) or len(task_order) != len(present_tasks):
    extra = [t for t in task_order if t not in present_tasks]
    absent = [t for t in present_tasks if t not in task_order]
    raise ValueError(
      f"`task_order` must contain exactly the tasks present in the data. "
      f"Unknown tasks in `task_order`: {extra}; tasks missing from `task_order`: {absent}."
    )
  return task_order


def load_scores_from_dataframe(
  df,
  run_col: str,
  task_col: str,
  score_col: str,
  step_col: str | None = None,
  task_order: Sequence | None = None,
) -> tuple[np.ndarray, list]:
  """Converts a long-format DataFrame into a dense score array.

  Args:
    df: Long-format DataFrame with one row per (run, task) or
      (run, task, step) combination.
    run_col: Name of the column identifying the run (seed).
    task_col: Name of the column identifying the task (environment/game).
    score_col: Name of the column holding the score.
    step_col: Optional name of the column holding the training step. When given,
      the returned array gains a trailing step axis.
    task_order: Optional explicit task ordering. Must contain exactly the tasks
      present in `df`. Defaults to sorted unique tasks.

  Returns:
    Tuple of (scores, tasks) where `scores` has shape
    (`num_runs`, `num_tasks`) or (`num_runs`, `num_tasks`, `num_steps`) and
    dtype float64, and `tasks` lists the tasks in the order used along axis 1.

  Raises:
    KeyError: If any of the requested columns is absent from `df`.
    ValueError: If the frame contains duplicate index combinations, or if any
      (run, task[, step]) combination has no score.
  """
  import pandas as pd  # Local import: keeps the core package numpy/scipy only.

  columns = {"run_col": run_col, "task_col": task_col, "score_col": score_col}
  if step_col is not None:
    columns["step_col"] = step_col
  _check_columns_exist(df, columns)

  index_cols = [run_col, task_col] + ([step_col] if step_col is not None else [])
  data = df[index_cols + [score_col]]

  duplicates = data.duplicated(subset=index_cols, keep=False)
  if bool(duplicates.any()):
    duplicated_rows = data.loc[duplicates, index_cols].drop_duplicates()
    listed = [tuple(row) for row in duplicated_rows.to_numpy()[:10]]
    raise ValueError(
      f"Found {len(duplicated_rows)} duplicated {tuple(index_cols)} combination(s); "
      f"each combination must appear exactly once. First up to 10: {listed}"
    )

  runs = sorted(data[run_col].unique())
  tasks = _resolve_tasks(sorted(data[task_col].unique()), task_order)
  levels = [runs, tasks]
  if step_col is not None:
    levels.append(sorted(data[step_col].unique()))

  full_index = pd.MultiIndex.from_product(levels, names=index_cols)
  # Reindexing onto the full product turns every absent combination into NaN.
  # A row that was present but carried NaN is just as unusable downstream, so
  # both cases are reported the same way rather than silently propagating NaN.
  observed = data.set_index(index_cols)[score_col]
  reindexed = observed.reindex(full_index)
  holes = reindexed.isna()
  if bool(holes.any()):
    missing_index = holes[holes].index
    listed = [tuple(combo) for combo in list(missing_index)[:10]]
    per_task = missing_index.get_level_values(task_col).value_counts().to_dict()
    raise ValueError(
      f"{len(missing_index)} (run, task{', step' if step_col else ''}) combination(s) have no score. "
      f"First up to 10: {listed}. Missing per task: {per_task}"
    )

  shape = (len(runs), len(tasks)) + ((len(levels[2]),) if step_col is not None else ())
  scores = np.asarray(reindexed.to_numpy(), dtype=np.float64).reshape(shape)
  return scores, list(tasks)


def load_score_dict_from_dataframe(
  df,
  algorithm_col: str,
  run_col: str,
  task_col: str,
  score_col: str,
  step_col: str | None = None,
  task_order: Sequence | None = None,
) -> tuple[dict[str, np.ndarray], list]:
  """Converts a long-format DataFrame into the score dict rliable expects.

  Every algorithm is pivoted with the same task ordering, so the returned
  arrays are aligned along axis 1 and can be compared task by task.

  Args:
    df: Long-format DataFrame with an algorithm column.
    algorithm_col: Name of the column identifying the algorithm.
    run_col: Name of the column identifying the run (seed).
    task_col: Name of the column identifying the task.
    score_col: Name of the column holding the score.
    step_col: Optional name of the column holding the training step.
    task_order: Optional explicit task ordering, applied to every algorithm.

  Returns:
    Tuple of (score_dict, tasks) mapping algorithm name to its score array,
    plus the shared task list.

  Raises:
    KeyError: If any of the requested columns is absent from `df`.
    ValueError: If any algorithm's data is incomplete, duplicated, or covers a
      different set of tasks than the first algorithm.
  """
  _check_columns_exist(df, {"algorithm_col": algorithm_col})

  score_dict: dict[str, np.ndarray] = {}
  shared_tasks: list | None = task_order if task_order is None else list(task_order)
  for algorithm in sorted(df[algorithm_col].unique()):
    group = df[df[algorithm_col] == algorithm]
    try:
      scores, tasks = load_scores_from_dataframe(
        group, run_col=run_col, task_col=task_col, score_col=score_col, step_col=step_col, task_order=shared_tasks
      )
    except ValueError as error:
      raise ValueError(f"Algorithm {algorithm!r}: {error}") from error
    # The first algorithm fixes the ordering that the rest must reproduce.
    shared_tasks = tasks
    score_dict[algorithm] = scores
  if shared_tasks is None:
    raise ValueError(f"No rows found: column {algorithm_col!r} has no values.")
  return score_dict, list(shared_tasks)
