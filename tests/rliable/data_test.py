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
"""Tests the long-format DataFrame loaders."""

import numpy as np
import pandas as pd
import pytest

from rliable import data

_RUNS = [0, 1, 2]
_TASKS = ["breakout", "asterix", "pong"]
_STEPS = [10, 20]
_ALGORITHMS = ["dqn", "rainbow"]


def _long_frame(scores, runs=_RUNS, tasks=_TASKS, steps=None, algorithm=None):
  """Melts a dense score array back into one row per index combination."""
  rows = []
  for i, run in enumerate(runs):
    for j, task in enumerate(tasks):
      if steps is None:
        rows.append({"run": run, "task": task, "score": scores[i, j]})
      else:
        for k, step in enumerate(steps):
          rows.append({"run": run, "task": task, "step": step, "score": scores[i, j, k]})
  df = pd.DataFrame(rows)
  if algorithm is not None:
    df["algorithm"] = algorithm
  return df


def _dense(shape, seed=0):
  return np.random.default_rng(seed).normal(size=shape)


def test_round_trip_without_steps():
  scores = _dense((len(_RUNS), len(_TASKS)))
  df = _long_frame(scores)
  loaded, tasks = data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score")
  assert tasks == sorted(_TASKS)
  # The default ordering is sorted tasks, so reorder the source to compare.
  order = [_TASKS.index(t) for t in tasks]
  np.testing.assert_allclose(loaded, scores[:, order])
  assert loaded.dtype == np.float64


def test_round_trip_with_steps():
  scores = _dense((len(_RUNS), len(_TASKS), len(_STEPS)))
  df = _long_frame(scores, steps=_STEPS)
  loaded, tasks = data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score", step_col="step")
  assert loaded.shape == (len(_RUNS), len(_TASKS), len(_STEPS))
  order = [_TASKS.index(t) for t in tasks]
  np.testing.assert_allclose(loaded, scores[:, order])


def test_shuffled_rows_round_trip():
  scores = _dense((len(_RUNS), len(_TASKS)))
  df = _long_frame(scores).sample(frac=1.0, random_state=0)
  loaded, tasks = data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score")
  order = [_TASKS.index(t) for t in tasks]
  np.testing.assert_allclose(loaded, scores[:, order])


def test_task_order_respected():
  scores = _dense((len(_RUNS), len(_TASKS)))
  df = _long_frame(scores)
  wanted = ["pong", "breakout", "asterix"]
  loaded, tasks = data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score", task_order=wanted)
  assert tasks == wanted
  order = [_TASKS.index(t) for t in wanted]
  np.testing.assert_allclose(loaded, scores[:, order])


@pytest.mark.parametrize(
  "bad_order",
  [
    ["pong", "breakout"],  # subset: would silently drop a task
    ["pong", "breakout", "asterix", "seaquest"],  # unknown task
    ["pong", "breakout", "seaquest"],  # same length, wrong member
  ],
)
def test_wrong_task_order_raises(bad_order):
  df = _long_frame(_dense((len(_RUNS), len(_TASKS))))
  with pytest.raises(ValueError, match="task_order"):
    data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score", task_order=bad_order)


def test_missing_combination_raises_and_names_it():
  df = _long_frame(_dense((len(_RUNS), len(_TASKS))))
  df = df[~((df["run"] == 1) & (df["task"] == "pong"))]
  with pytest.raises(ValueError) as excinfo:
    data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score")
  message = str(excinfo.value)
  assert "pong" in message and "(1, 'pong')" in message


def test_missing_step_combination_raises():
  df = _long_frame(_dense((len(_RUNS), len(_TASKS), len(_STEPS))), steps=_STEPS)
  df = df[~((df["run"] == 0) & (df["task"] == "asterix") & (df["step"] == 20))]
  with pytest.raises(ValueError) as excinfo:
    data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score", step_col="step")
  assert "(0, 'asterix', 20)" in str(excinfo.value)


def test_duplicate_rows_raise():
  df = _long_frame(_dense((len(_RUNS), len(_TASKS))))
  df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
  with pytest.raises(ValueError) as excinfo:
    data.load_scores_from_dataframe(df, run_col="run", task_col="task", score_col="score")
  message = str(excinfo.value)
  assert "duplicated" in message and "breakout" in message


@pytest.mark.parametrize("missing_col", ["run_col", "task_col", "score_col"])
def test_missing_column_raises_key_error(missing_col):
  df = _long_frame(_dense((len(_RUNS), len(_TASKS))))
  kwargs = {"run_col": "run", "task_col": "task", "score_col": "score"}
  kwargs[missing_col] = "nonexistent"
  with pytest.raises(KeyError, match="nonexistent"):
    data.load_scores_from_dataframe(df, **kwargs)


def test_score_dict_loader_aligns_tasks():
  # The second algorithm's rows are ordered differently on purpose; the loader
  # must still return arrays whose columns line up task by task.
  scores = {alg: _dense((len(_RUNS), len(_TASKS)), seed=i) for i, alg in enumerate(_ALGORITHMS)}
  frames = [_long_frame(scores[alg], algorithm=alg) for alg in _ALGORITHMS]
  df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=1)
  score_dict, tasks = data.load_score_dict_from_dataframe(
    df, algorithm_col="algorithm", run_col="run", task_col="task", score_col="score"
  )
  assert sorted(score_dict) == sorted(_ALGORITHMS)
  assert tasks == sorted(_TASKS)
  order = [_TASKS.index(t) for t in tasks]
  for alg in _ALGORITHMS:
    assert score_dict[alg].shape == (len(_RUNS), len(_TASKS))
    np.testing.assert_allclose(score_dict[alg], scores[alg][:, order])


def test_score_dict_loader_task_order_shared():
  scores = {alg: _dense((len(_RUNS), len(_TASKS)), seed=i) for i, alg in enumerate(_ALGORITHMS)}
  df = pd.concat([_long_frame(scores[alg], algorithm=alg) for alg in _ALGORITHMS], ignore_index=True)
  wanted = ["pong", "asterix", "breakout"]
  score_dict, tasks = data.load_score_dict_from_dataframe(
    df, algorithm_col="algorithm", run_col="run", task_col="task", score_col="score", task_order=wanted
  )
  assert tasks == wanted
  order = [_TASKS.index(t) for t in wanted]
  for alg in _ALGORITHMS:
    np.testing.assert_allclose(score_dict[alg], scores[alg][:, order])


def test_score_dict_loader_rejects_mismatched_tasks():
  # `rainbow` is missing a task entirely, so it cannot share `dqn`'s ordering.
  scores = {alg: _dense((len(_RUNS), len(_TASKS)), seed=i) for i, alg in enumerate(_ALGORITHMS)}
  rainbow = _long_frame(scores["rainbow"], algorithm="rainbow")
  rainbow = rainbow[rainbow["task"] != "pong"]
  df = pd.concat([_long_frame(scores["dqn"], algorithm="dqn"), rainbow], ignore_index=True)
  with pytest.raises(ValueError) as excinfo:
    data.load_score_dict_from_dataframe(df, algorithm_col="algorithm", run_col="run", task_col="task", score_col="score")
  assert "rainbow" in str(excinfo.value)


def test_score_dict_loader_with_steps():
  scores = {alg: _dense((len(_RUNS), len(_TASKS), len(_STEPS)), seed=i) for i, alg in enumerate(_ALGORITHMS)}
  df = pd.concat([_long_frame(scores[alg], steps=_STEPS, algorithm=alg) for alg in _ALGORITHMS], ignore_index=True)
  score_dict, tasks = data.load_score_dict_from_dataframe(
    df, algorithm_col="algorithm", run_col="run", task_col="task", score_col="score", step_col="step"
  )
  order = [_TASKS.index(t) for t in tasks]
  for alg in _ALGORITHMS:
    assert score_dict[alg].shape == (len(_RUNS), len(_TASKS), len(_STEPS))
    np.testing.assert_allclose(score_dict[alg], scores[alg][:, order])
