"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("fit_rules", "train_rules", "extract_all")

import datetime
import typing

import ray

from .events import Signal, SignalEvents, extract
from .history import signals_from_datasets
from .mining import Params, mine
from .model import AutomationModel


def extract_all(series: typing.Sequence[typing.Tuple[Signal, typing.Sequence[float], typing.Sequence]]) -> typing.List[SignalEvents]:
    return [extract(signal, times, values) for signal, times, values in series]


def fit_rules(datasets, topic_meta, now: datetime.datetime, params: Params, history_days: int):
    """
    The model from the input history, with the log lines and whether a requested history could not be read; runs as
    a Ray task. The model is unpublished.
    """
    notes: typing.List[str] = []
    series, history_failed = signals_from_datasets(datasets, topic_meta, notes)
    signals = extract_all(series)
    rules = mine(signals, params)
    notes.append(f"{len(series)} series, {sum(len(s.events) for s in signals)} events, {len(rules)} suggestions")
    zone = params.zone()
    today = now.astimezone(zone).date()
    history = {"from": (today - datetime.timedelta(days=history_days)).isoformat(), "to": today.isoformat()}
    trained_at = now.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return AutomationModel(rules, trained_at, history), notes, history_failed


# The in-pod Ray Client cannot execute a dataset, the training runs inside a task.
train_rules = ray.remote(num_cpus=1)(fit_rules)
