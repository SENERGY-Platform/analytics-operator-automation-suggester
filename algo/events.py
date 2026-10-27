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

"""
Raw time series to discrete events (state changes, presses, movements).

A series is a Signal (device, service, path, input port) with sorted epoch seconds and raw values. The port decides
how a value is read, so nothing is guessed from the value alone. Every extractor returns a SignalEvents: the
sorted (time, token) pairs, where the token names what happened ("on", "off", "closed", "up", "L3", an action text).
"""

__all__ = ("Signal", "SignalEvents", "PORTS", "TRIGGER_PORTS", "TARGET_PORT", "TARGET_PORTS", "is_target", "BURST_S", "SETTLE_S", "LEVELS",
           "debounce", "extract")

import dataclasses
import math
import typing

import numpy as np

from .power import PowerSpec, analyze as analyze_power

# `power` is a trigger only: the state "on" or "off" of a device that measures its consumption, with thresholds
# learned from the history (algo/power.py).
TRIGGER_PORTS = ("binary_state", "contact_state", "button_press", "position", "power")
TARGET_PORT = "target_state"
# Always numeric, one input each because the engine ANDs the criteria of a port: a position (fraction or percent)
# and a speed of a controllable device.
NUMERIC_TARGET_PORTS = ("target_position", "target_speed")
TARGET_PORTS = (TARGET_PORT,) + NUMERIC_TARGET_PORTS
PORTS = TRIGGER_PORTS + TARGET_PORTS

# Messages closer together than this are one burst: devices repeat a report and sometimes contradict themselves
# with the same timestamp.
BURST_S = 2.0
# A numeric target value counts as changed only when the new level stands this long.
SETTLE_S = 60.0
LEVELS = 10
# A position moves by at least a tenth to count; the same direction within this time is one movement.
POSITION_STEP = 0.1
POSITION_SAME_MOVE_S = 180.0

_ON = {"on", "true", "1", "1.0"}
_OFF = {"off", "false", "0", "0.0"}
_CLOSED = {"closed", "close", "true", "on", "1", "1.0"}
_OPEN = {"open", "opened", "false", "off", "0", "0.0"}
_BLANK = {"", "none", "null", "nan"}


def is_target(port: str) -> bool:
    return port in TARGET_PORTS


@dataclasses.dataclass(frozen=True)
class Signal:
    device_id: str
    service_id: str
    path: str
    port: str

    @property
    def key(self) -> typing.Tuple[str, str, str]:
        return self.device_id, self.service_id, self.path


@dataclasses.dataclass
class SignalEvents:
    signal: Signal
    kind: str  # "binary", "contact", "button", "position", "power" or, for the target port, "binary", "text", "level"
    events: typing.List[typing.Tuple[float, str]]
    # Level kind only: the median raw value of each level, which is what an action sets.
    level_values: typing.Dict[str, float] = dataclasses.field(default_factory=dict)
    # Power kind only: the thresholds the events were read with.
    power: typing.Optional[PowerSpec] = None


def _text(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return str(value).strip().lower()


def _norm_binary(value) -> typing.Optional[str]:
    t = _text(value)
    return "on" if t in _ON else "off" if t in _OFF else None


def _norm_contact(value) -> typing.Optional[str]:
    t = _text(value)
    return "closed" if t in _CLOSED else "open" if t in _OPEN else None


def debounce(times: typing.Sequence[float], values: typing.Sequence[str]) -> typing.Tuple[typing.List[float], typing.List[str]]:
    """
    One value per burst (messages less than BURST_S apart, chained): the value most often in it, the later one on a
    tie. The burst's time is its first message. Input sorted by time.
    """
    out_t: typing.List[float] = []
    out_v: typing.List[str] = []
    burst: typing.List[str] = []
    start = last = None
    for t, v in zip(times, values):
        if burst and t - last < BURST_S:
            burst.append(v)
        else:
            if burst:
                out_t.append(start)
                out_v.append(_majority(burst))
            burst, start = [v], t
        last = t
    if burst:
        out_t.append(start)
        out_v.append(_majority(burst))
    return out_t, out_v


def _majority(burst: typing.List[str]) -> str:
    counts: typing.Dict[str, int] = {}
    for v in burst:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    for v in reversed(burst):
        if counts[v] == best:
            return v
    return burst[-1]  # unreachable


def _changes(times, values) -> typing.List[typing.Tuple[float, str]]:
    """Debounced, then only the transitions; the first value is a state, not an event."""
    times, values = debounce(times, values)
    events = []
    prev = None
    for t, v in zip(times, values):
        if prev is not None and v != prev:
            events.append((t, v))
        prev = v
    return events


def _state_events(signal, times, values, normalize, kind) -> SignalEvents:
    pairs = [(t, normalize(v)) for t, v in zip(times, values)]
    pairs = [(t, v) for t, v in pairs if v is not None]
    return SignalEvents(signal, kind, _changes([t for t, _ in pairs], [v for _, v in pairs]))


def _button_events(signal, times, values) -> SignalEvents:
    events = []
    last: typing.Dict[str, float] = {}
    for t, v in zip(times, values):
        token = _text(v)
        if token in _BLANK:
            continue
        # A repeated report of the same press is one press; the clock runs on with every repeat.
        if t - last.get(token, -math.inf) > BURST_S:
            events.append((t, token))
        last[token] = t
    return SignalEvents(signal, "button", events)


def _position_events(signal, times, values) -> SignalEvents:
    nums = []
    for t, v in zip(times, values):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            nums.append((t, f))
    if nums and max(f for _, f in nums) > 1.0:
        nums = [(t, f / 100.0) for t, f in nums]  # percent
    events = []
    prev = None
    last_move: typing.Tuple[float, typing.Optional[str]] = (-math.inf, None)
    for t, f in nums:
        if prev is not None and abs(f - prev) >= POSITION_STEP - 1e-9:
            direction = "down" if f < prev else "up"
            if not (direction == last_move[1] and t - last_move[0] < POSITION_SAME_MOVE_S):
                events.append((t, direction))
            last_move = (t, direction)
        prev = f
    return SignalEvents(signal, "position", events)


def _target_events(signal, times, values) -> SignalEvents:
    """target_state is binary: any value that is not on/off in some spelling is no event."""
    return _state_events(signal, times, values, _norm_binary, "binary")


def _numeric_target_events(signal, times, values) -> SignalEvents:
    numbers = []
    for t, v in zip(times, values):
        if isinstance(v, (bool, np.bool_)):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            numbers.append((t, f))
    if signal.port == "target_position" and numbers and max(f for _, f in numbers) > 1.0:
        numbers = [(t, f / 100.0) for t, f in numbers]  # percent, as for the position trigger
    if not numbers:
        return SignalEvents(signal, "level", [])
    return _level_events(signal, numbers)


def _level_events(signal, numbers) -> SignalEvents:
    raw = np.array([f for _, f in numbers])
    lo, hi = float(raw.min()), float(raw.max())
    if hi > lo:
        levels = np.minimum((raw - lo) / (hi - lo) * LEVELS, LEVELS - 1).astype(int)
    else:
        levels = np.zeros(len(raw), dtype=int)
    medians = {f"L{lv}": float(np.median(raw[levels == lv])) for lv in np.unique(levels)}
    # Runs of one level; a run that ends sooner than SETTLE_S after it began never settled.
    runs = []
    for i, lv in enumerate(levels):
        if not runs or runs[-1][1] != lv:
            runs.append([numbers[i][0], int(lv)])
    events = []
    stable_level = None
    for i, (start, lv) in enumerate(runs):
        settled = i + 1 >= len(runs) or runs[i + 1][0] - start >= SETTLE_S
        if not settled:
            continue
        if stable_level is not None and lv != stable_level:
            events.append((start, f"L{lv}"))
        stable_level = lv
    return SignalEvents(signal, "level", events, medians)


def _power_events(signal, times, values, bucket_s: float) -> SignalEvents:
    """A device that is no clear on/off device, or cycles, has no events and no thresholds."""
    result = analyze_power(times, values, bucket_s)
    return SignalEvents(signal, "power", result.events, power=result.spec if result.suitable else None)


def extract(signal: Signal, times: typing.Sequence[float], values: typing.Sequence, power_bucket_s: float = 60.0) -> SignalEvents:
    """The events of one series, read by the signal's port. times ascending; power_bucket_s is the bucket of `power`."""
    if signal.port == "binary_state":
        return _state_events(signal, times, values, _norm_binary, "binary")
    if signal.port == "contact_state":
        return _state_events(signal, times, values, _norm_contact, "contact")
    if signal.port == "button_press":
        return _button_events(signal, times, values)
    if signal.port == "position":
        return _position_events(signal, times, values)
    if signal.port == "power":
        return _power_events(signal, times, values, power_bucket_s)
    if signal.port == TARGET_PORT:
        return _target_events(signal, times, values)
    if signal.port in NUMERIC_TARGET_PORTS:
        return _numeric_target_events(signal, times, values)
    raise ValueError(f"unknown port {signal.port!r}")
