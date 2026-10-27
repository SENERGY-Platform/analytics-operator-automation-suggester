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
Power readings to on/off episodes for a device that measures its consumption.

A reading is a step function: each row holds its value until the next row. The history of the operator is read in
buckets (maximum per bucket) with unchanged consecutive buckets left out, so rows are far apart where the device
stands still; everything here weighs a row by the time it holds.

The thresholds come from the data: the log of the power is split in a low and a high class (Otsu), the geometric mean
of the top of the low class and the bottom of the high class, rounded to the 1-2-5 series, is the switch-on
threshold and half of it the switch-off threshold. Only a device with two clearly separate levels is a trigger.
"""

__all__ = ("PowerSpec", "PowerAnalysis", "analyze", "round_125", "HOLD_S", "MIN_GAP_W", "MIN_ON_SHARE",
           "MAX_ON_SHARE", "MAX_ON_PER_DAY", "MIN_CYCLE_PER_DAY", "CYCLE_CV")

import dataclasses
import math
import typing

import numpy as np

# A new state counts when it stands HOLD_S plus one bucket, "on" and "off" alike: the span the exported hold_s gives
# the process, so history and runtime count the same switchings.
HOLD_S = 60
# The classes must differ by at least this much, and the share of time in "on" lies in between.
MIN_GAP_W = 5.0
MIN_ON_SHARE = 0.005
MAX_ON_SHARE = 0.95
# A device that switches on more often than this, or this often and at regular intervals, is a thermostat.
MAX_ON_PER_DAY = 12.0
MIN_CYCLE_PER_DAY = 3.0
CYCLE_CV = 0.3
# The last row holds no known time; count it as one bucket. A row weighs at most a day, so a device that was offline
# for weeks with its last value standing does not decide the levels.
LAST_ROW_S = 60.0
MAX_ROW_S = 86400.0
DAY_S = 86400.0


@dataclasses.dataclass(frozen=True)
class PowerSpec:
    on_above_w: float
    off_below_w: float
    hold_s: int = HOLD_S


@dataclasses.dataclass
class PowerAnalysis:
    suitable: bool
    reason: str  # why not, empty when suitable
    spec: typing.Optional[PowerSpec] = None
    low_hi: typing.Optional[float] = None
    high_lo: typing.Optional[float] = None
    on_share: typing.Optional[float] = None
    on_per_day: typing.Optional[float] = None
    # (epoch seconds, "on" or "off"): the start of every confirmed change
    events: typing.List[typing.Tuple[float, str]] = dataclasses.field(default_factory=list)


def round_125(value: float) -> float:
    """The nearest number of the 1-2-5 series (..., 1, 2, 5, 10, 20, ...), nearest on the log scale."""
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"no 1-2-5 number for {value!r}")
    exponent = math.floor(math.log10(value))
    mantissa = value / 10.0 ** exponent
    best = min((1, 2, 5, 10), key=lambda m: abs(math.log(mantissa / m)))
    return float(f"{best}e{exponent}")


def _number(value: float) -> typing.Union[int, float]:
    return int(value) if float(value).is_integer() else value


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cum, q * cum[-1], side="left")), len(v) - 1)])


def _otsu_split(x: np.ndarray, w: np.ndarray) -> typing.Optional[float]:
    """The threshold on x that maximises the variance between the classes, weighted; None for one level only."""
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    w0 = np.cumsum(ws)
    s0 = np.cumsum(ws * xs)
    total_w, total_s = w0[-1], s0[-1]
    w1 = total_w - w0
    valid = (xs[:-1] < xs[1:]) & (w0[:-1] > 0) & (w1[:-1] > 0)
    if not valid.any():
        return None
    m0 = s0[:-1] / np.where(w0[:-1] > 0, w0[:-1], 1)
    m1 = (total_s - s0[:-1]) / np.where(w1[:-1] > 0, w1[:-1], 1)
    between = np.where(valid, w0[:-1] * w1[:-1] * (m0 - m1) ** 2, -1.0)
    i = int(np.argmax(between))
    return float((xs[i] + xs[i + 1]) / 2)


def _episodes(times: np.ndarray, power: np.ndarray, on: float, off: float):
    """[state, start, end] of the raw hysteresis states; between the thresholds the previous state stays."""
    level = np.where(power > on, 1, np.where(power < off, 0, -1))
    definite = np.flatnonzero(level >= 0)
    if len(definite) == 0:
        return []
    states = level[definite]
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = [float(times[definite[0]])] + [float(times[definite[c]]) for c in changes]
    kinds = [int(states[0])] + [int(states[c]) for c in changes]
    ends = starts[1:] + [float(times[-1])]
    return [[k, s, e] for k, s, e in zip(kinds, starts, ends)]


def _confirm(episodes, bucket_s: float) -> typing.List[list]:
    """
    Drops what does not stand: an "on" or "off" of HOLD_S plus a bucket or less. The first and the last
    episode have no known start or end and stay. The neighbours of a dropped episode have the same state and join.
    """
    out: typing.List[list] = []
    for episode in episodes:
        out.append(list(episode))
        while len(out) >= 3:
            mid = out[-2]
            duration = mid[2] - mid[1]
            # The maximum per bucket stretches an "on" by up to a bucket: a spike of seconds fills one or two buckets.
            # What lasts HOLD_S in reality therefore measures more than HOLD_S plus a bucket, and a shorter one
            # is no switching on; the same holds for "off".
            stands = duration > HOLD_S + bucket_s
            if stands:
                break
            last = out.pop()
            out.pop()
            out[-1][2] = last[2]
    return out


def _unsuitable(reason: str, **kw) -> PowerAnalysis:
    return PowerAnalysis(False, reason, **kw)


def analyze(times: typing.Sequence[float], values: typing.Sequence, bucket_s: float = 60.0) -> PowerAnalysis:
    """
    Thresholds, suitability and events of one power series. times ascending, values raw; bucket_s is the bucket the
    series was read in (0 for raw readings).
    """
    rows = []
    for t, v in zip(times, values):
        if isinstance(v, (bool, np.bool_)):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and math.isfinite(t):
            rows.append((float(t), max(f, 0.0)))
    if len(rows) < 2:
        return _unsuitable("too few readings")
    t = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    w = np.append(np.minimum(np.diff(t), MAX_ROW_S), LAST_ROW_S)
    x = np.log10(p + 0.1)

    split = _otsu_split(x, w)
    if split is None:
        return _unsuitable("one level only")
    low, high = x <= split, x > split
    low_hi = _weighted_quantile(p[low], w[low], 0.95)
    high_lo = _weighted_quantile(p[high], w[high], 0.05)
    on_share = float(w[high].sum() / w.sum())
    info = dict(low_hi=low_hi, high_lo=high_lo, on_share=on_share)

    on = round_125(math.sqrt(max(low_hi, 1.0) * high_lo))
    off = on / 2
    # The process holds a change this long before it counts, the same span after which the history counts it:
    # HOLD_S plus a bucket, because a bucket's maximum stretches a short spike by up to one bucket.
    spec = PowerSpec(_number(on), _number(off), int(round(HOLD_S + bucket_s)))
    if not low_hi < off:
        return _unsuitable("the low level reaches the switch-off threshold", spec=spec, **info)
    if not high_lo > on:
        return _unsuitable("the high level reaches the switch-on threshold", spec=spec, **info)
    if high_lo - low_hi < MIN_GAP_W:
        return _unsuitable("the levels are less than 5 W apart", spec=spec, **info)
    if not MIN_ON_SHARE <= on_share <= MAX_ON_SHARE:
        return _unsuitable("the share of time on is out of range", spec=spec, **info)

    episodes = _confirm(_episodes(t, p, float(on), float(off)), bucket_s)
    events = [(e[1], "on" if e[0] == 1 else "off") for e in episodes[1:]]
    ons = np.array([s for s, k in events if k == "on"])
    span_days = max((t[-1] - t[0]) / DAY_S, 1.0)
    per_day = len(ons) / span_days
    info["on_per_day"] = per_day
    cycling = per_day > MAX_ON_PER_DAY
    if not cycling and per_day >= MIN_CYCLE_PER_DAY and len(ons) >= 3:
        gaps = np.diff(ons)
        cycling = float(gaps.mean()) > 0 and float(gaps.std() / gaps.mean()) < CYCLE_CV
    if cycling:
        return _unsuitable("switches on in a regular cycle", spec=spec, **info)
    return PowerAnalysis(True, "", spec, events=events, **info)
