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
"If A then B" patterns from device events.

A pair is a trigger event A (a state change, a press, a movement) and a target event B on another device. B counts as
a hit when it lies in [a - window_before_s, a + window_after_s]. The baseline is the same window at the same local
clock time on the 7 days before and after, so a daily routine is not mistaken for a coupling of A and B. A pair that
is no routine is cut down to the hours and weekdays it occurs in, and judged there.
"""

__all__ = ("Params", "mine", "rule_key", "suggestion_id", "list_version", "BASELINE_DAYS")

import dataclasses
import datetime
import hashlib
import json
import typing
import zoneinfo

import numpy as np

from . import window
from .events import BURST_S, TARGET_PORTS, SignalEvents, is_target

STATE_PORTS = ("binary_state", "contact_state", "position") + TARGET_PORTS

BASELINE_DAYS = (-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7)
DEFAULT_TZ = "Europe/Berlin"
# A target that is switched on and off again quickly is a candidate for switching off by itself.
AUTO_OFF_SHARE = 0.7
AUTO_OFF_SHORT_S = 600.0
# R1: a switching that this many other devices repeat within GROUP_S is a group control.
GROUP_S = 60.0
# R3: trigger series whose events fall together within CLUSTER_S in CLUSTER_SHARE of the cases are one source.
CLUSTER_S = 2.0
CLUSTER_SHARE = 0.8


@dataclasses.dataclass
class Params:
    window_before_s: float = 300.0
    window_after_s: float = 600.0
    min_hits: int = 10
    min_lift: float = 3.0
    min_confidence: float = 0.25
    # A binary target that is switched off needs this much; an off is what happens anyway after most things.
    min_confidence_off: float = 0.5
    coupling_max_delay_s: float = 2.0
    # A pair whose median signed delay of the hits is at or below this is no consequence of the trigger.
    min_median_delay_s: float = -10.0
    max_suggestions: int = 10
    # Device ids that count as controllable; empty means every device on the target port.
    targets: typing.Tuple[str, ...] = ()

    def zone(self) -> zoneinfo.ZoneInfo:
        return zoneinfo.ZoneInfo(DEFAULT_TZ)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rule_key(trigger: dict, win: dict, target_tokens: typing.Iterable[typing.Sequence[str]]) -> str:
    """
    Trigger, window and the set of targets with their value. Numbers like the hit count are not part of it,
    or the list would be new every day.
    """
    return json.dumps({
        "t": [trigger["device_id"], trigger["service_id"], trigger["path"], trigger["kind"], trigger["from"], trigger["to"]],
        "w": [win["from"], win["to"], list(win["weekdays"])],
        "g": sorted(list(t) for t in target_tokens),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def suggestion_id(key: str) -> str:
    return _sha16(key)


def list_version(ids: typing.Iterable[str]) -> str:
    """Over the sorted ids, each of which is the hash of a rule key; statistics are not in it."""
    return _sha16("\n".join(sorted(ids)))


@dataclasses.dataclass
class _Trigger:
    events: SignalEvents
    token: str
    times: np.ndarray
    hours: np.ndarray
    weekdays: np.ndarray
    shifted: np.ndarray  # (n, 14) same clock time on the baseline days


@dataclasses.dataclass
class _Rule:
    trigger: _Trigger
    target: SignalEvents
    token: str
    mask: np.ndarray  # occurrences inside the window
    hit: np.ndarray  # per masked occurrence
    base: np.ndarray  # (masked n, 14)
    delay: np.ndarray  # per masked occurrence, nan without a hit
    win: typing.Tuple[int, int]
    weekdays: typing.List[int]
    lift: float
    hits: int
    hit_full: np.ndarray = None  # over all occurrences, before the window


def _by_token(se: SignalEvents, episode_gap_s: typing.Optional[float] = None) -> typing.Dict[str, np.ndarray]:
    """
    Event times per token. With episode_gap_s, events of one token that follow the previous one within that time
    count as one occurrence, at the time of the first (a motion sensor that reports every few seconds).
    """
    out: typing.Dict[str, typing.List[float]] = {}
    for t, token in se.events:
        out.setdefault(token, []).append(t)
    result = {}
    for k, v in out.items():
        arr = np.sort(np.asarray(v, dtype=float))
        if episode_gap_s is not None and len(arr) > 1:
            starts = np.concatenate([[True], np.diff(arr) > episode_gap_s])
            arr = arr[starts]
        result[k] = arr
    return result


def _local_fields(times: np.ndarray, zone) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hour, weekday and baseline-day times of each event, in local clock time."""
    n = len(times)
    hours = np.empty(n, dtype=int)
    weekdays = np.empty(n, dtype=int)
    shifted = np.empty((n, len(BASELINE_DAYS)))
    midnights: typing.Dict[int, float] = {}

    def midnight(ordinal: int) -> float:
        m = midnights.get(ordinal)
        if m is None:
            d = datetime.date.fromordinal(ordinal)
            m = midnights[ordinal] = datetime.datetime(d.year, d.month, d.day, tzinfo=zone).timestamp()
        return m

    for i, t in enumerate(times):
        dt = datetime.datetime.fromtimestamp(t, zone)
        hours[i] = dt.hour
        weekdays[i] = dt.weekday()
        wall = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        o = dt.date().toordinal()
        for j, k in enumerate(BASELINE_DAYS):
            shifted[i, j] = midnight(o + k) + wall
    return hours, weekdays, shifted


def _hit(tb: np.ndarray, x: np.ndarray, before: float, after: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Whether a target event lies in [x - before, x + after], and the signed delay of the closest one."""
    m = len(tb)
    j = np.searchsorted(tb, x, side="left")
    right = tb[np.minimum(j, m - 1)] - x
    left = tb[np.maximum(j - 1, 0)] - x
    right_ok = (j < m) & (right <= after)
    left_ok = (j > 0) & (left >= -before)
    delay = np.full(len(x), np.nan)
    both = right_ok & left_ok
    only_r = right_ok & ~left_ok
    only_l = left_ok & ~right_ok
    delay[only_r] = right[only_r]
    delay[only_l] = left[only_l]
    pick_right = np.abs(right) <= np.abs(left)
    delay[both] = np.where(pick_right, right, left)[both]
    return right_ok | left_ok, delay


def _hit_matrix(tb: np.ndarray, shifted: np.ndarray, before: float, after: float) -> np.ndarray:
    flat = shifted.reshape(-1)
    hit, _ = _hit(tb, flat, before, after)
    return hit.reshape(shifted.shape)


def _is_coupling(hit: np.ndarray, delay: np.ndarray, p: Params) -> bool:
    """B switches within moments of A: an existing coupling, nothing to suggest."""
    if hit.sum() == 0:
        return False
    return float(np.median(np.abs(delay[hit]))) <= p.coupling_max_delay_s


def _trigger_fields(se: SignalEvents, token: str) -> typing.Tuple[typing.Optional[str], str]:
    """(from, to) of the trigger. A press or a movement has no previous state."""
    if se.kind == "binary":
        return ("on" if token == "off" else "off"), token
    if se.kind == "contact":
        return ("open" if token == "closed" else "closed"), token
    return None, token


def _target_value(se: SignalEvents, token: str):
    if se.kind == "binary":
        return token == "on"
    if se.kind == "level":
        return round(se.level_values[token], 4)
    return token


def _on_durations(se: SignalEvents) -> typing.List[float]:
    durations = []
    on_at = None
    for t, tok in se.events:
        if tok == "on":
            on_at = t
        elif tok == "off" and on_at is not None:
            durations.append(t - on_at)
            on_at = None
    return durations


def _auto_off_s(se: SignalEvents, token: str, p: Params) -> typing.Optional[int]:
    if se.kind != "binary" or token != "on":
        return None
    durations = _on_durations(se)
    if len(durations) < p.min_hits:
        return None
    short = sum(1 for d in durations if d < AUTO_OFF_SHORT_S)
    if short < AUTO_OFF_SHARE * len(durations):
        return None
    return max(1, int(round(float(np.median(durations)))))


def _rule_in_mask(trig: _Trigger, target: SignalEvents, token: str, tb: np.ndarray, p: Params,
                  hit: np.ndarray, delay: np.ndarray, mask: np.ndarray,
                  win: typing.Tuple[int, int], days: typing.List[int]) -> typing.Optional[_Rule]:
    """The pair judged on the occurrences inside the window."""
    n = int(mask.sum())
    hit_m, delay_m = hit[mask], delay[mask]
    hits = int(hit_m.sum())
    if n == 0 or hits < p.min_hits or _is_coupling(hit_m, delay_m, p):
        return None
    if float(np.median(delay_m[hit_m])) <= p.min_median_delay_s:
        return None  # R10: a target that mostly switches before the trigger is no consequence of it
    confidence = hits / n
    base = _hit_matrix(tb, trig.shifted[mask], p.window_before_s, p.window_after_s)
    baseline = max(float(base.mean()), 1.0 / base.size)
    lift = confidence / baseline
    needed = p.min_confidence_off if target.kind == "binary" and token == "off" else p.min_confidence
    if confidence < needed or lift < p.min_lift:
        return None
    return _Rule(trig, target, token, mask, hit_m, base, delay_m, win, days, lift, hits, hit)


def _mask_for(trig: _Trigger, hit: np.ndarray):
    start, length = window.hour_window(trig.hours[hit])
    mask = np.array([window.in_hours(int(h), start, length) for h in trig.hours])
    days = window.weekdays_for(trig.weekdays[hit & mask])
    mask &= np.isin(trig.weekdays, days)
    return mask, (start, length), days


BLOCK_MIN_HITS = 2
BLOCK_SHARE = 0.05


def hour_blocks(hit_hours: typing.Sequence[int]) -> typing.List[typing.Tuple[int, int]]:
    """
    R5: (start, length) of each cyclic run of local hours that carry at least max(2, 5 %) of the hits.
    """
    counts = np.bincount(np.asarray(hit_hours, dtype=int) % 24, minlength=24)
    need = max(BLOCK_MIN_HITS, BLOCK_SHARE * int(counts.sum()))
    active = counts >= need
    if not active.any():
        return []
    if active.all():
        return [(0, 24)]
    blocks = []
    for h in range(24):
        if active[h] and not active[(h - 1) % 24]:
            length = 1
            while active[(h + length) % 24]:
                length += 1
            blocks.append((h, length))
    return blocks


def _prefer_on(rules: typing.List[_Rule], p: Params) -> typing.List[_Rule]:
    """
    R2: a light that is on only for a short time is suggested as "on" with the automatic off, never as "off" (the
    off follows the on). If the "on" does not qualify, the target is no suggestion at all.
    """
    short: typing.Dict[tuple, bool] = {}
    out = []
    for r in rules:
        key = r.target.signal.key
        if key not in short:
            durations = _on_durations(r.target) if r.target.kind == "binary" else []
            short[key] = bool(durations) and float(np.median(durations)) < p.window_before_s + p.window_after_s
        if not (short[key] and r.token == "off"):
            out.append(r)
    return out


def _group_action_events(signals: typing.Sequence[SignalEvents]) -> typing.Dict[str, np.ndarray]:
    """Per device, the times of target events that at least two other devices repeat within +-GROUP_S."""
    by_token: typing.Dict[str, typing.List[typing.Tuple[float, str]]] = {}
    for se in signals:
        if is_target(se.signal.port):
            for t, token in se.events:
                by_token.setdefault(token, []).append((t, se.signal.device_id))
    flagged: typing.Dict[str, typing.List[float]] = {}
    for items in by_token.values():
        items.sort()
        times = np.array([t for t, _ in items])
        for i, (t, device) in enumerate(items):
            lo = int(np.searchsorted(times, t - GROUP_S, side="left"))
            hi = int(np.searchsorted(times, t + GROUP_S, side="right"))
            others = {items[j][1] for j in range(lo, hi)} - {device}
            if len(others) >= 2:
                flagged.setdefault(device, []).append(t)
    return {d: np.sort(np.array(v)) for d, v in flagged.items()}


def _near_any(times: np.ndarray, marks: np.ndarray, within: float) -> np.ndarray:
    j = np.searchsorted(marks, times)
    left = np.abs(times - marks[np.maximum(j - 1, 0)])
    right = np.abs(marks[np.minimum(j, len(marks) - 1)] - times)
    return np.minimum(left, right) <= within


def _drop_group_actions(signals: typing.List[SignalEvents]) -> typing.List[SignalEvents]:
    """R1: what a group control does is automated already, no target and no trigger."""
    flagged = _group_action_events(signals)
    if not flagged:
        return signals
    out = []
    for se in signals:
        marks = flagged.get(se.signal.device_id)
        if marks is None or not se.events:
            out.append(se)
            continue
        times = np.array([t for t, _ in se.events])
        # A trigger port of the same device reports the same switching, at the same moment.
        within = 0.0 if is_target(se.signal.port) else BURST_S
        drop = _near_any(times, marks, within)
        out.append(dataclasses.replace(se, events=[e for e, d in zip(se.events, drop) if not d]))
    return out


def _coincide(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the events of the smaller series that the other has within CLUSTER_S."""
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return float(_near_any(small, large, CLUSTER_S).mean())


def _trigger_clusters(signals: typing.Sequence[SignalEvents], controllable_devices: typing.Set[str],
                      p: Params) -> typing.Tuple[typing.Set[tuple], typing.Dict[tuple, typing.Set[str]]]:
    """
    R3: trigger series that switch at the same moments are one source. Returns the keys of the dropped members
    and, for each representative, the devices of its cluster.
    """
    trig = [se for se in signals if not is_target(se.signal.port) and len(se.events) >= p.min_hits]
    times = [np.sort(np.array([t for t, _ in se.events])) for se in trig]
    parent = list(range(len(trig)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(trig)):
        for j in range(i + 1, len(trig)):
            if trig[i].signal.device_id != trig[j].signal.device_id and _coincide(times[i], times[j]) >= CLUSTER_SHARE:
                parent[find(i)] = find(j)
    groups: typing.Dict[int, typing.List[int]] = {}
    for i in range(len(trig)):
        groups.setdefault(find(i), []).append(i)
    dropped: typing.Set[tuple] = set()
    devices: typing.Dict[tuple, typing.Set[str]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = max(members, key=lambda i: (trig[i].signal.device_id in controllable_devices, len(trig[i].events),
                                          trig[i].signal.key))
        devices[(trig[rep].signal.key, trig[rep].signal.port)] = {trig[i].signal.device_id for i in members}
        dropped |= {(trig[i].signal.key, trig[i].signal.port) for i in members if i != rep}
    return dropped, devices


def _suggestion(group: typing.List[_Rule], p: Params) -> dict:
    first = group[0]
    trig = first.trigger
    se = trig.events
    start, length = first.win
    kind_from, kind_to = _trigger_fields(se, trig.token)
    trigger = {"device_id": se.signal.device_id, "service_id": se.signal.service_id, "path": se.signal.path,
               "kind": se.kind, "from": kind_from, "to": kind_to}
    win = {"from": window.format_hour(start), "to": window.format_hour(start + length),
           "weekdays": list(first.weekdays), "tz": DEFAULT_TZ}
    group = sorted(group, key=lambda r: (r.target.signal.key, r.token))

    hit = np.zeros(len(first.hit), dtype=bool)
    base = np.zeros(first.base.shape, dtype=bool)
    delay = np.full(len(first.hit), np.nan)
    for r in group:
        hit |= r.hit
        base |= r.base
        closer = ~np.isnan(r.delay) & (np.isnan(delay) | (np.abs(r.delay) < np.abs(delay)))
        delay[closer] = r.delay[closer]
    n = len(hit)
    hits = int(hit.sum())
    confidence = hits / n
    baseline = max(float(base.mean()), 1.0 / base.size)

    targets = [{"device_id": r.target.signal.device_id, "service_id": r.target.signal.service_id,
                "path": r.target.signal.path, "function_id": None, "value": _target_value(r.target, r.token)}
               for r in group]
    offs = [_auto_off_s(r.target, r.token, p) for r in group]
    auto_off = max(offs) if all(o is not None for o in offs) else None
    key = rule_key(trigger, win, [(r.target.signal.device_id, r.target.signal.service_id, r.target.signal.path, r.token)
                                  for r in group])
    return {
        "id": suggestion_id(key),
        "trigger": trigger,
        "window": win,
        "targets": targets,
        "auto_off_s": auto_off,
        "stats": {"occurrences": n, "hits": hits, "confidence": round(confidence, 4), "baseline": round(baseline, 4),
                  "lift": round(confidence / baseline, 2), "median_delay_s": round(float(np.median(delay[hit])), 1)},
    }


def _rank(r: "_Rule") -> float:
    return r.lift * r.hits ** 0.5


def _one_direction(entries: typing.List[typing.Tuple[_Trigger, typing.List[_Rule]]]):
    """R8: of the two directions between two devices only the better ranked stays."""
    best: typing.Dict[frozenset, typing.Tuple[float, tuple]] = {}
    for trig, rules in entries:
        for r in rules:
            a, b = trig.events.signal.device_id, r.target.signal.device_id
            pair = frozenset((a, b))
            if pair not in best or _rank(r) > best[pair][0]:
                best[pair] = (_rank(r), (a, b))
    out = []
    for trig, rules in entries:
        kept = [r for r in rules
                if best[frozenset((trig.events.signal.device_id, r.target.signal.device_id))][1]
                == (trig.events.signal.device_id, r.target.signal.device_id)]
        if kept:
            out.append((trig, kept))
    return out


def _drop_self_stated_buttons(signals: typing.List[SignalEvents]) -> typing.List[SignalEvents]:
    """R7: a button of a device that reports its own state or position only repeats what that state says."""
    stated = {se.signal.device_id for se in signals if se.signal.port in STATE_PORTS}
    return [se for se in signals if not (se.signal.port == "button_press" and se.signal.device_id in stated)]


def mine(signals: typing.Sequence[SignalEvents], p: Params) -> typing.List[dict]:
    """The suggestions, best first, at most p.max_suggestions. `signals` are the extracted events of every series."""
    controllable = set(p.targets)
    zone = p.zone()
    signals = [se for se in signals
               if not is_target(se.signal.port) or not controllable or se.signal.device_id in controllable]
    signals = _drop_self_stated_buttons(signals)
    signals = _drop_group_actions(list(signals))
    target_devices = {se.signal.device_id for se in signals if is_target(se.signal.port)}
    dropped, cluster_devices = _trigger_clusters(signals, target_devices, p)

    triggers: typing.List[_Trigger] = []
    targets: typing.List[typing.Tuple[SignalEvents, str, np.ndarray]] = []
    for se in signals:
        if is_target(se.signal.port):
            for token, times in _by_token(se).items():
                if len(times) >= p.min_hits:
                    targets.append((se, token, times))
        elif (se.signal.key, se.signal.port) not in dropped:
            for token, times in _by_token(se, p.window_after_s).items():
                if len(times) >= p.min_hits:
                    hours, weekdays, shifted = _local_fields(times, zone)
                    triggers.append(_Trigger(se, token, times, hours, weekdays, shifted))

    entries: typing.List[typing.Tuple[_Trigger, typing.List[_Rule]]] = []
    for trig in triggers:
        sig = trig.events.signal
        own_cluster = cluster_devices.get((sig.key, sig.port), set())
        cands = []
        for target, token, tb in targets:
            device = target.signal.device_id
            if device == sig.device_id or device in own_cluster:
                continue
            hit, delay = _hit(tb, trig.times, p.window_before_s, p.window_after_s)
            if hit.sum() < p.min_hits or _is_coupling(hit, delay, p):
                continue
            cands.append((target, token, tb, hit, delay))
        # Each pair in the hours it occurs in; their hits tell where the trigger has hits at all.
        first = []
        for target, token, tb, hit, delay in cands:
            mask, win, days = _mask_for(trig, hit)
            rule = _rule_in_mask(trig, target, token, tb, p, hit, delay, mask, win, days)
            if rule is not None:
                first.append(rule)
        first = _prefer_on(first, p)
        if not first:
            continue
        any_hit = np.zeros(len(trig.times), dtype=bool)
        for r in first:
            any_hit |= r.hit_full
        # R5: every block of hours is a window of its own, with every target judged in it.
        for start, length in hour_blocks(trig.hours[any_hit]):
            in_block = np.array([window.in_hours(int(h), start, length) for h in trig.hours])
            days = window.weekdays_for(trig.weekdays[any_hit & in_block])
            mask = in_block & np.isin(trig.weekdays, days)
            rules = []
            for target, token, tb, hit, delay in cands:
                rule = _rule_in_mask(trig, target, token, tb, p, hit, delay, mask, (start, length), days)
                if rule is not None:
                    rules.append(rule)
            rules = _prefer_on(rules, p)
            if rules:
                entries.append((trig, rules))

    out = []
    for trig, rules in _one_direction(entries):
        # One target device and path takes one value per suggestion; the better supported wins.
        best: typing.Dict[tuple, _Rule] = {}
        for r in sorted(rules, key=lambda r: (-r.hits, -r.lift, r.token)):
            best.setdefault(r.target.signal.key, r)
        out.append(_suggestion(list(best.values()), p))
    out.sort(key=lambda s: (-s["stats"]["lift"] * s["stats"]["hits"] ** 0.5, s["id"]))
    return out[:max(0, p.max_suggestions)]
