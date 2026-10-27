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
Mines suggestions from history that is already on disk. Offline: no network, no platform token.

    uv run --frozen python scripts/replay_history.py mapping.json response1.json [response2.json ...]

A response file is the body of one POST /db/v3/queries/v2: a list of elements
{"deviceId", "serviceId", "data": [[[time, value], ...], ...]}, one series in `data` per requested column.
The mapping file says what each series is, a list of {"device_id", "service_id", "path", "port"} with the port one
of binary_state, contact_state, button_press, position, target_state, target_position, target_speed. The series of an element are matched to the
mapping entries of its device and service in file order, so list a service's paths in the order they were requested.
Series without a mapping entry are skipped.

The mining is the one the operator runs; the options are its config values.
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from algo.events import PORTS
from algo.history import signals_from_frames
from algo.mining import Params, mine
from algo.training import extract_all


def load_mapping(path):
    with open(path) as f:
        entries = json.load(f)
    by_service = collections.defaultdict(list)
    for e in entries:
        if e["port"] not in PORTS:
            raise SystemExit(f"unknown port {e['port']!r} in {path}, expected one of {', '.join(PORTS)}")
        by_service[(e["device_id"], e["service_id"])].append(e)
    return by_service


def load_series(mapping, response_files):
    """(signal, epoch seconds, values) per mapped series."""
    frames = []
    for path in response_files:
        with open(path) as f:
            body = json.load(f)
        for element in body if isinstance(body, list) else [body]:
            entries = mapping.get((element.get("deviceId"), element.get("serviceId")))
            if not entries:
                continue
            for entry, rows in zip(entries, element.get("data") or []):
                rows = [r for r in rows or [] if r and len(r) >= 2 and r[1] is not None]
                if not rows:
                    continue
                frame = pd.DataFrame({"time": [r[0] for r in rows], entry["port"]: [r[1] for r in rows]})
                meta = {"name": entry["service_id"], "filterType": "DeviceId", "filterValue": entry["device_id"],
                        "mappings": [{"dest": entry["port"], "source": entry["path"]}]}
                frames.append((frame, meta))
    return signals_from_frames(frames)


def describe(s):
    t = s["trigger"]
    w = s["window"]
    st = s["stats"]
    days = "Mon-Fri" if w["weekdays"] == [0, 1, 2, 3, 4] else "daily"
    when = f"{t['device_id']} {t['path']} {t['from']} -> {t['to']}" if t["from"] else f"{t['device_id']} {t['path']} {t['to']}"
    lines = [f"{s['id']}  {when}  {w['from']}-{w['to']} {days}"]
    for g in s["targets"]:
        lines.append(f"    then {g['device_id']} {g['path']} = {g['value']}")
    off = f", auto off after {s['auto_off_s']} s" if s["auto_off_s"] else ""
    lines.append(f"    {st['hits']} of {st['occurrences']} ({st['confidence']:.0%}, baseline {st['baseline']:.1%}, "
                 f"lift {st['lift']}), median delay {st['median_delay_s']} s{off}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mapping")
    ap.add_argument("responses", nargs="+")
    ap.add_argument("--json", action="store_true", help="print the suggestions as JSON")
    d = Params()
    ap.add_argument("--window-before-s", type=float, default=d.window_before_s)
    ap.add_argument("--window-after-s", type=float, default=d.window_after_s)
    ap.add_argument("--min-hits", type=int, default=d.min_hits)
    ap.add_argument("--min-lift", type=float, default=d.min_lift)
    ap.add_argument("--min-confidence", type=float, default=d.min_confidence)
    ap.add_argument("--min-confidence-off", type=float, default=d.min_confidence_off)
    ap.add_argument("--coupling-max-delay-s", type=float, default=d.coupling_max_delay_s)
    ap.add_argument("--min-median-delay-s", type=float, default=d.min_median_delay_s, help="use --min-median-delay-s=-10 for negative values")
    ap.add_argument("--max-suggestions", type=int, default=d.max_suggestions)
    ap.add_argument("--targets", default="", help="comma separated device ids; empty: every device on a target port")
    a = ap.parse_args()

    series = load_series(load_mapping(a.mapping), a.responses)
    signals = extract_all(series)
    params = Params(a.window_before_s, a.window_after_s, a.min_hits, a.min_lift, a.min_confidence, a.min_confidence_off,
                    a.coupling_max_delay_s, a.min_median_delay_s, a.max_suggestions,
                    tuple(x.strip() for x in a.targets.split(",") if x.strip()))
    found = mine(signals, params)
    print(f"{len(series)} series, {sum(len(s.events) for s in signals)} events, {len(found)} suggestions", file=sys.stderr)
    if a.json:
        print(json.dumps(found, indent=2, ensure_ascii=False))
    else:
        for s in found:
            print(describe(s))


if __name__ == "__main__":
    main()
