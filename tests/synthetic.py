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

"""Synthetic histories. Every id and value is made up; nothing here comes from a real installation."""

import datetime
import zoneinfo

import numpy as np
import pandas as pd

from algo.events import Signal

BERLIN = zoneinfo.ZoneInfo("Europe/Berlin")
START = datetime.date(2026, 1, 5)  # a Monday; the 180 days cross the change to summer time

LAMP = "dev-lamp"
EG = "dev-light-eg"
OG = "dev-light-og"
STRIP = "dev-strip"
SVC = "urn:infai:ses:service:svc-1"


def local(day: int, hour: int, minute: int = 0, second: float = 0.0) -> float:
    """Epoch seconds of `hour:minute` of START + day in Berlin; an hour past 24 belongs to the next day."""
    d = START + datetime.timedelta(days=day + hour // 24)
    dt = datetime.datetime(d.year, d.month, d.day, hour % 24, minute, tzinfo=BERLIN)
    return dt.timestamp() + second


def sig(device, port, path="value.state"):
    return Signal(device, SVC, path, port)


def series(device, port, pairs, path="value.state"):
    """A raw series (signal, times, values); pairs are (time, value) in any order."""
    pairs = sorted(pairs, key=lambda p: p[0])
    return sig(device, port, path), [t for t, _ in pairs], [v for _, v in pairs]


# 20 days of the evening pattern: the first and the last hour carry a quarter of the lamp-off times each, so the
# smallest run of hours with 80 % of the hits is all seven of them, 19 to 01.
EVENING = [19] * 5 + [20, 20, 21, 21, 22, 22, 23, 23, 24, 24] + [25] * 5
DAYS = 180


def evening(light_cycle_s=1800.0, hit_every=3, lamp=LAMP, eg=EG, og=OG, minute_offset=0, og_hours=None):
    """
    Raw series of the evening example. The lamp is on from 18:00 and off in the evening on every day; on two of
    three days (all but every `hit_every`-th) both lights come on 30 s and 45 s later and go off after
    light_cycle_s. With og_hours the second light follows only when the lamp went off in one of those hours.
    """
    lamp_s, eg_s, og_s = [], [], []
    for i in range(DAYS):
        hour = EVENING[i % len(EVENING)]
        off = local(i, hour, (i * 7 + minute_offset) % 60)
        lamp_s += [(local(i, 18), True), (off, False)]
        if i % hit_every != 0:
            eg_s += [(off + 30, True), (off + 30 + light_cycle_s, False)]
            if og_hours is None or hour in og_hours:
                og_s += [(off + 45, True), (off + 45 + light_cycle_s, False)]
        # Both lights are also used at noon, which belongs to nobody.
        eg_s += [(local(i, 12, 5), True), (local(i, 12, 5) + light_cycle_s, False)]
        og_s += [(local(i, 13, 5), True), (local(i, 13, 5) + light_cycle_s, False)]
    return [series(lamp, "binary_state", lamp_s), series(eg, "target_state", eg_s, "value.on"),
            series(og, "target_state", og_s, "value.on")]


def frame(times, **extra):
    """A frame as provide_historic_data returns it: naive UTC times."""
    index = pd.to_datetime(np.asarray(times, dtype=float), unit="s")
    f = pd.DataFrame({"time": index})
    for k, v in extra.items():
        f[k] = v
    return f


class FakeDataset:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame


def datasets_of(raw):
    """Datasets and topic meta, one topic per signal, as the platform hands them over."""
    sets, meta = [], []
    for s, times, values in raw:
        sets.append(FakeDataset(frame(times, **{s.port: values})))
        meta.append({"name": s.service_id.replace(":", "_"), "filterType": "DeviceId", "filterValue": s.device_id,
                     "mappings": [{"dest": s.port, "source": s.path}]})
    return sets, meta
