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


POWER_PATH = "value.sensor.ENERGY.Power"
MINUTE = 60.0


def as_power(raw, on_w=60.0, off_w=1.0, delay_s=0.0, device=None):
    """A binary series as the power of a plug: on_w while it is on, off_w while it is off, delay_s later."""
    s, times, values = raw
    return (Signal(device or s.device_id, s.service_id, POWER_PATH, "power"), [t + delay_s for t in times],
            [on_w if v else off_w for v in values])


def thin(times, values):
    """What `changes_only` leaves of a series: the rows that differ from a neighbour, and the first and the last."""
    n = len(values)
    keep = [i for i in range(n) if i in (0, n - 1) or values[i] != values[i - 1] or values[i] != values[i + 1]]
    return [times[i] for i in keep], [values[i] for i in keep]


def tv_power(days=60, seed=7, wake_buckets=0, dips=2):
    """
    A TV plug read in minute buckets (the maximum per bucket): 1 W in standby; one evening session a day, 2 to 4 hours
    long from 18:30 to 21:30, at 80 to 160 W with `dips` single minutes of standby in it; and, if wake_buckets,
    a spike of 105 W of that many buckets once a day between 01:00 and 16:00.
    """
    rs = np.random.RandomState(seed)
    base = local(0, 0)
    values = np.full(days * 1440 + 300, 1.0)  # the last session may run past midnight
    for d in range(days):
        start = d * 1440 + 18 * 60 + 30 + int(rs.randint(0, 180))
        length = int(rs.randint(120, 240))
        values[start:start + length] = rs.uniform(80, 160, length)
        for j in range(dips):  # apart, so that no two make a gap of two minutes
            values[start + 5 + 20 * j + int(rs.randint(0, 10))] = 1.0
        minute = int(rs.randint(60, 16 * 60))  # at no fixed time, or it would look like a timer
        values[d * 1440 + minute:d * 1440 + minute + wake_buckets] = 105.0
    times = [base + i * MINUTE for i in range(len(values))]
    return times, values.tolist()
