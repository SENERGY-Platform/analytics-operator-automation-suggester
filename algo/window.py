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
The time of day a rule holds: the smallest run of whole local hours that carries most hits, and the weekdays.

Hours are the hours of the trigger's local time. The weekday is that of the trigger's local time as well, so for a
window across midnight the occurrences after midnight count to the new day.
"""

__all__ = ("HIT_SHARE", "WEEKDAY_SHARE", "hour_window", "in_hours", "weekdays_for", "format_hour")

import typing

import numpy as np

HIT_SHARE = 0.8
WEEKDAY_SHARE = 0.9
ALL_DAYS = [0, 1, 2, 3, 4, 5, 6]
WORKDAYS = [0, 1, 2, 3, 4]


def hour_window(hit_hours: typing.Sequence[int], share: float = HIT_SHARE) -> typing.Tuple[int, int]:
    """
    (start, length) of the smallest circular run of hours, 1 to 24 long, holding at least `share` of the hits. Among
    runs of that length the one with most hits wins, then the earliest start. Without hits the whole day.
    """
    counts = np.bincount(np.asarray(hit_hours, dtype=int) % 24, minlength=24)
    total = int(counts.sum())
    if total == 0:
        return 0, 24
    need = share * total
    doubled = np.concatenate([counts, counts])
    prefix = np.concatenate([[0], np.cumsum(doubled)])
    for length in range(1, 25):
        sums = [int(prefix[s + length] - prefix[s]) for s in range(24)]
        best = max(sums)
        if best >= need - 1e-9:
            return sums.index(best), length
    return 0, 24


def in_hours(hour: int, start: int, length: int) -> bool:
    return length >= 24 or (hour - start) % 24 < length


def format_hour(hour: int) -> str:
    return f"{hour % 24:02d}:00"


def weekdays_for(hit_weekdays: typing.Sequence[int], share: float = WEEKDAY_SHARE) -> typing.List[int]:
    """Monday to Friday when at least `share` of the hits fall on those days, else every day."""
    if len(hit_weekdays) == 0:
        return list(ALL_DAYS)
    work = sum(1 for d in hit_weekdays if d < 5)
    return list(WORKDAYS) if work >= share * len(hit_weekdays) - 1e-9 else list(ALL_DAYS)
