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
import threading
import time
import unittest
from unittest import mock

from algo import history
from tests import synthetic as syn


class FlakyDataset:
    """Fails `failures` times, then gives its frame; sleeps `delay` seconds while reading."""

    def __init__(self, device, failures=0, delay=0.0, tracker=None):
        self.device = device
        self.failures = failures
        self.delay = delay
        self.calls = 0
        self.tracker = tracker

    def to_pandas(self):
        self.calls += 1
        if self.tracker is not None:
            self.tracker.enter()
        try:
            time.sleep(self.delay)
            if self.calls <= self.failures:
                raise RuntimeError("could not translate host name")
            return syn.frame([1000.0, 1010.0], binary_state=[True, False])
        finally:
            if self.tracker is not None:
                self.tracker.leave()


class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)

    def leave(self):
        with self.lock:
            self.running -= 1


def meta_of(devices):
    return [{"name": "urn_infai_ses_service_x", "filterType": "DeviceId", "filterValue": d,
             "mappings": [{"dest": "binary_state", "source": "p"}]} for d in devices]


class Retry(unittest.TestCase):
    """N1: a dataset is read up to three times, 15 s apart, before it counts as failed."""

    def test_two_failures_then_success(self):
        ds = FlakyDataset("a", failures=2)
        notes = []
        with mock.patch("algo.history.time.sleep") as sleep:
            out, failed = history.signals_from_datasets([ds], meta_of(["a"]), notes)
        self.assertFalse(failed)
        self.assertEqual((ds.calls, len(out), notes), (3, 1, []))
        self.assertEqual([c.args[0] for c in sleep.call_args_list if c.args[0] == 15.0], [15.0, 15.0])

    def test_three_failures_set_history_failed_and_the_rest_is_kept(self):
        bad, good = FlakyDataset("a", failures=3), FlakyDataset("b")
        notes = []
        with mock.patch("algo.history.time.sleep"):
            out, failed = history.signals_from_datasets([bad, good], meta_of(["a", "b"]), notes)
        self.assertTrue(failed)
        self.assertEqual(bad.calls, 3)
        self.assertEqual([s.device_id for s, _, _ in out], ["b"])
        self.assertIn("topic 0 not readable", notes[0])


class Parallel(unittest.TestCase):
    """N2: datasets are read concurrently, at most eight at once, and the results keep the order of the topics."""

    def test_order_follows_topic_meta_whatever_finishes_first(self):
        devices = [f"d{i}" for i in range(12)]
        # The first datasets are the slowest, so they finish last.
        sets = [FlakyDataset(d, delay=0.002 * (len(devices) - i)) for i, d in enumerate(devices)]
        out, failed = history.signals_from_datasets(sets, meta_of(devices), [])
        self.assertFalse(failed)
        self.assertEqual([s.device_id for s, _, _ in out], devices)

    def test_at_most_eight_at_once_and_really_parallel(self):
        tracker = Tracker()
        sets = [FlakyDataset(f"d{i}", delay=0.05, tracker=tracker) for i in range(20)]
        history.signals_from_datasets(sets, meta_of([f"d{i}" for i in range(20)]), [])
        self.assertGreater(tracker.peak, 1)
        self.assertLessEqual(tracker.peak, 8)


if __name__ == "__main__":
    unittest.main()
