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
import unittest

from algo import history
from tests import synthetic as syn


class FlakyDataset:
    """Fails `failures` times, then gives its frame."""

    def __init__(self, device, failures=0):
        self.device = device
        self.failures = failures
        self.calls = 0

    def to_pandas(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("could not translate host name")
        return syn.frame([1000.0, 1010.0], binary_state=[True, False])


def meta_of(devices):
    return [{"name": "urn_infai_ses_service_x", "filterType": "DeviceId", "filterValue": d,
             "mappings": [{"dest": "binary_state", "source": "p"}]} for d in devices]


class Read(unittest.TestCase):
    """The lib retries a table itself; a dataset that still fails sets history_failed once, the rest is kept in order."""

    def test_a_failing_dataset_sets_history_failed_and_the_rest_is_kept(self):
        bad, good = FlakyDataset("a", failures=1), FlakyDataset("b")
        notes = []
        out, failed = history.signals_from_datasets([bad, good], meta_of(["a", "b"]), notes)
        self.assertTrue(failed)
        self.assertEqual(bad.calls, 1)
        self.assertEqual([s.device_id for s, _, _ in out], ["b"])
        self.assertIn("topic 0 not readable", notes[0])

    def test_order_follows_topic_meta(self):
        devices = [f"d{i}" for i in range(12)]
        out, failed = history.signals_from_datasets([FlakyDataset(d) for d in devices], meta_of(devices), [])
        self.assertFalse(failed)
        self.assertEqual([s.device_id for s, _, _ in out], devices)


if __name__ == "__main__":
    unittest.main()
