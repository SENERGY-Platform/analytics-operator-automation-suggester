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

import datetime
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests import synthetic as syn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Replay(unittest.TestCase):
    def test_offline_replay_prints_the_evening_suggestion(self):
        raw = syn.evening()
        mapping, elements = [], []
        for signal, times, values in raw:
            mapping.append({"device_id": signal.device_id, "service_id": signal.service_id, "path": signal.path,
                            "port": signal.port})
            elements.append({"deviceId": signal.device_id, "serviceId": signal.service_id,
                             "data": [[[iso(t), v] for t, v in zip(times, values)]]})
        with tempfile.TemporaryDirectory() as d:
            m, r1, r2 = (os.path.join(d, n) for n in ("mapping.json", "r1.json", "r2.json"))
            for path, body in ((m, mapping), (r1, elements[:1]), (r2, elements[1:])):
                with open(path, "w") as f:
                    json.dump(body, f)
            out = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "replay_history.py"), m, r1, r2,
                                  "--json"], capture_output=True, text=True, cwd=d, check=True)
        found = json.loads(out.stdout)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]["window"]["from"], found[0]["window"]["to"]), ("19:00", "02:00"))
        self.assertEqual(len(found[0]["targets"]), 2)


def load_script():
    spec = importlib.util.spec_from_file_location("replay_history", os.path.join(ROOT, "scripts", "replay_history.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PowerReplay(unittest.TestCase):
    def test_buckets_take_the_maximum_and_changes_only_keeps_the_edges(self):
        replay = load_script()
        times, maxima = replay.bucket_max([0, 10, 59, 60, 119, 130], [1, 5, 2, "x", 7, 3], 60)
        self.assertEqual((times.tolist(), maxima.tolist()), ([0.0, 60.0, 120.0], [5.0, 7.0, 3.0]))
        t, v = replay.changes_only(np.arange(7.0), np.array([1, 1, 1, 2, 2, 2, 2.0]))
        self.assertEqual((t.tolist(), v.tolist()), ([0.0, 2.0, 3.0, 6.0], [1.0, 1.0, 2.0, 2.0]))

    def test_offline_replay_reads_power_as_the_operator_does(self):
        times, values = syn.tv_power(days=30)
        data = []
        for t, v in zip(times, values):
            # two readings per bucket; the bucket holds the larger one
            data += [[iso(t + 3), v * 0.5], [iso(t + 31), v]]
        element = {"deviceId": "dev-tv", "serviceId": syn.SVC, "data": [data]}
        mapping = [{"device_id": "dev-tv", "service_id": syn.SVC, "path": syn.POWER_PATH, "port": "power"}]
        with tempfile.TemporaryDirectory() as d:
            m, r = os.path.join(d, "mapping.json"), os.path.join(d, "r.json")
            for path, body in ((m, mapping), (r, [element])):
                with open(path, "w") as f:
                    json.dump(body, f)
            out = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "replay_history.py"), m, r, "--json"],
                                 capture_output=True, text=True, cwd=d, check=True)
            short = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "replay_history.py"), m, r,
                                    "--power-history-days", "10", "--json"], capture_output=True, text=True, cwd=d, check=True)
        self.assertEqual(json.loads(out.stdout), [])
        self.assertIn("suitable, on above 10 W, off below 5 W", out.stderr)
        self.assertIn(", 30 switch-ons", out.stderr)
        self.assertIn(", 10 switch-ons", short.stderr)


if __name__ == "__main__":
    unittest.main()
