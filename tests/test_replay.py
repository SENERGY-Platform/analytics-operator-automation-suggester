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
import json
import os
import subprocess
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
