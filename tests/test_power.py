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

import json
import unittest

import numpy as np

from algo import power
from algo.events import extract
from algo.mining import Params, mine, rule_key
from algo.power import PowerSpec
from algo.training import extract_all
from tests import synthetic as syn
from tests.test_mining import coupled


def kinds(analysis, kind):
    return [t for t, k in analysis.events if k == kind]


class Rounding(unittest.TestCase):
    def test_nearest_number_of_the_125_series_on_the_log_scale(self):
        cases = {1.0: 1, 1.4: 1, 1.5: 2, 3.1: 2, 3.2: 5, 7.0: 5, 7.1: 10, 8.8: 10, 14.0: 10, 15.0: 20, 31.0: 20, 32.0: 50,
                 70.0: 50, 72.0: 100, 0.3: 0.2, 0.0072: 0.01}
        for value, expected in cases.items():
            self.assertEqual(power.round_125(value), expected, value)

    def test_rejects_what_has_no_log(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                power.round_125(bad)


class States(unittest.TestCase):
    def test_between_the_thresholds_the_previous_state_stays(self):
        t = np.arange(8) * 60.0
        p = np.array([1, 1, 100, 100, 7, 7, 100, 1.0])
        self.assertEqual(power._episodes(t, p, 10.0, 5.0), [[0, 0.0, 120.0], [1, 120.0, 420.0], [0, 420.0, 420.0]])

    def test_a_series_that_never_leaves_the_band_has_no_state(self):
        self.assertEqual(power._episodes(np.arange(3) * 60.0, np.array([6.0, 7.0, 8.0]), 10.0, 5.0), [])

    def test_a_short_on_does_not_stand_and_its_neighbours_join(self):
        # off 0-600, on 600-660 (one bucket), off 660-1260, on 1260 on
        eps = [[0, 0, 600], [1, 600, 660], [0, 660, 1260], [1, 1260, 5000]]
        self.assertEqual(power._confirm(eps, 60.0), [[0, 0, 1260], [1, 1260, 5000]])

    def test_an_on_stands_only_beyond_the_hold_and_a_bucket(self):
        for length, stands in ((120, False), (121, True)):
            eps = [[0, 0, 600], [1, 600, 600 + length], [0, 600 + length, 5000], [1, 5000, 6000]]
            self.assertEqual(len(power._confirm(eps, 60.0)), 4 if stands else 2, length)
        # raw readings have no bucket: the hold alone counts
        eps = [[0, 0, 600], [1, 600, 661], [0, 661, 5000], [1, 5000, 6000]]
        self.assertEqual(len(power._confirm(eps, 0.0)), 4)

    def test_an_off_gap_of_120_s_or_less_is_no_switching_off(self):
        # The same bound as for "on": HOLD_S plus a bucket, the hold_s the process applies.
        for gap, stands in ((120, False), (121, True)):
            eps = [[0, 0, 600], [1, 600, 3600], [0, 3600, 3600 + gap], [1, 3600 + gap, 9000]]
            confirmed = power._confirm(eps, 60.0)
            self.assertEqual(len(confirmed), 4 if stands else 2, gap)

    def test_first_and_last_episode_stay(self):
        eps = [[1, 0, 30], [0, 30, 4000], [1, 4000, 4030]]
        self.assertEqual(power._confirm(eps, 60.0), eps)


class TvPattern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times, cls.values = syn.tv_power()
        cls.result = power.analyze(cls.times, cls.values)

    def test_thresholds_and_one_switching_per_session(self):
        r = self.result
        self.assertTrue(r.suitable, r.reason)
        self.assertEqual(r.spec, PowerSpec(10, 5, 120))
        # a dip to standby of one minute inside a session is no switching off
        self.assertEqual((len(kinds(r, "on")), len(kinds(r, "off"))), (60, 60))
        self.assertAlmostEqual(r.on_per_day, 1.0, delta=0.05)
        self.assertLess(r.low_hi, 5)
        self.assertGreater(r.high_lo, 10)

    def test_an_event_is_the_start_of_the_changed_bucket(self):
        first_on = kinds(self.result, "on")[0]
        self.assertGreater(self.values[self.times.index(first_on)], 10)
        self.assertEqual(self.values[self.times.index(first_on) - 1], 1.0)

    def test_what_changes_only_leaves_gives_the_same_events(self):
        thinned = power.analyze(*syn.thin(self.times, self.values))
        self.assertLess(len(syn.thin(self.times, self.values)[0]), len(self.times) // 4)
        self.assertEqual(thinned.events, self.result.events)
        self.assertEqual(thinned.spec, self.result.spec)

    def test_a_spike_of_two_buckets_is_no_switching_on_but_one_of_three_is(self):
        for wake, expected in ((1, 60), (2, 60), (3, 60 + 60)):
            times, values = syn.tv_power(wake_buckets=wake)
            r = power.analyze(times, values)
            self.assertEqual(len(kinds(r, "on")), expected, wake)

    def test_extract_gives_events_and_the_thresholds(self):
        se = extract(syn.sig("tv", "power", syn.POWER_PATH), self.times, self.values)
        self.assertEqual((se.kind, se.power, len(se.events)), ("power", PowerSpec(10, 5, 120), 120))


class Unsuitable(unittest.TestCase):
    def test_base_load_without_an_off_state(self):
        rs = np.random.RandomState(3)
        times = [syn.local(0, 0) + i * 60.0 for i in range(60 * 1440)]
        r = power.analyze(times, rs.uniform(14, 24, len(times)).tolist())
        self.assertFalse(r.suitable)
        self.assertIn("low level reaches", r.reason)
        self.assertEqual(extract(syn.sig("pc", "power"), times, rs.uniform(14, 24, len(times)).tolist()).events, [])

    def test_a_device_that_hardly_ever_draws_power(self):
        values = [0.0] * (60 * 1440)
        for d in range(0, 60, 3):
            values[d * 1440 + 600] = 1500.0
        times = [syn.local(0, 0) + i * 60.0 for i in range(len(values))]
        r = power.analyze(times, values)
        self.assertFalse(r.suitable)
        self.assertIn("share", r.reason)
        self.assertLess(r.on_share, power.MIN_ON_SHARE)

    def test_one_level_only(self):
        r = power.analyze([0.0, 60.0, 120.0], [5.0, 5.0, 5.0])
        self.assertEqual((r.suitable, r.reason), (False, "one level only"))

    def test_no_numbers(self):
        self.assertFalse(power.analyze([0.0, 1.0, 2.0], ["a", None, True]).suitable)
        self.assertFalse(power.analyze([], []).suitable)

    def test_negative_readings_count_as_zero(self):
        times, values = syn.tv_power(days=30)
        shifted = [v - 1.0 if v == 1.0 else v for v in values]  # standby 0 W
        shifted[5] = -3.0
        self.assertTrue(power.analyze(times, shifted).suitable)

    @staticmethod
    def thermostat(period_min, on_min, jitter_min=0, days=30, seed=5):
        rs = np.random.RandomState(seed)
        values = np.full(days * 1440, 0.0)
        for start in range(0, days * 1440 - on_min - jitter_min, period_min):
            s = start + (int(rs.randint(0, jitter_min + 1)) if jitter_min else 0)
            values[s:s + on_min] = 1000.0
        return [syn.local(0, 0) + i * 60.0 for i in range(len(values))], values.tolist()

    def test_a_thermostat_switching_on_more_than_12_times_a_day_is_excluded(self):
        r = power.analyze(*self.thermostat(60, 15))
        self.assertFalse(r.suitable)
        self.assertEqual(r.reason, "switches on in a regular cycle")
        self.assertGreater(r.on_per_day, 12)

    def test_a_regular_cycle_of_four_a_day_is_excluded(self):
        r = power.analyze(*self.thermostat(360, 20, jitter_min=10))
        self.assertFalse(r.suitable)
        self.assertEqual(r.reason, "switches on in a regular cycle")
        self.assertAlmostEqual(r.on_per_day, 4, delta=0.2)

    def test_irregular_switching_on_four_times_a_day_is_not_a_cycle(self):
        rs = np.random.RandomState(11)
        values = np.zeros(30 * 1440)
        for d in range(30):
            starts = []
            while len(starts) < 4:
                s = int(rs.randint(0, 1440 - 30))
                if all(abs(s - o) > 60 for o in starts):
                    starts.append(s)
            for s in starts:
                values[d * 1440 + s:d * 1440 + s + 20] = 1000.0
        r = power.analyze([syn.local(0, 0) + i * 60.0 for i in range(len(values))], values.tolist())
        self.assertTrue(r.suitable, r.reason)
        self.assertGreaterEqual(r.on_per_day, 3)


class Mining(unittest.TestCase):
    @staticmethod
    def evening_power(delay_s=0.0, keep_binary=False, follow_s=90.0):
        """
        The evening example with the lamp's power instead of, or beside, its binary state, `delay_s` after the
        switching. The lights follow follow_s later than in syn.evening(): a power trigger is no trigger for a
        follower within a bucket.
        """
        lamp, eg, og = syn.evening()
        eg, og = ((s_, [t + follow_s for t in ts], vs) for s_, ts, vs in (eg, og))
        raw = [syn.as_power(lamp, delay_s=delay_s), eg, og]
        return ([lamp] + raw) if keep_binary else raw

    def test_the_evening_pattern_from_power_alone(self):
        found = mine(extract_all(self.evening_power()), Params())
        self.assertEqual(len(found), 1)
        s = found[0]
        self.assertEqual(s["trigger"], {"device_id": syn.LAMP, "service_id": syn.SVC, "path": syn.POWER_PATH,
                                        "kind": "power", "from": "on", "to": "off", "on_above_w": 10, "off_below_w": 5,
                                        "hold_s": 120})
        self.assertEqual((s["window"]["from"], s["window"]["to"]), ("19:00", "02:00"))
        self.assertEqual(len(s["targets"]), 2)

    def test_the_trigger_carries_its_numbers_through_json(self):
        s = json.loads(json.dumps(mine(extract_all(self.evening_power()), Params())[0]))
        self.assertEqual(list(s["trigger"]), ["device_id", "service_id", "path", "kind", "from", "to", "on_above_w",
                                              "off_below_w", "hold_s"])
        self.assertIsInstance(s["trigger"]["on_above_w"], int)

    def test_thresholds_are_part_of_the_rule(self):
        trigger = {"device_id": "d", "service_id": "s", "path": "p", "kind": "power", "from": "on", "to": "off",
                   "on_above_w": 10, "off_below_w": 5, "hold_s": 60}
        win = {"from": "19:00", "to": "02:00", "weekdays": [0, 1]}
        tokens = [("t", "s", "p", "on")]
        key = rule_key(trigger, win, tokens)
        self.assertNotEqual(key, rule_key(dict(trigger, on_above_w=20, off_below_w=10), win, tokens))
        self.assertNotEqual(key, rule_key(dict(trigger, hold_s=30), win, tokens))
        binary = dict(trigger, kind="binary")
        self.assertEqual(set(json.loads(rule_key(binary, win, tokens))), {"t", "w", "g"})  # as it was for every other kind

    def test_a_power_change_at_the_moment_of_a_switching_is_dropped(self):
        # The plug reports the switching and, a moment away, the power: the binary trigger covers it.
        for delay in (20.0, -20.0, -100.0):
            found = mine(extract_all(self.evening_power(delay_s=delay, keep_binary=True)), Params())
            self.assertEqual([s["trigger"]["kind"] for s in found], ["binary"], delay)
        # Beyond the bucket plus a minute (120 s) it is a change of its own.
        found = mine(extract_all(self.evening_power(delay_s=-300.0, keep_binary=True)), Params())
        self.assertEqual(sorted(s["trigger"]["kind"] for s in found), ["binary", "power"])

    def test_without_the_binary_series_the_power_trigger_stays(self):
        found = mine(extract_all(self.evening_power(delay_s=-100.0)), Params())
        self.assertEqual([s["trigger"]["kind"] for s in found], ["power"])

    def test_the_exclusion_is_per_device(self):
        lamp, eg, og = self.evening_power(keep_binary=True)[:1] + self.evening_power()[1:]
        other = syn.as_power(lamp, device="dev-other-plug", delay_s=-20.0)
        found = mine(extract_all([lamp, eg, og, other]), Params())
        self.assertEqual(sorted(s["trigger"]["device_id"] for s in found), [syn.LAMP, "dev-other-plug"])

    def test_a_follower_within_the_bucket_is_a_coupling_for_a_power_trigger_only(self):
        lamp, strip = coupled(40.0)
        self.assertEqual(mine(extract_all([syn.as_power(lamp), strip]), Params()), [])
        found = mine(extract_all([lamp, strip]), Params())
        self.assertGreaterEqual(len(found), 1)  # 40 s is no coupling for a binary trigger (2 s)
        found = mine(extract_all([syn.as_power(lamp), strip]), Params(power_bucket_s=30.0))
        self.assertGreaterEqual(len(found), 1)
        self.assertEqual(found[0]["trigger"]["kind"], "power")

    def test_power_is_never_a_target(self):
        lamp, eg, og = syn.evening()
        found = mine(extract_all([lamp, syn.as_power(eg)]), Params())
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
