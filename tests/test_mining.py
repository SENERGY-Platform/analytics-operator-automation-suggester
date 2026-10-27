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

from algo import events, window
from algo.events import Signal, debounce, extract
from algo.mining import Params, mine
from algo.training import extract_all
from tests import synthetic as syn


class Debounce(unittest.TestCase):
    def test_burst_takes_the_majority(self):
        t, v = debounce([0, 0.4, 0.8, 1.2, 1.6], ["on", "off", "on", "off", "off"])
        self.assertEqual((t, v), ([0], ["off"]))

    def test_tie_takes_the_later_value(self):
        self.assertEqual(debounce([0, 1], ["on", "off"]), ([0], ["off"]))

    def test_chained_burst_and_gap(self):
        t, v = debounce([0, 1.5, 3.0, 10, 11], ["on", "on", "on", "off", "off"])
        self.assertEqual((t, v), ([0, 10], ["on", "off"]))

    def test_burst_in_a_series_is_one_change(self):
        base = 1000.0
        times = [base, base + 100, base + 100.4, base + 100.8, base + 101.2, base + 101.6, base + 400]
        values = [True, True, False, True, False, False, False]
        se = extract(syn.sig("d", "binary_state"), times, values)
        self.assertEqual(se.events, [(base + 100, "off")])

    def test_text_and_number_spellings(self):
        se = extract(syn.sig("d", "binary_state"), [0, 10, 20, 30], ["ON", "off", 1, "x"])
        self.assertEqual(se.events, [(10, "off"), (20, "on")])


class Events(unittest.TestCase):
    def test_contact(self):
        se = extract(syn.sig("d", "contact_state"), [0, 10, 20], ["closed", "open", True])
        self.assertEqual(se.events, [(10, "open"), (20, "closed")])

    def test_button_repeats_within_two_seconds_are_one_press(self):
        se = extract(syn.sig("d", "button_press"), [0, 1, 2, 3.5, 100], ["single", "single", "single", "single", "double"])
        self.assertEqual([tok for _, tok in se.events], ["single", "double"])

    def test_position_direction_and_same_move(self):
        se = extract(syn.sig("d", "position"), [0, 10, 20, 30, 400, 410], [1.0, 0.8, 0.5, 0.5, 0.2, 0.9])
        self.assertEqual(se.events, [(10, "down"), (400, "down"), (410, "up")])

    def test_numeric_target_levels_and_settling(self):
        # 0..90 percent; a blip to 50 that lasts 20 s is no change, a move to 90 that stays is.
        times = [0, 100, 120, 200, 1000, 1100]
        values = [0, 50, 0, 20, 90, 90]
        se = extract(syn.sig("d", "target_position"), times, values)
        self.assertEqual(se.kind, "level")
        self.assertEqual([tok for _, tok in se.events], ["L2", "L9"])
        self.assertEqual(se.level_values["L9"], 0.9)

    def test_target_state_takes_only_binary_values(self):
        values = ["ON", "unavailable", "off", 1, "Off", "x", "on"]
        se = extract(syn.sig("d", "target_state"), [0, 10, 20, 30, 40, 50, 60], values)
        self.assertEqual(se.kind, "binary")
        self.assertEqual(se.events, [(20, "off"), (30, "on"), (40, "off"), (60, "on")])

    def test_numbers_and_text_on_target_state_are_no_events(self):
        for values in ([10, 50, 90], ["heat", "cool", "heat"]):
            se = extract(syn.sig("d", "target_state"), [0, 100, 200], values)
            self.assertEqual((se.kind, se.events), ("binary", []))


class Window(unittest.TestCase):
    def test_circular_run(self):
        self.assertEqual(window.hour_window([23, 23, 0, 0, 1, 1, 1, 1, 12]), (23, 3))

    def test_weekdays(self):
        self.assertEqual(window.weekdays_for([0, 1, 2, 3, 4, 4, 4, 4, 4, 2]), [0, 1, 2, 3, 4])
        self.assertEqual(window.weekdays_for([0, 1, 5, 6]), [0, 1, 2, 3, 4, 5, 6])


class Mining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals = extract_all(syn.evening())
        cls.found = mine(cls.signals, Params())

    def test_evening_window_and_two_targets(self):
        self.assertEqual(len(self.found), 1)
        s = self.found[0]
        self.assertEqual(s["trigger"], {"device_id": syn.LAMP, "service_id": syn.SVC, "path": "value.state",
                                        "kind": "binary", "from": "on", "to": "off"})
        self.assertEqual((s["window"]["from"], s["window"]["to"]), ("19:00", "02:00"))
        self.assertEqual(s["window"]["weekdays"], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(s["window"]["tz"], "Europe/Berlin")
        self.assertEqual([(t["device_id"], t["value"]) for t in s["targets"]], [(syn.EG, True), (syn.OG, True)])
        self.assertTrue(all(t["function_id"] is None for t in s["targets"]))
        stats = s["stats"]
        self.assertEqual((stats["occurrences"], stats["hits"]), (180, 120))
        self.assertAlmostEqual(stats["confidence"], 120 / 180, places=3)
        self.assertGreater(stats["lift"], 3)
        self.assertAlmostEqual(stats["median_delay_s"], 30, delta=1)
        self.assertIsNone(s["auto_off_s"])  # the lights stay on for half an hour

    def test_auto_off_for_short_cycles(self):
        found = mine(extract_all(syn.evening(light_cycle_s=120.0)), Params())
        self.assertEqual(found[0]["auto_off_s"], 120)

    def test_stats_do_not_change_the_id(self):
        again = mine(self.signals, Params(min_confidence=0.1))
        self.assertEqual([s["id"] for s in again], [s["id"] for s in self.found])

    def test_targets_config_limits_the_controllable_devices(self):
        found = mine(self.signals, Params(targets=(syn.EG,)))
        self.assertEqual([t["device_id"] for t in found[0]["targets"]], [syn.EG])
        self.assertEqual(mine(self.signals, Params(targets=("dev-other",))), [])

    def test_same_device_is_no_pair(self):
        lamp = syn.evening()[0]
        as_target = (syn.sig(syn.LAMP, "target_state"), lamp[1], lamp[2])
        self.assertEqual(mine(extract_all([lamp, as_target]), Params()), [])


def coupled(delay_s, share_every=10, light_device=syn.STRIP, follows=None):
    """The lamp goes off at all hours and the strip follows after delay_s in 9 of 10 cases."""
    lamp, strip = [], []
    for i in range(syn.DAYS):
        hour = 7 + (i * 5) % 16
        off = syn.local(i, hour, (i * 11) % 60)
        lamp += [(off - 600, True), (off, False)]
        if (follows(i) if follows else i % share_every != 0):
            strip += [(off + delay_s, True), (off + delay_s + 900, False)]
    return [syn.series(syn.LAMP, "binary_state", lamp), syn.series(light_device, "target_state", strip, "value.on")]


class Coupling(unittest.TestCase):
    def test_zero_second_coupling_is_dropped(self):
        signals = extract_all(coupled(0.5))
        self.assertEqual(mine(signals, Params()), [])
        # Without the coupling rule the same pair is a perfectly good suggestion: the rule is what drops it.
        found = mine(signals, Params(coupling_max_delay_s=0.0))
        self.assertGreaterEqual(len(found), 1)
        self.assertGreaterEqual(found[0]["stats"]["confidence"], 0.8)

    def test_delayed_follower_is_kept(self):
        found = mine(extract_all(coupled(40.0)), Params())
        self.assertGreaterEqual(len(found), 1)
        for s in found:
            self.assertAlmostEqual(s["stats"]["median_delay_s"], 40, delta=1)


def routine():
    """The lamp goes off at 19:30 every day and the light comes on at 19:33: a routine, no coupling."""
    lamp, light = [], []
    for i in range(syn.DAYS):
        off = syn.local(i, 19, 30, i % 3)
        lamp += [(off - 600, True), (off, False)]
        light += [(off + 180 + i % 5, True), (off + 1800, False)]
    return [syn.series(syn.LAMP, "binary_state", lamp), syn.series(syn.EG, "target_state", light, "value.on")]


class Routine(unittest.TestCase):
    def test_daily_routine_stays_below_lift_three(self):
        signals = extract_all(routine())
        self.assertEqual(mine(signals, Params()), [])
        found = mine(signals, Params(min_lift=0, min_confidence=0))
        self.assertEqual(len(found), 1)
        self.assertLess(found[0]["stats"]["lift"], 3)
        self.assertGreater(found[0]["stats"]["confidence"], 0.9)


def mornings():
    """Lamp off between 06 and 08 on workdays only, the light comes on a minute later."""
    lamp, light = [], []
    for i in range(syn.DAYS):
        if (syn.START.weekday() + i) % 7 >= 5:
            continue
        off = syn.local(i, 6 + i % 2, (i * 13) % 60)
        lamp += [(off - 600, True), (off, False)]
        light += [(off + 60, True), (off + 1800, False)]
    return [syn.series(syn.LAMP, "binary_state", lamp), syn.series(syn.EG, "target_state", light, "value.on")]


class Weekdays(unittest.TestCase):
    def test_workdays_only(self):
        found = mine(extract_all(mornings()), Params())
        self.assertEqual(found[0]["window"]["weekdays"], [0, 1, 2, 3, 4])
        self.assertEqual(found[0]["window"]["from"], "06:00")


def group_of(n_devices, delay_s=20.0):
    """n devices that are switched together every day, each reporting the state as target and as trigger."""
    raw = []
    for k in range(n_devices):
        pairs = []
        for i in range(syn.DAYS):
            t = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60) + k * delay_s
            pairs += [(t, True), (t + 1800, False)]
        raw += [syn.series(f"dev-group-{k}", "binary_state", pairs), syn.series(f"dev-group-{k}", "target_state", pairs)]
    return raw


class GroupAction(unittest.TestCase):
    """R1: a switching that two other devices repeat within a minute is a group control, no target and no trigger."""

    def test_two_devices_are_a_pair(self):
        self.assertGreaterEqual(len(mine(extract_all(group_of(2)), Params())), 1)

    def test_three_devices_are_a_group(self):
        self.assertEqual(mine(extract_all(group_of(3)), Params()), [])


def stair_light():
    """The light goes off 150 s after the lamp every day; it was on already on a quarter of the days."""
    lamp, light = [], []
    for i in range(syn.DAYS):
        off = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60)
        lamp += [(off - 7200, True), (off, False)]
        if i % 4 == 0:
            light += [(off - 3000, True)]
        else:
            light += [(off + 30, True)]
        light += [(off + 150, False)]
    return [syn.series(syn.LAMP, "binary_state", lamp), syn.series(syn.EG, "target_state", light, "value.on")]


class Intent(unittest.TestCase):
    """R2: a light that goes off again soon is suggested as "on" with the automatic off, not as "off"."""

    def test_on_with_auto_off_not_off(self):
        found = mine(extract_all(syn.evening(light_cycle_s=120.0)), Params())
        self.assertEqual(len(found), 1)
        self.assertEqual([t["value"] for t in found[0]["targets"]], [True, True])
        self.assertEqual(found[0]["auto_off_s"], 120)

    def test_off_has_more_hits_but_on_wins(self):
        found = mine(extract_all(stair_light()), Params())
        self.assertEqual(len(found), 1)
        self.assertEqual([t["value"] for t in found[0]["targets"]], [True])
        self.assertEqual(found[0]["auto_off_s"], 120)


def stair_light_before():
    """The light comes on 30 s before the lamp goes off and goes off 150 s after it: only the off follows."""
    lamp, light = [], []
    for i in range(syn.DAYS):
        off = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60)
        lamp += [(off - 7200, True), (off, False)]
        light += [(off - 30, True), (off + 150, False)]
    return [syn.series(syn.LAMP, "binary_state", lamp), syn.series(syn.EG, "target_state", light, "value.on")]


class ShortOnNeverOff(Intent):
    def test_off_alone_is_no_suggestion_when_on_does_not_qualify(self):
        self.assertEqual(mine(extract_all(stair_light_before()), Params()), [])

    def test_min_median_delay_lets_the_on_through(self):
        found = mine(extract_all(stair_light_before()), Params(min_median_delay_s=-60.0))
        self.assertEqual([t["value"] for s in found for t in s["targets"]], [True])


def with_follower(raw):
    """A strip that mirrors the lamp: as a trigger in 85 % of the cases, as a target five seconds later in 70 %."""
    lamp = raw[0]
    trig, tgt = [], []
    for n, (t, v) in enumerate(zip(lamp[1], lamp[2])):
        if n % 20 < 17:
            trig.append((t + 1.0, v))
        if n % 10 < 7:
            tgt.append((t + 5.0, v))
    own = (syn.sig(syn.LAMP, "target_state"), lamp[1], lamp[2])  # the lamp is controllable, too
    return raw + [own, syn.series(syn.STRIP, "binary_state", trig), syn.series(syn.STRIP, "target_state", tgt, "value.on")]


class TriggerCluster(unittest.TestCase):
    """R3: triggers that switch together are one source; its members are no targets of the representative."""

    def test_one_representative_and_no_member_as_target(self):
        found = mine(extract_all(with_follower(syn.evening())), Params())
        self.assertEqual([s["trigger"]["device_id"] for s in found], [syn.LAMP])
        self.assertEqual([t["device_id"] for t in found[0]["targets"]], [syn.EG, syn.OG])

    def test_without_the_cluster_the_follower_would_be_a_target(self):
        # Same data with the strip's trigger events moved to other moments: no cluster.
        raw = with_follower(syn.evening())
        moved = [(syn.sig(syn.STRIP, "binary_state"), [t + 40 * 3600 * (n % 3) for n, t in enumerate(r[1])], r[2])
                 if r[0].device_id == syn.STRIP and r[0].port == "binary_state" else r for r in raw]
        found = mine(extract_all(moved), Params())
        self.assertIn(syn.STRIP, [t["device_id"] for s in found for t in s["targets"]])


def episodes():
    """A motion sensor that reports on and off every 30 s for five minutes; the light comes on 20 s after the first."""
    pir, light = [], []
    for i in range(syn.DAYS):
        start = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60)
        for k in range(6):
            pir += [(start + 60 * k, True), (start + 60 * k + 30, False)]
        light += [(start + 20, True), (start + 20 + 1800, False)]
    return [syn.series("dev-pir", "binary_state", pir), syn.series(syn.EG, "target_state", light, "value.on")]


class Episodes(unittest.TestCase):
    """R4: repeated reports within window_after_s are one occurrence."""

    def test_occurrences_count_episodes(self):
        found = [s for s in mine(extract_all(episodes()), Params()) if s["trigger"]["to"] == "on"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["stats"]["occurrences"], syn.DAYS)
        # The first value of a series is a state, not an event: the first light of the first day is missing.
        self.assertEqual(found[0]["stats"]["hits"], syn.DAYS - 1)


class OneWindow(unittest.TestCase):
    """R5: the window comes from the hits of all targets of the trigger, one suggestion per trigger."""

    def test_two_targets_with_different_own_windows_are_one_suggestion(self):
        raw = syn.evening(og_hours={19, 20, 21, 22})  # the second light follows only in the early evening
        found = mine(extract_all(raw), Params())
        self.assertEqual(len(found), 1)
        self.assertEqual([t["device_id"] for t in found[0]["targets"]], [syn.EG, syn.OG])
        self.assertEqual((found[0]["window"]["from"], found[0]["window"]["to"]), ("19:00", "02:00"))


def several(count):
    raw = []
    for k in range(count):
        raw += syn.evening(hit_every=3 + k, lamp=f"dev-lamp-{k}", eg=f"dev-eg-{k}", og=f"dev-og-{k}", minute_offset=20 * k)
    return extract_all(raw)


class Cap(unittest.TestCase):
    """R6: at most max_suggestions, the best by lift times the root of the hits."""

    def test_cap_keeps_the_best(self):
        signals = several(3)
        everything = mine(signals, Params(max_suggestions=100))
        self.assertEqual(len(everything), 3)
        capped = mine(signals, Params(max_suggestions=2))
        self.assertEqual([s["id"] for s in capped], [s["id"] for s in everything[:2]])
        score = [s["stats"]["lift"] * s["stats"]["hits"] ** 0.5 for s in everything]
        self.assertEqual(score, sorted(score, reverse=True))

    def test_default_is_ten(self):
        self.assertEqual(Params().max_suggestions, 10)


def two_blocks():
    """Lamp off on workday mornings (06-08) and every evening (19-21); the light follows 30 s later in both."""
    lamp, light = [], []
    for i in range(syn.DAYS):
        weekday = (syn.START.weekday() + i) % 7
        slots = []
        if weekday < 5:
            slots.append(6 + i % 3)
        if i % 3 != 0:
            slots.append(19 + (i // 3) % 3)
        for hour in slots:
            off = syn.local(i, hour, (i * 7) % 60)
            lamp += [(off - 7200, True), (off, False)]
            light += [(off + 30, True), (off + 1830, False)]
    return [syn.series(syn.LAMP, "binary_state", lamp), syn.series(syn.EG, "target_state", light, "value.on")]


class HourBlocks(unittest.TestCase):
    """R5: each block of hours with hits is a window of its own, with its own weekdays."""

    def test_morning_and_evening_are_two_suggestions(self):
        found = sorted(mine(extract_all(two_blocks()), Params()), key=lambda s: s["window"]["from"])
        self.assertEqual([(s["window"]["from"], s["window"]["to"], s["window"]["weekdays"]) for s in found],
                         [("06:00", "09:00", [0, 1, 2, 3, 4]), ("19:00", "22:00", [0, 1, 2, 3, 4, 5, 6])])

    def test_hour_blocks(self):
        from algo.mining import hour_blocks
        self.assertEqual(hour_blocks([23] * 10 + [0] * 10 + [7] * 10 + [12]), [(7, 1), (23, 2)])
        self.assertEqual(hour_blocks([5] * 100 + [9] * 3), [(5, 1)])  # 3 of 103 is under 5 %
        self.assertEqual(hour_blocks([]), [])


def pressed():
    """A button is pressed in the evening, the light comes on 30 s later; the button's device has a (silent) state."""
    press, light = [], []
    for i in range(syn.DAYS):
        t = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60)
        press += [(t, "single")]
        light += [(t + 30, True), (t + 1830, False)]
    return [syn.series("dev-button", "button_press", press, "value.action"),
            syn.series(syn.EG, "target_state", light, "value.on")]


class Buttons(unittest.TestCase):
    """R7: the button of a device that has a state of its own is no trigger."""

    def test_button_of_a_plain_remote_is_a_trigger(self):
        found = mine(extract_all(pressed()), Params())
        self.assertEqual([s["trigger"]["device_id"] for s in found], ["dev-button"])
        self.assertEqual(found[0]["trigger"]["kind"], "button")

    def test_button_of_a_device_with_a_state_is_none(self):
        state = syn.series("dev-button", "binary_state", [(syn.local(0, 1), True), (syn.local(1, 1), True)])
        self.assertEqual(mine(extract_all(pressed() + [state]), Params()), [])


def mutual():
    """A on is followed by B on 30 s later every day; B off is followed by A off 30 s later on 70 % of the days."""
    a, b = [], []
    for i in range(syn.DAYS):
        t = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60)
        a += [(t, True)]
        b += [(t + 30, True), (t + 1830, False)]
        if i % 10 < 7:
            a += [(t + 1860, False)]
        else:
            a += [(t + 20000, False)]
    raw = []
    for dev, pairs in (("dev-a", a), ("dev-b", b)):
        raw += [syn.series(dev, "binary_state", pairs), syn.series(dev, "target_state", pairs)]
    return raw


class OneDirection(unittest.TestCase):
    """R8: of the two directions between two devices only the better ranked stays."""

    def test_only_the_stronger_direction(self):
        found = mine(extract_all(mutual()), Params())
        self.assertEqual([(s["trigger"]["device_id"], s["trigger"]["to"]) for s in found], [("dev-a", "on")])


class FastCoupling(unittest.TestCase):
    """R9: a delay of at most coupling_max_delay_s is a coupling, however often it happens."""

    def test_rare_but_instant_follower_is_dropped(self):
        signals = extract_all(coupled(1.5, follows=lambda i: i % 5 < 3))  # 60 % of the cases
        self.assertEqual(mine(signals, Params()), [])
        self.assertGreaterEqual(len(mine(signals, Params(coupling_max_delay_s=0.0))), 1)


class Consequence(unittest.TestCase):
    """R10: a target that switches before the trigger is no consequence of it."""

    def test_target_before_trigger_is_dropped(self):
        lamp, light = [], []
        for i in range(syn.DAYS):
            off = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60)
            lamp += [(off - 7200, True), (off, False)]
            light += [(off - 30, True), (off + 1770, False)]
        raw = [syn.series(syn.LAMP, "binary_state", lamp), syn.series(syn.EG, "target_state", light, "value.on")]
        self.assertEqual(mine(extract_all(raw), Params()), [])


def numeric_target(port, high, low=0.0, device="dev-blind"):
    """The lamp goes off in the evening; 30 s later the device moves to `high`, after half an hour back to `low`."""
    lamp, target = [], []
    for i in range(syn.DAYS):
        off = syn.local(i, syn.EVENING[i % len(syn.EVENING)], (i * 7) % 60)
        lamp += [(off - 7200, True), (off, False)]
        target += [(off + 30, high), (off + 1830, low)]
    return [syn.series(syn.LAMP, "binary_state", lamp), syn.series(device, port, target, "value.level")]


class NumericTargetPorts(unittest.TestCase):
    """target_position and target_speed are target ports; a number becomes a level, the action sets its median."""

    def test_target_position_in_percent_becomes_a_fraction(self):
        found = mine(extract_all(numeric_target("target_position", 80.0, 0.0)), Params())
        self.assertEqual(len(found), 1)
        self.assertEqual([(t["device_id"], t["path"], t["value"]) for t in found[0]["targets"]],
                         [("dev-blind", "value.level", 0.8)])

    def test_target_speed_keeps_its_scale(self):
        found = mine(extract_all(numeric_target("target_speed", 50.0, 0.0, "dev-fan")), Params())
        self.assertEqual(len(found), 1)
        self.assertEqual([(t["device_id"], t["value"]) for t in found[0]["targets"]], [("dev-fan", 50.0)])

    def test_targets_filter_applies_to_them(self):
        raw = extract_all(numeric_target("target_speed", 50.0, 0.0, "dev-fan"))
        self.assertEqual(mine(raw, Params(targets=("dev-other",))), [])
        self.assertEqual(len(mine(raw, Params(targets=("dev-fan",)))), 1)

    def test_a_group_of_them_is_a_group_action(self):
        raw = []
        for k in range(3):
            raw += numeric_target("target_position", 80.0, 0.0, f"dev-blind-{k}")[1:]
        raw.append(numeric_target("target_position", 80.0)[0])
        self.assertEqual(mine(extract_all(raw), Params()), [])


class Contract(unittest.TestCase):
    def test_suggestion_is_json_serialisable_with_the_contract_keys(self):
        s = json.loads(json.dumps(mine(extract_all(syn.evening()), Params())[0]))
        self.assertEqual(set(s), {"id", "trigger", "window", "targets", "auto_off_s", "stats"})
        self.assertEqual(set(s["trigger"]), {"device_id", "service_id", "path", "kind", "from", "to"})
        self.assertEqual(set(s["window"]), {"from", "to", "weekdays", "tz"})
        self.assertEqual(set(s["targets"][0]), {"device_id", "service_id", "path", "function_id", "value"})
        self.assertEqual(set(s["stats"]), {"occurrences", "hits", "confidence", "baseline", "lift", "median_delay_s"})
        self.assertEqual(len(s["id"]), 16)


if __name__ == "__main__":
    unittest.main()
