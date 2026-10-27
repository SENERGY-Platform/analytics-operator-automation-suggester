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
import unittest
from unittest import mock

import pandas as pd

import main
from operator_lib.util.helpers import SqlOverride
from main import CustomConfig, Operator, params_of
from algo.model import AutomationModel
from algo.training import fit_rules
from tests import synthetic as syn

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 7, 1, 1, 12, tzinfo=UTC)
META_KEYS = ("name", "filterType", "filterValue", "mappings")


class FakePyFuncModel:
    """What mlflow.pyfunc.load_model returns: the python model behind unwrap_python_model()."""

    def __init__(self, python_model):
        self.python_model = python_model

    def unwrap_python_model(self):
        return self.python_model


def make_op(**config):
    op = Operator()
    op.config = CustomConfig(dict(config))
    return op


def run_train(op, model, raw, now=NOW):
    sets, meta = syn.datasets_of(raw)
    remote = mock.Mock(side_effect=lambda *a: fit_rules(*a))
    with mock.patch("main.provide_historic_data", return_value=sets), mock.patch("main._topic_meta", return_value=meta), \
            mock.patch("main.train_rules", mock.Mock(remote=remote)), mock.patch("main.ray.get", side_effect=lambda r: r), \
            mock.patch.object(main.clock, "now", return_value=now):
        return op.train(model, None)


class Config(unittest.TestCase):
    def test_defaults_and_empty_strings(self):
        for raw in ({}, {n: "" for n in ("history_days", "min_hits", "min_lift", "targets")}):
            p = params_of(CustomConfig(raw))
            self.assertEqual((p.window_before_s, p.window_after_s, p.min_hits, p.min_lift, p.min_confidence),
                             (300, 600, 10, 3.0, 0.25))
            self.assertEqual(p.min_confidence_off, 0.5)
            self.assertEqual(p.max_suggestions, 10)
            self.assertEqual((p.coupling_max_delay_s, p.min_median_delay_s, p.targets),
                             (2.0, -10.0, ()))
            self.assertEqual(CustomConfig(raw).history_days, 365)

    def test_values_from_strings(self):
        c = CustomConfig({"min_hits": "7", "min_lift": "2.5", "history_days": "30", "targets": "a, b,a"})
        self.assertEqual((c.min_hits, c.min_lift, c.history_days, params_of(c).targets), (7, 2.5, 30, ("a", "b")))
        self.assertEqual(params_of(CustomConfig({"targets": '["x","y"]'})).targets, ("x", "y"))
        self.assertEqual(CustomConfig({"min_hits": "many"}).min_hits, 10)


class PowerInput(unittest.TestCase):
    def test_config_defaults_and_values(self):
        c = CustomConfig({})
        self.assertEqual((c.power_history_days, c.power_bucket_s, params_of(c).power_bucket_s), (90, 60, 60))
        c = CustomConfig({"power_history_days": "30", "power_bucket_s": "120"})
        self.assertEqual((c.power_history_days, params_of(c).power_bucket_s), (30, 120))

    def test_values_below_one_fall_back_to_the_defaults(self):
        with mock.patch("main.logger"):
            c = CustomConfig({"power_history_days": "0", "power_bucket_s": "-5"})
        self.assertEqual((c.power_history_days, c.power_bucket_s), (90, 60))

    @staticmethod
    def train_with(raw, config=None):
        op = make_op(**(config or {}))
        sets, meta = syn.datasets_of(raw)
        provide = mock.Mock(return_value=sets)
        remote = mock.Mock(side_effect=lambda *a: fit_rules(*a))
        with mock.patch("main.provide_historic_data", provide), mock.patch("main._topic_meta", return_value=meta), \
                mock.patch("main.train_rules", mock.Mock(remote=remote)), mock.patch("main.ray.get", side_effect=lambda r: r), \
                mock.patch.object(main.clock, "now", return_value=NOW):
            model = op.train(None, None)
        return model, provide

    def test_the_history_of_power_is_read_per_bucket_over_its_own_days(self):
        _, provide = self.train_with([syn.as_power(syn.evening()[0])])
        args, kwargs = provide.call_args
        self.assertEqual(args, (datetime.timedelta(days=365),))
        self.assertEqual((kwargs["reader"], kwargs["changes_only"]), ("sql", True))
        self.assertEqual(kwargs["overrides"], {"power": SqlOverride(duration=datetime.timedelta(days=90),
                                                                    bucket=datetime.timedelta(seconds=60))})
        _, provide = self.train_with([syn.as_power(syn.evening()[0])], {"power_history_days": "30", "power_bucket_s": "120"})
        self.assertEqual(provide.call_args.kwargs["overrides"]["power"],
                         SqlOverride(duration=datetime.timedelta(days=30), bucket=datetime.timedelta(seconds=120)))

    def test_without_a_power_input_there_is_no_override(self):
        _, provide = self.train_with(syn.evening())
        self.assertIsNone(provide.call_args.kwargs["overrides"])

    def test_power_triggers_are_trained_once(self):
        lamp, eg, og = syn.evening()
        eg, og = ((s_, [t + 90.0 for t in ts], vs) for s_, ts, vs in (eg, og))
        raw = [syn.as_power(lamp), eg, og]
        op = make_op()
        first = run_train(op, None, raw)
        self.assertEqual([r["trigger"]["kind"] for r in first.rules], ["power"])
        self.assertEqual(first.rules[0]["trigger"]["on_above_w"], 10)
        self.assertIsNone(run_train(op, FakePyFuncModel(first), raw, NOW + datetime.timedelta(days=1)))
        self.assertEqual(json.loads(main.export_json(first))["suggestions"][0]["trigger"]["hold_s"], 120)


class Training(unittest.TestCase):
    def test_first_training_returns_an_unpublished_model(self):
        model = run_train(make_op(), None, syn.evening())
        self.assertIsInstance(model, AutomationModel)
        self.assertFalse(model.published)
        self.assertEqual(len(model.rules), 1)
        self.assertEqual(model.trained_at, "2026-07-01T01:12:00Z")
        self.assertEqual(model.history, {"from": "2025-07-01", "to": "2026-07-01"})
        self.assertEqual(len(model.version), 16)

    def test_second_training_on_identical_data_returns_none(self):
        op = make_op()
        first = run_train(op, None, syn.evening())
        self.assertIsNone(run_train(op, FakePyFuncModel(first), syn.evening(), NOW + datetime.timedelta(days=1)))

    def test_changed_list_returns_a_new_unpublished_model(self):
        op = make_op()
        first = run_train(op, None, syn.evening()).as_published()
        second = run_train(op, FakePyFuncModel(first), syn.evening()[:2])  # without the second light
        self.assertIsNotNone(second)
        self.assertNotEqual(second.version, first.version)
        self.assertFalse(second.published)

    def test_empty_history_gives_an_empty_list_once(self):
        op = make_op()
        first = run_train(op, None, [])
        self.assertEqual(first.rules, [])
        self.assertIsNone(run_train(op, FakePyFuncModel(first), []))

    def test_unreadable_history_keeps_the_model_and_retries(self):
        op = make_op()
        with mock.patch("main.provide_historic_data", side_effect=RuntimeError("no cpu")), \
                mock.patch("main._topic_meta", return_value=[]), mock.patch("main.logger"):
            self.assertIsNone(op.train(None, None))
        self.assertIsNotNone(op._retry_at)
        self.assertIsNone(op._trained_day)

    def test_needs_retraining_once_per_berlin_day(self):
        op = make_op()
        with mock.patch.object(main.clock, "now", return_value=NOW):
            self.assertTrue(op.need_retraining(None))
        run_train(op, None, syn.evening())
        # 2026-07-01 01:12 UTC is 03:12 in Berlin; 21:59 UTC the same day is 23:59 there, 22:00 is the next day.
        for hour, minute, due in ((1, 12, False), (21, 59, False), (22, 0, True)):
            t = datetime.datetime(2026, 7, 1, hour, minute, tzinfo=UTC)
            with mock.patch.object(main.clock, "now", return_value=t):
                self.assertEqual(op.need_retraining(None), due, (hour, minute))


class Output(unittest.TestCase):
    def setUp(self):
        self.op = make_op()
        self.model = run_train(self.op, None, syn.evening())
        # The message is old; the output carries the time it is produced.
        self.ts = datetime.datetime(2020, 1, 1, tzinfo=UTC)
        patcher = mock.patch.object(main.clock, "now", return_value=NOW)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_infer_emits_exactly_once_and_returns_a_published_model(self):
        ts, result, new_model = self.op.infer(FakePyFuncModel(self.model), {}, "binary_state", "dev", self.ts)
        self.assertEqual(ts, NOW)
        self.assertEqual(set(result), {"suggestions", "version", "trained_at"})
        self.assertEqual(result["version"], self.model.version)
        self.assertEqual(result["trained_at"], self.model.trained_at)
        self.assertIsInstance(new_model, AutomationModel)
        self.assertTrue(new_model.published)
        self.assertEqual(new_model.version, self.model.version)
        # The registered model is the new one: the next message produces nothing and no model.
        self.assertEqual(self.op.infer(FakePyFuncModel(new_model), {}, "binary_state", "dev", self.ts), (NOW, None, None))
        self.assertEqual(self.op.infer(FakePyFuncModel(new_model), {}, "target_state", "dev", self.ts), (NOW, None, None))

    def test_exported_json_follows_the_contract(self):
        _, result, _ = self.op.infer(FakePyFuncModel(self.model), {}, "binary_state", "dev", self.ts)
        doc = json.loads(result["suggestions"])
        self.assertEqual(set(doc), {"version", "trained_at", "history", "suggestions"})
        self.assertEqual(doc["version"], result["version"])
        self.assertEqual(doc["history"], {"from": "2025-07-01", "to": "2026-07-01"})
        self.assertEqual(len(doc["suggestions"]), 1)
        self.assertEqual(doc["suggestions"][0]["window"]["from"], "19:00")

    def test_no_model_no_output(self):
        self.assertEqual(self.op.infer(None, {}, "x", "d", self.ts), (NOW, None, None))

    def test_foreign_model_counts_as_none(self):
        with mock.patch("main.logger"):
            self.assertEqual(self.op.infer(FakePyFuncModel(object()), {}, "x", "d", self.ts), (NOW, None, None))

    def test_empty_list_is_emitted_once_too(self):
        empty = AutomationModel([], "2026-07-01T01:12:00Z", {"from": "a", "to": "b"})
        _, result, new_model = self.op.infer(FakePyFuncModel(empty), {}, "x", "d", self.ts)
        self.assertEqual(json.loads(result["suggestions"])["suggestions"], [])
        self.assertTrue(new_model.published)


class History(unittest.TestCase):
    def test_rows_of_one_time_keep_their_values_and_the_majority_decides(self):
        from algo.history import signals_from_frames
        from algo.events import extract
        f = syn.frame([0.0, 100.0, 100.0, 100.0], binary_state=[False, True, True, False])
        meta = {"filterType": "DeviceId", "filterValue": "d", "mappings": [{"dest": "binary_state", "source": "p"}]}
        (sig, times, values), = signals_from_frames([(f, meta)])
        self.assertEqual(values, [False, True, True, False])
        self.assertEqual(extract(sig, times, values).events, [(100.0, "on")])

    def test_multi_device_topic_with_device_column(self):
        from algo.history import signals_from_frames
        t = [1000.0, 1010.0, 1020.0, 1030.0]
        f = syn.frame(t, binary_state=[True, False, False, True], device_id=["a", "a", "b", "b"])
        meta = {"name": "urn_infai_ses_service_abc", "filterType": "DeviceId", "filterValue": "a,b",
                "mappings": [{"dest": "binary_state", "source": "value.state"}]}
        out = signals_from_frames([(f, meta)])
        self.assertEqual({s.device_id for s, _, _ in out}, {"a", "b"})
        sig, times, values = [o for o in out if o[0].device_id == "a"][0]
        self.assertEqual((sig.service_id, sig.path, sig.port), ("urn:infai:ses:service:abc", "value.state", "binary_state"))
        self.assertEqual(list(times), [1000.0, 1010.0])
        self.assertEqual(values, [True, False])

    def test_aware_times_and_missing_values(self):
        f = pd.DataFrame({"time": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z"], utc=True),
                          "position": [0.5, None]})
        from algo.history import signals_from_frames
        out = signals_from_frames([(f, {"filterType": "DeviceId", "filterValue": "d", "mappings": []})])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0][1]), 1)
        self.assertEqual(out[0][0].path, "position")


if __name__ == "__main__":
    unittest.main()
