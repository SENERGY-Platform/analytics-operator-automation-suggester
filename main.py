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

import dotenv
dotenv.load_dotenv()

import datetime
import json
import time
import typing
import zoneinfo

import ray

from operator_lib import util
from operator_lib.util import MLOperator, Config, Inputs, Outputs, logger
from operator_lib.util import clock
from operator_lib.util.helpers import provide_historic_data, SqlOverride

from algo.mining import Params
from algo.model import SCHEMA_VERSION, AutomationModel, export_json
from algo.training import fit_rules, train_rules

__all__ = ("Operator", "CustomConfig", "OperatorInputs", "OperatorOutputs")

# A requested history that cannot be read, as when the Ray cluster has no CPU free, is tried again this often, this
# far apart, before the day counts as trained.
HISTORY_RETRY_S = 600
HISTORY_RETRIES = 6
BERLIN = zoneinfo.ZoneInfo("Europe/Berlin")
UTC = datetime.timezone.utc


def _number(value, default, cast):
    # A designer node hands over a config value it never set as an empty string.
    if value is None or str(value).strip() == "":
        return default
    try:
        return cast(str(value).strip())
    except ValueError:
        logger.warning(f"config value {value!r} is no {cast.__name__}, using {default}")
        return default


def _device_list(value) -> typing.Tuple[str, ...]:
    """Device ids as a JSON list or a comma separated string."""
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        text = str(value or "").strip()
        if text.startswith("["):
            try:
                items = [str(v) for v in json.loads(text)]
            except ValueError:
                items = []
        else:
            items = text.split(",")
    out = []
    for item in items:
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return tuple(out)


class CustomConfig(Config):
    # Redeclared so the manifest lists it, with the library's default.
    logger_level: str = "warning"

    history_days: int = 365
    # Power readings change all the time, so they are read per bucket (the maximum in it) and over a shorter time.
    power_history_days: int = 90
    power_bucket_s: int = 60
    window_before_s: int = 300
    window_after_s: int = 600
    min_hits: int = 10
    min_lift: float = 3.0
    min_confidence: float = 0.25
    min_confidence_off: float = 0.5
    coupling_max_delay_s: float = 2.0
    min_median_delay_s: float = -10.0
    max_suggestions: int = 10
    # Device ids that count as controllable, comma separated or a JSON list. Empty: every device on a target port.
    targets: str = ""

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        cls = type(self)
        for name, cast in (("history_days", int), ("power_history_days", int), ("power_bucket_s", int), ("max_suggestions", int), ("window_before_s", int), ("window_after_s", int), ("min_hits", int),
                           ("min_lift", float), ("min_confidence", float), ("min_confidence_off", float), ("coupling_max_delay_s", float), ("min_median_delay_s", float)):
            setattr(self, name, _number(getattr(self, name), getattr(cls, name), cast))
        for name in ("power_history_days", "power_bucket_s"):
            if getattr(self, name) < 1:
                logger.warning(f"{name} {getattr(self, name)} is below 1, using {getattr(cls, name)}")
                setattr(self, name, getattr(cls, name))


def params_of(config: CustomConfig) -> Params:
    # Not a method: simple_struct copies every public class attribute into the instance, a method included.
    return Params(window_before_s=config.window_before_s, window_after_s=config.window_after_s,
                  min_hits=config.min_hits, min_lift=config.min_lift, min_confidence=config.min_confidence,
                  min_confidence_off=config.min_confidence_off,
                  coupling_max_delay_s=config.coupling_max_delay_s,
                  min_median_delay_s=config.min_median_delay_s,
                  max_suggestions=config.max_suggestions, power_bucket_s=config.power_bucket_s,
                  targets=_device_list(config.targets))


class OperatorInputs(Inputs):
    # One port per kind, so the reading of a value is not guessed from the value.
    binary_state: bool
    contact_state: str
    button_press: str
    position: float
    target_state: str
    target_position: float
    target_speed: float
    # The consumption of a device in W; a trigger only, read as "on" or "off" against thresholds learned from the history.
    power: float


class OperatorOutputs(Outputs):
    suggestions: str
    version: str
    trained_at: str
    new_count: int


def _unwrap(model) -> typing.Optional[AutomationModel]:
    """The model behind what mlflow loaded; one of another schema version counts as none."""
    if model is None:
        return None
    if hasattr(model, "unwrap_python_model"):
        model = model.unwrap_python_model()
    if not isinstance(model, AutomationModel) or model.schema_version != SCHEMA_VERSION:
        logger.warning(f"ignoring a model of unexpected type or schema version: {type(model).__name__}")
        return None
    return model


def _topic_meta() -> typing.List[dict]:
    topics = util.OperatorConfig(util.load_operator_config_json(util.DeploymentConfig())).inputTopics
    return [{"name": t.name, "filterType": t.filterType, "filterValue": t.filterValue,
             "mappings": [{"dest": m.dest, "source": m.source} for m in t.mappings]} for t in topics]


def power_overrides(config: CustomConfig, topic_meta: typing.Sequence[dict]) -> typing.Optional[typing.Dict[str, SqlOverride]]:
    """The window and bucket for the input `power`; none when no topic maps it, as with an instance that predates it."""
    if not any(m["dest"] == "power" for t in topic_meta for m in t["mappings"]):
        return None
    days, bucket = config.power_history_days, config.power_bucket_s
    return {"power": SqlOverride(duration=datetime.timedelta(days=days), bucket=datetime.timedelta(seconds=bucket))}


def _berlin_date(t: datetime.datetime) -> datetime.date:
    t = t.replace(tzinfo=UTC) if t.tzinfo is None else t
    return t.astimezone(BERLIN).date()


class Operator(MLOperator):
    configType = CustomConfig
    inputs = OperatorInputs
    outputs = OperatorOutputs

    def __init__(self):
        # The Berlin day of the last finished training. Not the model's, which stays when nothing changed.
        self._trained_day: typing.Optional[datetime.date] = None
        self._retry_at: typing.Optional[float] = None  # monotonic time of the next training after an unreadable history
        self._history_retries = 0

    # training

    def train(self, model, train_logger):
        datasets = None
        try:
            topic_meta = _topic_meta()
            datasets = provide_historic_data(datetime.timedelta(days=self.config.history_days), reader="sql", changes_only=True,
                                             overrides=power_overrides(self.config, topic_meta))
        except Exception as ex:
            logger.warning(f"history not available: {type(ex).__name__}: {ex}")
        now = clock.now()
        trained = None
        unreadable = datasets is None
        if datasets is not None:
            try:
                trained, notes, history_failed = ray.get(train_rules.remote(
                    datasets, topic_meta, now, params_of(self.config), self.config.history_days))
                for note in notes:
                    logger.info(note)
                unreadable = history_failed
            except Exception:
                logger.exception("training task failed")
                unreadable = True

        previous = _unwrap(model)
        if unreadable and self._history_retries < HISTORY_RETRIES:
            # The day stays untrained, so the next training reads the history again and the model stays.
            self._history_retries += 1
            self._retry_at = time.monotonic() + HISTORY_RETRY_S
            logger.warning(f"history unreadable, training again in {HISTORY_RETRY_S // 60} minutes "
                           f"(retry {self._history_retries} of {HISTORY_RETRIES})")
            return None
        self._history_retries = 0
        self._retry_at = None
        self._trained_day = _berlin_date(now)

        if unreadable and previous is not None:
            return None  # out of retries: the list stays as it is, a partial history must not replace it
        if trained is None:
            trained = AutomationModel([], now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), {})
        if train_logger is not None:
            try:
                train_logger.log_metrics({"suggestions": len(trained.rules)})
            except Exception:
                # A metrics store that is down must not keep the trained model from the operator.
                logger.exception("could not log the training metrics")
        if previous is not None and previous.version == trained.version:
            return None
        if previous is not None:
            trained = previous.successor(trained.rules, trained.trained_at, trained.history)
        return trained

    def need_retraining(self, model) -> bool:
        if self._retry_at is not None:
            return time.monotonic() >= self._retry_at
        return self._trained_day != _berlin_date(clock.now())

    # output

    def infer(self, model, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        now = clock.now()  # the time of the output, not of the message that happened to trigger it
        current = _unwrap(model)
        if current is None or current.published:
            return now, None, None
        result = {"suggestions": export_json(current), "version": current.version, "trained_at": current.trained_at,
                  "new_count": current.new_count}
        return now, result, current.as_published()


if __name__ == "__main__":
    from operator_lib.operator_lib import OperatorLib

    op = Operator()
    OperatorLib(op, name="automation-suggester-operator", git_info_file="git_commit")
