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

__all__ = ("SCHEMA_VERSION", "AutomationModel", "export_json")

import json
import typing

import mlflow.pyfunc

from .mining import list_version

SCHEMA_VERSION = 1


class AutomationModel(mlflow.pyfunc.PythonModel):
    """
    The suggestion list of one training. `published` says whether infer() has emitted it; the state lives in the
    registered model, so a restart does not emit a list a second time.
    """

    def __init__(self, rules: typing.List[dict], trained_at: str, history: typing.Dict[str, str],
                 published: bool = False, schema_version: int = SCHEMA_VERSION,
                 seen_ids: typing.Optional[typing.Set[str]] = None):
        self.schema_version = schema_version
        self.rules = rules
        self.version = list_version(r["id"] for r in rules)
        self.trained_at = trained_at
        self.history = history
        self.published = published
        # The ids of all lists emitted before this one; None while this is the first list of the model.
        self.seen_ids = seen_ids

    @property
    def ids(self) -> typing.Set[str]:
        return {r["id"] for r in self.rules}

    @property
    def new_count(self) -> int:
        """How many ids of this list were in no list emitted before; 0 for the first list."""
        seen = getattr(self, "seen_ids", None)  # a model stored before the field existed has none
        return 0 if seen is None else len(self.ids - seen)

    def successor(self, rules: typing.List[dict], trained_at: str, history: typing.Dict[str, str]) -> "AutomationModel":
        """The next list. Only a list that was emitted counts as seen: an unpublished one was replaced unseen."""
        seen = getattr(self, "seen_ids", None)
        if self.published:
            seen = (seen or set()) | self.ids
        return AutomationModel(rules, trained_at, history, seen_ids=seen)

    def as_published(self) -> "AutomationModel":
        return AutomationModel(self.rules, self.trained_at, self.history, True, self.schema_version,
                               getattr(self, "seen_ids", None))

    # No type hints: mlflow checks them only where present, and the model is never called through predict.
    def predict(self, context, model_input, params=None):
        return [r["id"] for r in self.rules]


def export_json(model: AutomationModel) -> str:
    """The value of the output `suggestions`: the whole list with its version, see the export contract."""
    return json.dumps({
        "version": model.version,
        "trained_at": model.trained_at,
        "history": model.history,
        "suggestions": model.rules,
    }, separators=(",", ":"), ensure_ascii=False)
