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
The history as Signals with times and values, from the datasets of provide_historic_data.

One dataset per input topic: a `time` column and one column per mapping, named after the mapping's dest, which here
is the port. A topic naming several devices adds a `device_id` column. The topic's name carries the service
(urn:infai:ses:service:... with ':' written as '_'), a mapping's source is the path of the value in its output.
"""

__all__ = ("signals_from_datasets", "signals_from_frames", "service_id_of")

import typing

import numpy as np
import pandas as pd
import ray

from .events import PORTS, Signal

SERVICE_PREFIX = "urn_infai_ses_service_"


def service_id_of(topic_name: typing.Optional[str]) -> str:
    name = topic_name or ""
    return name.replace("_", ":") if name.startswith(SERVICE_PREFIX) else name


def _read_frame(ds) -> pd.DataFrame:
    return (ray.get(ds) if isinstance(ds, ray.ObjectRef) else ds).to_pandas()


def _device_groups(frame: pd.DataFrame, meta: dict) -> typing.List[typing.Tuple[str, pd.DataFrame]]:
    if meta.get("filterType") == "DeviceId" and "device_id" in frame.columns:
        keyed = frame[frame["device_id"].notna()]
        return [(str(k), g) for k, g in keyed.groupby(keyed["device_id"].astype(str))]
    if meta.get("filterType") != "DeviceId" and "original_input_ids" in frame.columns:
        keyed = frame[frame["original_input_ids"].notna()]
        return [(str(k), g) for k, g in keyed.groupby(keyed["original_input_ids"].astype(str))]
    raw = str(meta.get("filterValue") or "")
    devices = [d for d in (part.strip() for part in raw.split(",")) if d]
    return [(devices[0] if len(devices) == 1 else raw.strip(), frame)]


def signals_from_frames(frames: typing.Sequence[typing.Tuple[pd.DataFrame, dict]]) -> typing.List[typing.Tuple[Signal, np.ndarray, list]]:
    """
    (signal, epoch seconds, raw values) per device, service, path and port. Duplicate series of one signal, as two
    topics of the same device and path, are merged by time.
    """
    parts: typing.Dict[Signal, typing.List[pd.DataFrame]] = {}
    for frame, meta in frames:
        if "time" not in frame.columns:
            continue
        paths = {m.get("dest"): m.get("source") for m in meta.get("mappings") or []}
        service = service_id_of(meta.get("name"))
        for port in PORTS:
            if port not in frame.columns:
                continue
            for device, group in _device_groups(frame, meta):
                sub = group[["time", port]].rename(columns={port: "value"})
                sub = sub[sub["value"].notna()]
                if sub.empty:
                    continue
                signal = Signal(device, service, str(paths.get(port) or port), port)
                parts.setdefault(signal, []).append(sub)
    out = []
    for signal, subs in parts.items():
        merged = pd.concat(subs, ignore_index=True)
        # Remote Timescale delivers aware times, Kafka naive UTC.
        t = pd.to_datetime(merged["time"], utc=True, errors="coerce")
        merged = merged.assign(_t=t)[t.notna().to_numpy()]
        merged = merged.sort_values("_t", kind="stable")
        # Rows of one time stay, also with different values: debounce decides those by majority.
        times = ((merged["_t"] - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
        out.append((signal, times, merged["value"].tolist()))
    return out


def signals_from_datasets(datasets, topic_meta: typing.Sequence[dict], notes: typing.List[str]):
    """The signals of all datasets, and whether a requested history could not be read."""
    if datasets is None:
        notes.append("no history requested")
        return [], False
    failed = False
    frames = []
    for i, ds in enumerate(datasets):
        try:
            frame = _read_frame(ds)
        except Exception as ex:
            failed = True
            notes.append(f"history of topic {i} not readable: {type(ex).__name__}: {ex}")
            continue
        frames.append((frame, topic_meta[i] if i < len(topic_meta) else {}))
    return signals_from_frames(frames), failed
