# Automation Suggester

Finds "if A then B" patterns in the history of a user's devices and emits the list of suggestions once per new version. A suggestion reads: when device A changes state (for example a lamp goes off) between 19:00 and 02:00, device B is switched on within minutes in a good share of the cases, far more often than at the same time of day on other days.

## Input and output

One input per kind of series, so a value is read by its port and not guessed:

| Input | Meaning |
| --- | --- |
| `binary_state` | trigger: on/off (true, "ON", 1) |
| `contact_state` | trigger: open/closed |
| `button_press` | trigger: the action text of a press |
| `position` | trigger: movement up or down in steps of 0.1 |
| `power` | trigger: a device that measures its consumption (W) switches on or off, see Power |
| `target_state` | target: on/off state of devices that can be controlled (on/off, true/false, 1/0, any case); other values are no events |
| `target_position` | target: position of a controllable device, fraction or percent, in 10 levels |
| `target_speed` | target: speed of a controllable device, in 10 levels |

Output, one message per new list:

| Name | Meaning |
| --- | --- |
| `suggestions` | JSON: `version`, `trained_at`, `history`, and `suggestions` with trigger, window, targets, `auto_off_s` and `stats` |
| `version` | hash over the suggestions' ids; statistics are not part of it |
| `trained_at` | UTC time of the training |

`function_id` of a target is `null`: the operator knows paths, not control functions. The consumer resolves the function from the device type.

## Method

1. Messages less than 2 s apart are one burst, reduced to the most frequent value. Only changes count as events.
2. A button of a device that also reports a state or position is no trigger.
3. A switching of a target that at least two other devices repeat within 60 s is a group control: automated already, neither target nor trigger.
4. Trigger series whose events fall together within 2 s in 80 % of the cases of the smaller series are one cluster. One member stays (a controllable device first, then the one with more events); the other members of the cluster are no targets of it.
5. Events of a trigger with the same `to` within `window_after_s` of the previous one are one occurrence.
6. A pair A to B (different devices) is a hit when B lies in [-300 s, +600 s] around A.
7. Baseline: the same window at the same local clock time on the 7 days before and after. Lift is confidence divided by baseline.
8. A pair with a median delay of at most 2 s (absolute) is an existing coupling and is dropped. A pair whose median signed delay is at most `min_median_delay_s` (default -10), because B mostly switches before A, is dropped. -10 s keeps targets that switch at practically the same moment as A, as a stair light switched on while the lamp goes off.
9. A binary target that is on for less than `window_before_s + window_after_s` at the median is suggested only as `on` with `auto_off_s`, never as `off`.
10. Hours that carry at least max(2, 5 %) of a trigger's hits form blocks of consecutive hours. Each block is a window of its own with its own weekdays (Monday to Friday if 90 % of the hits fall there); every target is judged inside it, and the targets form one suggestion per trigger, `to` and block.
11. Of two directions between the same two devices only the better ranked one stays.
12. `auto_off_s`: median on-time of the target if 70 % of its on-cycles are shorter than 10 minutes.
13. The best `max_suggestions` by lift times the root of the hits are kept.

A suggestion needs `min_hits` hits, `min_confidence` (`min_confidence_off` for a binary target that is switched off) and `min_lift`. Training reads `history_days` of history once per Europe/Berlin day and publishes a list only when its version differs from the last one. The state `published` is part of the registered model.

## Power

A device with a power reading is a trigger ("TV on", "TV off"). The history of `power` is read over `power_history_days` in buckets of `power_bucket_s` seconds (the maximum per bucket, then unchanged consecutive buckets left out), because power changes all the time and a day-long history of raw readings would be too much.

Per device the operator learns what "on" means from that history:

1. Each reading holds its value until the next one; all statistics weigh a row by that time, since the rows are far apart where a device stands still.
2. The log of the power (`log10(max(P, 0) + 0.1)`) is split in a low and a high class (Otsu). `low_hi` is the 95th percentile of the low class, `high_lo` the 5th percentile of the high class.
3. `on_above_w` is `sqrt(max(low_hi, 1) * high_lo)` rounded to the 1-2-5 series, `off_below_w` is half of it (so 2.5, 25, ... follow from 5, 50, ...). On is power above `on_above_w`, off is power below `off_below_w`; in between the previous state holds.
4. The device is a trigger only if `low_hi < off_below_w`, `high_lo > on_above_w`, the classes are at least 5 W apart and the device is on between 0.5 % and 95 % of the time. A device that never switches off (a base load) or hardly ever draws power is not.
5. A change counts when the new state stands longer than 60 s plus one bucket: the maximum per bucket stretches a short spike to one or two buckets, so a state counts only when it is longer than the hold and the bucket. The same holds for an off inside an on, so the history counts what the process, holding `hold_s`, would switch on. A row weighs at most a day, so weeks offline with the last value standing do not decide the levels. The event is at the start of the first bucket in the new state.
6. A device that switches on more than 12 times a day, or at least 3 times a day at regular intervals (coefficient of variation of the gaps below 0.3), is a thermostat and gives no events.
7. A power event within `power_bucket_s` + 60 s of a `binary_state` change of the same device is that device being switched; the binary trigger covers it and the power event is dropped.
8. A target that follows a power trigger within `power_bucket_s` is a coupling (rule 8 of Method uses `power_bucket_s` instead of `coupling_max_delay_s`).

Power is never a target. The trigger in the export carries its numbers, and they are part of the suggestion's id:

    {"device_id": "...", "service_id": "...", "path": "value.sensor.ENERGY.Power", "kind": "power",
     "from": "off", "to": "on", "on_above_w": 10, "off_below_w": 5, "hold_s": 120}

The consumer applies the thresholds as given: on above `on_above_w`, off below `off_below_w`, a change counts after `hold_s` seconds without a relapse. `hold_s` is 60 s plus one bucket, the span after which the history counts a change, so the process does not react to the short spikes the training left out.

## Config

`history_days` 365, `power_history_days` 90, `power_bucket_s` 60, `window_before_s` 300, `window_after_s` 600, `min_hits` 10, `min_lift` 3, `min_confidence` 0.25, `min_confidence_off` 0.5, `coupling_max_delay_s` 2, `min_median_delay_s` -10, `max_suggestions` 10, `targets` (device ids, comma separated or JSON list; empty: every device on a target port).

## Development

    uv run --frozen python -m unittest discover -s tests -t .
    uv run --frozen python -m operator_lib manifest main.py

`scripts/replay_history.py` runs the same mining on local responses of `/db/v3/queries/v2`, without network. Test data is synthetic. A `power` series in the mapping is read as the operator reads it (window, bucket maximum, `changes_only`), and its thresholds and verdict are printed to stderr.

## Notes

- Times of day, hours of a window and weekdays are Europe/Berlin, fixed and not configurable: the process that executes a suggestion computes Berlin time itself. `window.tz` in the export is always `Europe/Berlin`.
- The service of a series is taken from the topic name (`urn_infai_ses_service_...`, `_` read as `:`). Service ids that contain `#` are not recovered this way.
- Rows of one series with the same time and different values are kept and decided by majority per burst (see Method 1); the platform's history reader may already have reduced them to the last one.
- The mining is quadratic in the number of series: about 0.2 s for 10 series, 0.8 s for 20 and 2.8 s for 40 at 1500 events per series (reviewer's measurement). It runs once a day in the training task.
- The history is read with the library's `reader="sql"`: all inputs in one Ray task over one database connection, a table retried up to three times on a dropped connection. With `changes_only=True` only rows whose value differs from a neighbouring row are read, about 5 % of all rows. Repeats of an unchanged value inside a burst drop out with them, so a burst is decided by its distinct values more than by their counts; in a replay over one household this changed the counts of three suggestions slightly and dropped one borderline suggestion out of six. A dataset that still fails makes the whole read count as failed (the training is repeated later).
