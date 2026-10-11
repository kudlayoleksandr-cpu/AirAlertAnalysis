# Forecast design

Definitions for the three forecasts. Findings, limitations, data cleaning and
assumptions are in `README.md`. The v2 and v3 sections were written before
their code and fixed before their test runs.

## Forecast v1: next-day alert-hours

- **Target:** Kyiv City alert-hours for day D, from data up to the end of
  day D-1.
- **Training data:** 2024-01-01 onward, expanding by one day per forecast.
- **Test period:** the final 90 days, 2026-05-31 to 2026-08-28. Every method
  is refitted for each test day.
- **Baselines:** mean of the last 7 days; same weekday last week.
- **Models:** exponential smoothing and SARIMA, both from
  `statsmodels.tsa.statespace`.
- **Settings:** eight exponential smoothing and twelve SARIMA candidates
  (listed in `forecast.py`), half of each on the raw scale and half with a
  `log1p` transform. The
  winner of each family has the lowest one-day-ahead MAE on the 90 days before
  the test period (2026-03-02 to 2026-05-30), with parameters estimated on the
  data before that window. Nothing from 2026-05-31 onward is used.
- **Forecasts are limited to 0 to 24 hours.** With the log transform the
  back-transformed forecast is a median, not a mean.
- **Two constant references** (training median and training mean) were added
  to help read the scores. They are not part of the requested comparison.

### Leakage controls (v1)

- Every forecasting and fitting function takes the full history and a cutoff
  day and truncates the history itself (`forecast.history_until`).
- `tests/test_forecast.py` replaces all data after the cutoff with random
  values and asserts that each forecast, the validation forecasts and the
  chosen settings are unchanged.
- A second test changes data before the cutoff and asserts the forecast does
  change, so the first test cannot pass merely because a forecast is constant.

## Forecast v2: short-lead alert warnings for Kyiv City

v1 found no day-to-day skill. v2 asks a different question: during the
evening and night, is an alert in Kyiv City about to begin? The definitions
in this section were fixed before any v2 code was written.

### Target

- **Issue times:** every 15 minutes on the clock (:00, :15, :30, :45) with a
  Europe/Kyiv wall-clock hour from 18 to 23 or 0 to 5. That is 48 issue times
  on a normal night.
- **Lead times L:** 15, 30, 60, 120 and 180 minutes of elapsed time.
- **Target:** for issue time t and lead L, y = 1 if Kyiv City is under alert
  at any instant in [t, t + L), else 0.
- **Under alert at t** means an alert with `started_at <= t < finished_at`.
- **Alert start:** when Kyiv City is not under alert at t, y = 1 means an
  alert starts within L minutes. When it is under alert at t, y = 1 by
  definition, and every method except baseline (a) outputs 1 there.
- **Night:** issue times are grouped by the local date of t minus 6 hours, so
  00:00 to 05:45 belongs to the previous evening.

### Clock changes

- Kyiv's UTC offset is always a whole number of hours, so the 15-minute grid
  is the same in UTC and local time.
- Lead times and target windows are elapsed time and are never adjusted.
- Issue times are selected by wall-clock hour. The night of the spring change
  has no 03:00 hour and 44 issue times; the night of the autumn change has
  the 03:00 hour twice and 52.
- Hour of day, where used, is the wall-clock hour of the issue time. The two
  03:00 hours of the autumn night are pooled.

### Baselines

- **(a) Trailing hour-of-day rate.** For issue time t and lead L: the mean of
  y over issue times s with the same wall-clock hour, the same L,
  t - 90 days <= s, and s + L <= t (so the outcome is already known at t).
- **(b) Hour-of-day rate plus own state.** If Kyiv City is under alert at t,
  1. Otherwise the same trailing rate computed only over past issue times at
  which Kyiv City was not under alert.
- **(c) Own recent history.** A logistic regression on the logit of (b) and
  the minutes since Kyiv City's last alert ended (capped at 7 days, `log1p`).
  Alerts cluster within a night, so this is the hardest baseline: it knows
  everything about Kyiv City and nothing about its neighbours.
- Rates are limited to [0.001, 0.999] and need at least 30 past issue times.

### Model

- **Inputs:** those of baseline (c), plus one neighbour feature per oblast in
  the chosen set: minutes since its current alert started at t, or 0 if none
  is active. The model and (c) are nested, so a gain over (c) is attributable
  to the neighbours.
- **"Alert active in an oblast"** means an oblast-level or raion-level row of
  that oblast is active. The current alert started at the beginning of the
  current unbroken spell of such rows.
- **Hromada-level rows are ignored.** They cover one community, and every
  standing alert (over 7 days) is hromada-level, so this also removes those
  without using an alert's future length.
- **No "days since start" or other trend feature.**
- **Estimator:** logistic regression (scikit-learn, standardised inputs), one
  per lead. Per oblast it gets an "alert active" flag and `log1p` of the
  minutes: a value of 0 means no alert, which is not the low end of "minutes
  since it started", so a linear model needs the flag. This encoding was
  settled on synthetic data, before any real validation run.
- **Training rows:** issue times at which Kyiv City is not under alert and
  whose outcome is known at the fitting time.
- **Refitting:** once per night, at 18:00, on data known at that moment.
  Forecasts later that night use the 18:00 model with features observed at
  their own issue time.
- **No threat type.** The file does not say what an alert was declared for,
  so the model sees only that an alert is active and for how long.

### The raion-level switch

Models are trained on data from 2025-11-05 onward only.

- The oblast feature changes meaning at the switch. Before it, an oblast was
  alerted as a whole; after it, each raion is alerted separately, so "an alert
  is active somewhere in the oblast" is true more often and for longer spells.
- A definition that is stable across the switch would need raion-level
  information for the earlier period, which the file does not contain: an
  oblast-level row says nothing about which raions were actually threatened.
- A recent training window was requested anyway, and by the test period every
  candidate window lies after the switch. Restricting training to post-switch
  data costs little and removes the problem instead of modelling it.
- Baselines (a) and (b) use Kyiv City rows only, which did not change.

### Selection and test

- **Test period:** nights with evening dates 2026-05-31 to 2026-08-28. On the
  last evening only issue times whose window ends by local midnight count.
- **Validation period:** the 90 nights before it, 2026-03-02 to 2026-05-30.
- **Chosen on validation only:** the oblast set (Kyivska alone; Kyivska and
  six surrounding oblasts; all other oblasts) and the training window (60
  days, 90 days, or everything since the switch) for the model, and the
  training window for baseline (c).
- **Selection criterion:** mean Brier score over the five leads on validation
  issue times when Kyiv City is not under alert.
- **Pipeline:** `python scripts/run_nowcast.py select` uses data known at
  18:00 on 2026-05-31 and writes `results/nowcast_selection.json`.
  `python scripts/run_nowcast.py test` runs the test period once, writes
  `results/nowcast_test_forecasts.csv` and refuses to overwrite it. The
  selection was committed before the test was run.

### Metrics

- **Brier score** per lead, and Brier skill 1 - BS_model / BS_baseline
  against (a), (b) and (c).
- **Headline:** issue times when Kyiv City is not under alert. Scores over
  all issue times are also given, but there the target is 1 by definition
  whenever the city is already under alert.
- **Uncertainty:** 95% intervals from a bootstrap that resamples whole nights.

### Lead-lag analysis

- **Question:** do alerts in the surrounding oblasts lead alerts in Kyiv City,
  and by how long?
- **Series:** alert state (1 = active) sampled every 15 minutes, for Kyiv City
  and for each of Kyivska and the six surrounding oblasts, from 2025-11-05.
- **Hour-of-day cycle removed:** the mean for each quarter-hour of the day is
  subtracted from every series before correlating. Without this, any two
  series that are both busier at night would appear related.
- **Statistic:** correlation between the oblast's state at t and Kyiv City's
  at t + lag, for lags of -6 h to +6 h. A positive lag means the oblast leads.
- **Data:** up to 18:00 on 2026-05-31 only, like the selection.
- **Caveat:** state series are strongly autocorrelated, so peaks are broad and
  nearby lags are not independent evidence.

### Leakage controls

- `nowcast.known_at(alerts, cutoff)` drops alerts that start after the cutoff
  and hides the end time of alerts still open at the cutoff.
- Every feature, baseline rate, fitted model and lead-lag correlation is built
  from that view.
- Tests replace everything after a cutoff (new alerts, end times of open
  alerts) and assert that features, baselines, forecasts, selected settings
  and lead-lag correlations up to the cutoff are unchanged.
- A companion test changes data before the cutoff and asserts the forecast
  does change.
- The rolling evaluation works from one table built at the end of the data.
  A test asserts it gives the same forecasts as the function that rebuilds
  everything from data known at the issue time.

### What was dropped from v2

A tree-based model and alert-start precision and recall tables were planned
and then removed, to keep v2 a time series analysis with one small model. The
tree model was never run to completion. An exploratory validation run of an
earlier logistic model (without the own-history inputs) was looked at before
this design was final; nothing from the test period was.

## Forecast v3: day-ahead alert probability by 6-hour block

A coarser question than v2 and a more forgiving target than v1: not how many
hours, but whether there will be any alert in each quarter of tomorrow.

### Target

- **Blocks:** 00:00-06:00, 06:00-12:00, 12:00-18:00 and 18:00-24:00,
  Europe/Kyiv wall-clock time.
- **Target:** y = 1 if Kyiv City is under alert at any instant in the block.
- **Issue time:** midnight at the start of the day being forecast, using only
  data known at that instant. An alert still open at midnight is known to be
  open; its end time is not used.
- **Clock changes:** blocks follow the wall clock, so 00:00-06:00 lasts 5 or
  7 hours on the two change days. Not corrected.
- **Alert running at midnight:** then the 00:00-06:00 outcome is 1 by
  definition. No method here uses that state, so the 00:00-06:00 block is
  reported separately for days with and without an alert running at midnight.

### Methods

Nothing is fitted; every method is a rate over past days, per block.

- **Baseline:** all-history block rate, from 2022-03-16 to the day before the
  forecast day.
- **Candidates 1 to 4:** trailing block rate over 7, 14, 21 or 30 days.
- **Candidates 5 to 8:** the same, with a "same period last year" term: a
  fixed 50/50 blend of the trailing rate and the block rate in the 30 days
  centred on the date 365 days earlier. A fixed blend was chosen over a
  fitted weight to keep every method free of fitting.
- **Shrinkage:** every trailing or last-year rate is pulled toward the
  all-history rate as if 7 extra days at that rate had been observed:
  (alerts in window + 7 x history rate) / (days in window + 7). The 7 is a
  fixed prior, not tuned. Without it a short window gives probabilities of
  exactly 0 or 1.
- **Reference:** trailing 90-day rate with the same shrinkage. Section 3
  showed the level of the series drifts, so any recent window should beat the
  all-history rate a little. The reference shows whether a shorter window
  adds anything beyond that. It is not a candidate.

### Selection and test

- **Test period:** 2026-05-31 to 2026-08-28, 90 days and 360 forecasts.
- **Validation period:** the 90 days before it, 2026-03-02 to 2026-05-30.
- **Selection:** the candidate with the lowest Brier score over all four
  blocks on the validation period.
- **All eight candidates, the baseline and the reference are fixed in
  `src/blocks.py`** (`CANDIDATES`, `BASELINE`, `REFERENCE`), committed before
  the test was run.
- **Pipeline:** `python scripts/run_blocks.py select`, then
  `python scripts/run_blocks.py test`, which runs once and refuses to
  overwrite its output.
- **The test period was not unseen.** The same 90 days were scored in v1 and
  v2 and plotted in the notebook before v3 was designed, and it was known
  that late August 2026 was the heaviest stretch of the series. The design is
  mechanical and was fixed in code first, but this is weaker than a truly
  untouched test set.

### Metrics

- **Brier score** over all blocks and per block, with skill against the
  baseline and against the reference.
- **Uncertainty:** 95% intervals from a bootstrap that resamples whole days.

### Leakage controls

- `blocks.block_outcomes(alerts, issue_day)` builds outcomes from
  `nowcast.known_at(alerts, midnight)`.
- `forecast_blocks`, `rolling_forecasts` and `select` take the outcomes and
  an issue or cutoff day and use only rows for earlier days.
- Tests flip every outcome from the issue day onward and assert each method's
  forecast and the selection are unchanged, and that the forecast does change
  when the day before is altered.
