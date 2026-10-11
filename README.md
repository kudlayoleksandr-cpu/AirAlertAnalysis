# alert-tsa: air raid alerts in Kyiv City

A small time series analysis of air raid alerts in Ukraine, with three
forecasts for Kyiv City. The report is `notebooks/analysis.ipynb`.

| Forecast | Target | Result |
|---|---|---|
| v1 | Tomorrow's hours under alert | No skill: no method beats a constant |
| v2 | Alert within 15 to 180 minutes, re-issued every 15 minutes at night | Small real gain from neighbouring oblasts up to 60 minutes |
| v3 | Any alert in each 6-hour block of tomorrow | No skill against the all-history rate |

## Question

How has the daily time under air raid alert in Kyiv City evolved since March
2022, what regular patterns does it have, and can tomorrow's alert-hours be
forecast better than simple baselines?

## Data

| | |
|---|---|
| Repository | https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset |
| Commit | `bf0c323c78e3e4508721458797dcd38c95df13f3` |
| File | `datasets/official_data_en.csv` |
| SHA-256 of the downloaded file | `b396f87696243069eec0219880e87dcd642b20beb0a7bae0d0e1f61e1ed20ec4` |

Columns: `oblast, raion, hromada, level, started_at, finished_at, source`. All
timestamps are UTC. The volunteer file in the same repository is not used; it
records roughly half the alert time of the official file (checked by the
project owner, not re-checked here).

## How to run

```
pip install -r requirements.txt
python scripts/fetch_data.py        # downloads the file at the pinned commit
python -m src.load                  # prints the validation report
python -m pytest                    # 169 tests, about three minutes
python -m nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

The notebook takes about five minutes, almost all of it in the v1 forecast
cell. Its v2 and v3 sections read the saved files in `results/`. Those were
produced by `scripts/run_nowcast.py` and `scripts/run_blocks.py` (`select`,
then `test`); the `test` step refuses to run again while its output exists.
Tests that need the real file are skipped if it has not been fetched.

Versions used: Python 3.13, pandas 3.0.6, statsmodels 0.15.0, scikit-learn
1.9.1, matplotlib 3.11.2, numpy 2.5.4, scipy 1.18.1, pytest 9.1.1.

## Layout

```
src/load.py        load, validate, dedupe, clean
src/series.py      intervals -> daily and hourly series
src/forecast.py    v1: baselines, models, rolling evaluation
src/nowcast.py     v2: short-lead forecast and lead-lag analysis
src/blocks.py      v3: day-ahead 6-hour block probabilities
scripts/fetch_data.py
scripts/run_nowcast.py, scripts/run_blocks.py    selection, then one test run
results/           selections and test forecasts for v2 and v3
notebooks/analysis.ipynb
tests/
```

## Validation of the raw file

All of these are asserted in `tests/test_load.py` against the real file.

| Check | Result |
|---|---|
| Rows | 299,275 |
| Alert starts | 2022-03-15 16:10 UTC to 2026-09-07 05:28 UTC, not sorted |
| Exact duplicate rows | 113,845 (38.0%); 185,430 unique rows |
| `finished_at <= started_at` | 4 rows (2 unique), all zero-length |
| Alerts over 24 h | 1,318 raw, 897 unique; longest 14,497 h |
| Timestamps | all parse, all `+00:00` |
| `source` | always `official` |
| `level` vs filled fields | always consistent |
| Rows starting 2026-08-30 to 09-01 (UTC) | 0 |
| Rows starting 2026-08-29 (UTC) | 88 (240 to 598 per day over 1 to 28 Aug) |
| Feed break | last start 2026-08-29 04:55 UTC, next 2026-09-02 13:48 UTC |
| Next-longest nationwide silence | 19 h |
| Kyiv City levels | oblast only |

## Cleaning rules

Applied in `load.clean`, in this order.

1. **Exact duplicates are dropped.** The file is an older export (113,911
   rows, to 2026-01-25) followed by a full newer export, so no row appears
   more than twice and duplicates stop after 2026-01-25.
2. **Rows with `finished_at <= started_at` are dropped.**
3. **The analysis ends on 2026-08-28**, the last complete Europe/Kyiv day
   before the feed break. Rows starting at or after 2026-08-28 21:00 UTC
   (local midnight) are dropped. Eleven rows that straddle that instant are
   clipped to it; none is in Kyiv City, Kharkivska or Lvivska.
4. **Long alerts are flagged, not dropped**: `long_alert` (over 24 h) and
   `standing_alert` (over 7 days).

Result: 183,318 rows.

### Long alerts

- Kyiv City and Lvivska oblast are unaffected: their longest alerts are 11.2 h
  and 6.1 h.
- Before the cutoff, 850 unique alerts exceed 24 h, almost all in Donetska
  raions and the Nikopol-area hromadas of Dnipropetrovska.
- Thirteen exceed 7 days. All are hromada-level alerts in frontline areas.
- Two of the thirteen are in Kharkivska oblast: Lypetska hromada (2024-05-12
  to 2026-01-06) and Vovchanska hromada (2024-05-20 to 2025-08-01). Under "an
  alert is active anywhere in the oblast" they hold Kharkivska at 24 h/day for
  20 months.

Rule: alerts longer than 7 days are excluded from the oblast-wide series used
in the regional comparison (`series.region_rows`, `include_standing=False`).
For Kharkivska any threshold between 51 h and 10,000 h selects the same two
rows, which is asserted in `tests/test_series.py`. The notebook also plots the
series with them included.

## Series definitions

- **Daily series:** hours under alert per Europe/Kyiv calendar day, computed
  from the union of alert intervals, so overlapping alerts are counted once
  and an alert crossing midnight is split between the days.
- **Hourly series:** the same per elapsed hour, indexed by local hour.
- **Kyiv City:** rows with `oblast == "Kyiv City"` (all oblast-level).
- **Kharkivska and Lvivska oblasts:** an alert is active anywhere in the
  oblast, that is, the union of oblast-, raion- and hromada-level rows.
- **Range:** 2022-03-16 to 2026-08-28, 1,627 days.

Sanity totals for Kyiv City: 121.78 h in February 2025 and 74.97 h in August
2026. The August figure is lower than the roughly 80 h from the initial
profiling because that included 4.5 h on the discarded 2026-08-29.

## Assumptions

1. **The series starts on 2022-03-16, not 2022-03-15.** The file begins at
   16:10 UTC on the 15th, so that day is partial. This departs from the brief,
   which said 2022-03-15 onward.
2. **A time with no alert row means no alert.** Apart from the 2026 break, the
   file gives no way to tell a quiet period from missing data. Kyiv City's
   longest gap between alerts is 15.5 days in October 2023, in the quietest
   stretch of the series.
3. **Duplicates are export artefacts,** not separate alerts. An interval union
   would give the same result either way.
4. **Sixty-six Zhytomyrska rows survive deduplication.** The older export
   labels a raion-level alert as hromada-level. They do not affect the three
   regions analysed and would not change a union.
5. **Alerts open at the cutoff were still active until the end of 2026-08-28.**
   Their recorded end times fall after the feed resumed and are not trusted.
6. **Alerts longer than 7 days are real standing alerts.** This is inferred
   from their location in frontline hromadas and has not been checked against
   another source.
7. **Clock-change days are not corrected.** They are 23 or 25 hours long, and
   a day's value is elapsed hours under alert.
8. **The raion-level switch is treated as complete on 2025-11-05,** with
   August to early November 2025 as a transition. The share of oblast-level
   rows falls from 58% in July 2025 to 27% in August, 15% in September and
   October, and 2% in November. Outside Kyiv City the last oblast-level row is
   dated 2025-11-04. The timing differs by oblast: Kharkivska's last
   oblast-level row is 2025-10-19 and Lvivska's is 2025-11-02.
9. **Seasonality, STL and the forecast use 2024-01-01 onward.** The trend and
   regional sections use the full history.
10. **STL uses period 7 with a seasonal window of 91 days.** With the
    statsmodels default window of 7 the weekly component absorbs individual
    spikes; the notebook shows both.
11. **Statistical tests use Newey-West (HAC) errors,** with 14 lags for the
    day-of-week and before/after tests and 28 for the annual harmonic. The lag
    lengths were chosen by judgement, not tuned.

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

## Findings

1. **Trend.** No steady rise or fall. The 28-day mean moves in multi-month
   waves between 0.1 and 5.3 h/day around an overall mean of 1.59 h/day. 2023
   was the quietest year (1.10 h/day) and 2025 the heaviest (2.03). About 29%
   of days have no alert.
2. **Seasonality.**
   - Hour of day: strong. The city is under alert about 14% of the time
     between 00:00 and 06:00 and about 3% between 10:00 and 20:00.
   - Day of week: none detected (joint test p = 0.17).
   - Yearly: not reliable. Heavy months differ from year to year (rank
     correlation of monthly means between 2024 and 2025 is -0.39), an annual
     harmonic is borderline (p about 0.05, 2% of variance), and the
     autocorrelation at lag 365 is within noise. With 2.7 cycles a small
     yearly effect cannot be excluded.
3. **Autocorrelation and stationarity.** Autocorrelation is weak: 0.13 at lag
   1 and at most 0.17. ADF rejects a unit root (p < 0.001). KPSS rejects a
   constant level for 2024 onward (p = 0.02) and is borderline for the full
   history (p = 0.07). The series is noise around a slowly shifting level.
4. **STL.** The remainder carries close to 90% of the variance, the trend
   about a quarter, and the weekly component under 1%.
5. **Structural change.** No shift in Kyiv City can be attributed to the
   raion-level switch. The mean changes by -0.21 h/day with 90 days either
   side (p = 0.58) and by -0.59 with 180 days (p = 0.02), so the result depends
   on the window. Alerts per day and median duration are nearly unchanged, and
   Kyiv City's own recording did not change.
6. **Regions.** Kharkivska rises from 11.2 h/day (2024-08 to 2025-07) to 19.7
   h/day after the switch; Lvivska stays at 0.3 to 0.4 h/day; Kyiv City is
   between them. The oblast definition gets looser after the switch, so the
   two periods are not directly comparable and the Kharkivska rise cannot be
   read as a change in threat.
7. **Forecast v1: weak.** No method shows day-to-day skill.

   | Method | MAE | RMSE |
   |---|---|---|
   | Baseline: mean of last 7 days | 1.82 | 2.81 |
   | Baseline: same weekday last week | 2.14 | 3.27 |
   | Exponential smoothing | 1.59 | 2.85 |
   | SARIMA | 1.60 | 2.87 |
   | Reference: constant training median | 1.55 | 2.92 |
   | Reference: constant training mean | 1.79 | 2.76 |

   Both models beat the two baselines on MAE and are slightly worse than the
   7-day mean on RMSE. Neither beats a constant: the training median has the
   lowest MAE and the training mean the lowest RMSE. The models' MAE advantage
   comes from predicting a low, almost constant value (their forecasts vary by
   about 0.15 h against 2.8 h for the actuals), not from anticipating heavy
   days.

8. **Lead-lag with surrounding oblasts** (2025-11-05 to 2026-05-31,
   hour-of-day cycle removed). Every oblast's correlation with Kyiv City
   peaks at lag zero: alerts mostly begin together. The strongest are Kyivska
   (0.51), Vinnytska (0.51), Cherkaska (0.46) and Zhytomyrska (0.44); Sumska
   (0.12) and Chernihivska (0.19) are the weakest. There is a slight
   asymmetry toward the neighbours leading (Kyivska: 0.33 when it leads by 60
   minutes, 0.27 when Kyiv City does).
9. **Forecast v2: a small real signal at short leads.** Selected model:
   Kyivska and the six surrounding oblasts, trained on all post-switch data.
   Test period, issue times when Kyiv City is not under alert:

   | Lead | Share with an alert start | Skill vs (a) | Skill vs (b) | Skill vs (c) | 95% interval vs (c) |
   |---|---|---|---|---|---|
   | 15 min | 2.8% | +0.19 | +0.035 | +0.031 | +0.014 to +0.048 |
   | 30 min | 5.5% | +0.11 | +0.044 | +0.038 | +0.015 to +0.061 |
   | 60 min | 10.3% | +0.06 | +0.040 | +0.031 | +0.003 to +0.060 |
   | 120 min | 18.7% | +0.02 | +0.029 | +0.017 | -0.019 to +0.053 |
   | 180 min | 25.6% | +0.01 | +0.025 | +0.011 | -0.030 to +0.053 |

   - Against the hardest baseline the neighbours improve the Brier score by 3
     to 4% up to 60 minutes. At 120 and 180 minutes the gain is not
     distinguishable from zero.
   - The long-lead skill seen on validation (+0.068 against (c) at 180
     minutes) did not carry over to the test period.
   - Kyiv City's own recent history adds almost nothing: (c) beats (b) by 0.4
     to 1.5%, with intervals that include zero.
   - Most of the skill against (a) comes from setting aside issue times that
     are already under alert, not from forecasting.
   - A 3% Brier improvement on an event with a 3 to 10% base rate is too
     small to be a usable warning.
10. **Forecast v3: no skill.** The selected trailing 21-day rate scores a
    Brier of 0.2337 against 0.2333 for the all-history rate (skill -0.002,
    95% interval -0.05 to +0.05). The all-history rate was already the best
    of the ten methods on validation. All scores are close to the 0.25 of
    always saying 50%. The last-year variants look best on the test period
    (up to +0.014) but were the worst on validation and were not selected;
    that difference is noise and is not claimed as a result.

Taken together: for Kyiv City the history of alerts says very little about
tomorrow, whether the target is hours (v1) or occurrence (v3). The only usable
information found is what is happening around the city right now (v2), and it
reaches about an hour ahead.

## Limitations

- **One source.** Nothing here is checked against an independent record of
  alerts.
- **Possible undetected gaps.** The file has one known outage. Shorter ones
  would look like quiet periods.
- **Alert time is not attack intensity.** It records a declared threat, and
  declaration practice changed during the period.
- **Few annual cycles.** The yearly-pattern conclusion rests on 2.7 years.
- **One test window.** Forecast scores come from a single 90-day period that
  contains the two heaviest days of the whole series (14.0 h on 2026-08-20 and
  14.6 h on 2026-08-27); RMSE is driven by a handful of days.
- **Heavy days just before the feed break.** Those two days fall in the last
  nine days before the outage. Their rows look ordinary (many separate alerts
  of normal length), but it cannot be ruled out that the feed was already
  misbehaving.
- **Selection metric.** Settings were chosen by validation MAE. Choosing by
  RMSE would pick different variants; this was not evaluated on the test
  period.
- **Descriptive sections use the full series,** including the test period.
  No forecast setting was taken from them.
- **Regional comparison.** The oblast series mixes two recording regimes and,
  for Kharkivska, depends on excluding two standing alerts.
- **v2 and v3 share the v1 test period.** Settings were fixed and committed
  before each test run, but the 90 days had already been scored in v1.
- **v2 assumes real-time publication.** An alert's start is treated as known
  at its start time. If the feed records starts late or backfills them, the
  measured skill is too high.
- **v2 has no threat type.** The file does not say why an alert was declared,
  which is probably the most useful missing input.
- **v2 uses post-switch data only,** about 200 nights before the test period,
  and the test period was busier than the validation period (Kyiv City under
  alert at 9.3% of issue times against 6.1%).
- **v2 exploratory look.** An earlier logistic model without the own-history
  inputs was scored on the validation period before the final design.
