# alert-tsa: air raid alerts in Kyiv City

A small time series analysis of air raid alerts in Ukraine, with three
forecasts for Kyiv City. The report is `notebooks/analysis.ipynb`; the
detailed forecast definitions are in `DESIGN.md`.

## What this project found

**Kyiv City's alert history says very little about tomorrow.** The daily
series is close to noise around a slowly drifting level, and neither a
forecast of tomorrow's alert-hours nor a forecast of whether each quarter of
tomorrow will have an alert beats a simple rate. **The only usable information
is what is happening around the city right now,** and it reaches about an hour
ahead.

That is the result, not a failure to find one. Three differently framed
forecasts, each tested once against baselines chosen to be hard to beat, agree
with what the descriptive analysis shows: a lag-1 autocorrelation of 0.13, no
weekly or yearly pattern, and alerts in neighbouring oblasts that mostly begin
at the same moment as the city's.

| Forecast | Target | Result on the final 90 days |
|---|---|---|
| v1 | Tomorrow's hours under alert | No skill: no method beats a constant |
| v2 | Alert within 15 to 180 minutes, re-issued every 15 minutes at night | Brier score 3 to 4% better than the city's own history, up to 60 minutes; nothing reliable beyond |
| v3 | Any alert in each 6-hour block of tomorrow | No skill: -0.2% against the all-history rate |

What is regular is the time of day: the city is under alert about 14% of the
time between midnight and 06:00 and about 3% between 10:00 and 20:00.

## Question

How has the daily time under air raid alert in Kyiv City evolved since March
2022, what regular patterns does it have, and can tomorrow's alert-hours be
forecast better than simple baselines?

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
9. **Forecast v2: a small real signal at short leads.** Baselines (a), (b)
   and (c) are defined under "Forecast designs in brief" below. Selected model:
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

## Leakage controls

The rule is the same for all three forecasts: a function that produces a
forecast, a fitted quantity or a selection takes the history and a cutoff and
truncates to the cutoff itself.

| | What is truncated | Test |
|---|---|---|
| v1 | Daily series, to the day before the forecast day | All values after the cutoff replaced by random numbers |
| v2 | Alert records: later alerts dropped, end times of open alerts hidden | Later alerts replaced, open alerts given new end times |
| v3 | Block outcomes, to the day before the forecast day | Every outcome from the issue day onward flipped |

- **Each test asserts** that forecasts, validation forecasts and selected
  settings are unchanged by the alteration.
- **A companion test** alters data before the cutoff and asserts the forecast
  does change, so the first test cannot pass merely because a forecast is
  constant.
- **v2's rolling evaluation** works from one table built at the end of the
  data; a test asserts it matches the function that rebuilds everything from
  data known at the issue time.
- **Settings were selected on the 90 days before the test period** and, for v2
  and v3, committed before the test was run. The v2 and v3 test scripts refuse
  to run a second time.
- **The test period was not unseen for v2 and v3:** the same 90 days had been
  scored in v1.

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
python -m pytest                    # 135 tests, one to three minutes
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
DESIGN.md          forecast definitions in full
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

## Forecast designs in brief

Full definitions, including targets, clock-change handling, candidates and
selection, are in `DESIGN.md`.

- **v1:** one-day-ahead alert-hours. Baselines: mean of the last 7 days, same
  weekday last week. Models: exponential smoothing and SARIMA, refitted daily.
  Metrics: MAE and RMSE.
- **v2:** every 15 minutes from 18:00 to 06:00, the probability that Kyiv City
  is under alert at some instant in the next 15, 30, 60, 120 or 180 minutes.
  Baselines: (a) trailing 90-day hour-of-day rate; (b) the same given whether
  the city is under alert now; (c) a logistic regression adding the time
  since its last alert ended. Model: (c) plus, for Kyivska and six surrounding
  oblasts, the minutes since the oblast's current alert started. Trained on
  data after the raion-level switch only. Metric: Brier skill.
- **v3:** at midnight, the probability of any alert in each 6-hour block of
  the coming day. Baseline: all-history block rate. Candidates: trailing 7,
  14, 21 or 30 days, with and without a 50/50 blend with the same period a
  year earlier, all shrunk toward the baseline. Metric: Brier skill.
