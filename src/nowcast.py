"""Forecast v2: is an alert in Kyiv City about to begin?

Every 15 minutes through the evening and night, estimate the probability that
Kyiv City is under alert at some instant in the next L minutes. Definitions
are in the README section "Forecast v2".

Leakage rule: everything is derived from ``known_at(alerts, cutoff)``, which
drops alerts that start after the cutoff and hides the end time of alerts
still open at it. ``build_table``, ``forecast_at``, ``select`` and ``lead_lag``
take the alerts and a cutoff and apply that view themselves; ``fit_models`` takes the table and a
fitting cutoff and uses only rows whose outcome is known by then.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.load import KYIV_TZ
from src.series import covered_hours_before, merge_intervals

TARGET = "Kyiv City"
LEADS = (15, 30, 60, 120, 180)  # minutes
STEP = pd.Timedelta(minutes=15)
NIGHT_HOURS = (18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5)
RATE_DAYS = 90
MIN_RATE_COUNT = 30
RATE_BOUNDS = (0.001, 0.999)
TABLE_START = "2025-05-01"  # earliest issue time built; leaves 90 days of rates before the switch
SWITCH_DONE = pd.Timestamp("2025-11-05", tz=KYIV_TZ)  # models never train on earlier data
VALIDATION_NIGHTS = 90
FIRST_TEST_EVENING = "2026-05-31"
LAST_TEST_EVENING = "2026-08-28"
OPEN_HORIZON = pd.Timedelta(days=365)  # stand-in end for an alert still open at the cutoff
FEATURE_LEVELS = ("oblast", "raion")  # hromada-level rows are ignored, see README

KYIVSKA = ["Kyivska oblast"]
RING = KYIVSKA + ["Chernihivska oblast", "Sumska oblast", "Poltavska oblast",
                  "Cherkaska oblast", "Zhytomyrska oblast", "Vinnytska oblast"]
ALL_OTHERS = RING + [
    "Chernivetska oblast", "Dnipropetrovska oblast", "Donetska oblast", "Ivano-Frankivska oblast",
    "Kharkivska oblast", "Khersonska oblast", "Khmelnytska oblast", "Kirovohradska oblast",
    "Luhanska oblast", "Lvivska oblast", "Mykolaivska oblast", "Odeska oblast", "Rivnenska oblast",
    "Ternopilska oblast", "Volynska oblast", "Zakarpatska oblast", "Zaporizka oblast",
]
OBLAST_SETS = {"none": [], "kyivska": KYIVSKA, "ring": RING, "all": ALL_OTHERS}
OWN_GAP_CAP = 7 * 24 * 60  # minutes; "a long time since Kyiv City's last alert"

WINDOWS = (60, 90, None)  # training window in days; None = everything since the switch
# Baseline (c): hour-of-day rate, own state and time since Kyiv City's last alert. No neighbours.
BASELINE_C_CANDIDATES = [{"oblasts": "none", "window_days": window} for window in WINDOWS]
# The model: the inputs of (c) plus the neighbour features.
MODEL_CANDIDATES = [
    {"oblasts": oblasts, "window_days": window}
    for oblasts in ("kyivska", "ring", "all")
    for window in WINDOWS
]


def _utc(stamp) -> pd.Timestamp:
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is None:
        raise ValueError("times must be timezone-aware")
    return stamp.tz_convert("UTC")


def feature_column(oblast: str) -> str:
    return f"mins_{oblast}"


# ---- what is known at a given moment ----

def known_at(alerts: pd.DataFrame, cutoff) -> pd.DataFrame:
    """The alert records as they could be known at ``cutoff``.

    Alerts starting after the cutoff are dropped. Alerts still open at the
    cutoff keep their start, are marked ``open_at_cutoff`` and get a stand-in
    end far in the future, so that their real end time cannot be used.
    """
    cutoff = _utc(cutoff)
    visible = alerts.loc[alerts["started_at"] <= cutoff, ["oblast", "level", "started_at", "finished_at"]].copy()
    visible["open_at_cutoff"] = visible["finished_at"] > cutoff
    visible["finished_at"] = visible["finished_at"].where(~visible["open_at_cutoff"], cutoff + OPEN_HORIZON)
    return visible


def _active_and_minutes(merged: pd.DataFrame, instants: pd.DatetimeIndex):
    """For each instant: is a merged interval active, and minutes since it started (else 0)."""
    start = pd.DatetimeIndex(merged["started_at"]).as_unit("ns").asi8
    finish = pd.DatetimeIndex(merged["finished_at"]).as_unit("ns").asi8
    at = instants.as_unit("ns").asi8
    if len(start) == 0:
        return np.zeros(len(at), dtype=bool), np.zeros(len(at))
    k = np.searchsorted(start, at, side="right") - 1
    last = np.maximum(k, 0)
    active = (k >= 0) & (finish[last] > at)
    minutes = np.where(active, (at - start[last]) / 60e9, 0.0)
    return active, minutes


def _minutes_since_last_end(merged: pd.DataFrame, instants: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since the most recent interval ended; 0 while one is active; capped at OWN_GAP_CAP."""
    start = pd.DatetimeIndex(merged["started_at"]).as_unit("ns").asi8
    finish = pd.DatetimeIndex(merged["finished_at"]).as_unit("ns").asi8
    at = instants.as_unit("ns").asi8
    if len(start) == 0:
        return np.full(len(at), float(OWN_GAP_CAP))
    k = np.searchsorted(start, at, side="right") - 1
    last = np.maximum(k, 0)
    gap = np.where(k >= 0, (at - finish[last]) / 60e9, OWN_GAP_CAP)
    return np.clip(gap, 0.0, OWN_GAP_CAP)


def _trailing_rate(times_ns, known_ns, hour, outcome, eligible):
    """Mean of past outcomes with the same hour of day, known by each time.

    For row i: rows j with hour[j] == hour[i], eligible[j], known_ns[j] <= times_ns[i]
    and times_ns[j] >= times_ns[i] - RATE_DAYS.
    """
    rate = np.full(len(times_ns), np.nan)
    window = int(pd.Timedelta(days=RATE_DAYS).value)
    for h in np.unique(hour):
        rows = np.flatnonzero(hour == h)
        source = rows[eligible[rows] & ~np.isnan(outcome[rows])]
        totals = np.r_[0.0, np.cumsum(outcome[source])]
        upper = np.searchsorted(known_ns[source], times_ns[rows], side="right")
        lower = np.searchsorted(times_ns[source], times_ns[rows] - window, side="left")
        count = upper - lower
        enough = count >= MIN_RATE_COUNT
        rate[rows[enough]] = (totals[upper[enough]] - totals[lower[enough]]) / count[enough]
    return np.clip(rate, *RATE_BOUNDS)


def build_table(alerts: pd.DataFrame, cutoff, start=TABLE_START) -> pd.DataFrame:
    """One row per 15-minute issue time up to ``cutoff``, from data known at ``cutoff``.

    Columns: local hour, night, is_night, kyiv_on, own_gap (minutes since Kyiv
    City's last alert ended), minutes-since-alert-start per oblast, and for each lead L: y_L (NaN if not yet known at the cutoff),
    rate_a_L and rate_b_L (the two baselines).
    """
    cutoff = _utc(cutoff)
    visible = known_at(alerts, cutoff)
    first = pd.Timestamp(start).tz_localize(KYIV_TZ).tz_convert("UTC")
    grid = pd.date_range(first, cutoff.floor("15min"), freq=STEP)
    local = grid.tz_convert(KYIV_TZ)
    table = pd.DataFrame(index=grid)
    table.index.name = "issue_time"
    table["hour"] = local.hour
    # wall-clock arithmetic, so that a clock change cannot move an issue time to another night
    table["night"] = (local.tz_localize(None) - pd.Timedelta(hours=6)).normalize()
    table["is_night"] = np.isin(local.hour, NIGHT_HOURS)

    own = visible[visible["oblast"] == TARGET]
    kyiv = merge_intervals(own["started_at"], own["finished_at"])
    kyiv_on, _ = _active_and_minutes(kyiv, grid)
    table["kyiv_on"] = kyiv_on
    table["own_gap"] = _minutes_since_last_end(kyiv, grid)

    relevant = visible[visible["level"].isin(FEATURE_LEVELS)]
    by_oblast = {name: rows for name, rows in relevant.groupby("oblast")}
    for oblast in ALL_OTHERS:
        rows = by_oblast.get(oblast)
        if rows is None:
            table[feature_column(oblast)] = 0.0
            continue
        merged = merge_intervals(rows["started_at"], rows["finished_at"])
        table[feature_column(oblast)] = _active_and_minutes(merged, grid)[1]

    times_ns = grid.as_unit("ns").asi8
    hour = table["hour"].to_numpy()
    covered = covered_hours_before(kyiv, grid)
    everyone = np.ones(len(grid), dtype=bool)
    for lead in LEADS:
        window_end = grid + pd.Timedelta(minutes=lead)
        outcome = (covered_hours_before(kyiv, window_end) - covered > 1e-9).astype(float)
        outcome[window_end > cutoff] = np.nan  # not yet known at the cutoff
        known_ns = window_end.as_unit("ns").asi8
        table[f"y_{lead}"] = outcome
        table[f"rate_a_{lead}"] = _trailing_rate(times_ns, known_ns, hour, outcome, everyone)
        rate_b = _trailing_rate(times_ns, known_ns, hour, outcome, ~kyiv_on)
        table[f"rate_b_{lead}"] = np.where(kyiv_on, 1.0, rate_b)
    return table


# ---- models ----

def _logit(p):
    return np.log(p / (1 - p))


def _design(rows: pd.DataFrame, lead: int, spec: dict) -> np.ndarray:
    columns = [feature_column(oblast) for oblast in OBLAST_SETS[spec["oblasts"]]]
    return np.column_stack([_logit(rows[f"rate_b_{lead}"].to_numpy()), rows["own_gap"].to_numpy(),
                            rows[columns].to_numpy().reshape(len(rows), len(columns))])


def _encode(X):
    # columns: logit of baseline (b), own gap, then minutes since alert start per oblast.
    # For the neighbours 0 means "no alert", which is not the low end of "minutes
    # since it started": a linear model needs the active flag alongside the log minutes.
    neighbours = X[:, 2:]
    return np.column_stack([X[:, 0], np.log1p(X[:, 1]), neighbours > 0, np.log1p(neighbours)])


def _new_estimator():
    return make_pipeline(FunctionTransformer(_encode), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000))


def training_rows(table: pd.DataFrame, fit_cutoff, lead: int, spec: dict) -> pd.DataFrame:
    """Night issue times, not under alert, post-switch, in the window, outcome known by ``fit_cutoff``."""
    fit_cutoff = _utc(fit_cutoff)
    train_from = SWITCH_DONE.tz_convert("UTC")
    if spec["window_days"] is not None:
        train_from = max(train_from, fit_cutoff - pd.Timedelta(days=spec["window_days"]))
    known_by = table.index + pd.Timedelta(minutes=lead)
    keep = (
        table["is_night"].to_numpy() & ~table["kyiv_on"].to_numpy()
        & (table.index >= train_from) & (known_by <= fit_cutoff)
        & table[f"y_{lead}"].notna().to_numpy() & table[f"rate_b_{lead}"].notna().to_numpy()
    )
    return table[keep]


def fit_models(table: pd.DataFrame, fit_cutoff, spec: dict) -> dict:
    """One fitted estimator per lead, or None where the training outcomes are all alike."""
    models = {}
    for lead in LEADS:
        rows = training_rows(table, fit_cutoff, lead, spec)
        outcome = rows[f"y_{lead}"].to_numpy()
        if len(rows) == 0 or outcome.min() == outcome.max():
            models[lead] = None
            continue
        models[lead] = _new_estimator().fit(_design(rows, lead, spec), outcome)
    return models


def predict(models: dict, rows: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Probabilities per lead for ``rows``. Under alert -> 1; no fitted model -> baseline (b)."""
    out = pd.DataFrame(index=rows.index)
    off = ~rows["kyiv_on"].to_numpy()
    for lead in LEADS:
        p = rows[f"rate_b_{lead}"].to_numpy(dtype=float).copy()
        usable = off & ~np.isnan(p)
        if models[lead] is not None and usable.any():
            p[usable] = models[lead].predict_proba(_design(rows[usable], lead, spec))[:, 1]
        p[~off] = 1.0
        out[f"p_{lead}"] = p
    return out


def evening_start(evening) -> pd.Timestamp:
    """18:00 Europe/Kyiv on the given date, in UTC: when that night's models are fitted."""
    return (pd.Timestamp(evening).normalize().tz_localize(None).tz_localize(KYIV_TZ)
            + pd.Timedelta(hours=18)).tz_convert("UTC")


def night_forecasts(table: pd.DataFrame, evening, spec: dict) -> pd.DataFrame:
    """Forecasts for every issue time of one night, from models fitted at its 18:00."""
    evening = pd.Timestamp(evening)
    rows = table[table["is_night"] & (table["night"] == evening)]
    models = fit_models(table, evening_start(evening), spec)
    return predict(models, rows, spec)


def rolling_forecasts(table: pd.DataFrame, first_evening, last_evening, spec: dict) -> pd.DataFrame:
    nights = pd.date_range(first_evening, last_evening, freq="D")
    return pd.concat([night_forecasts(table, evening, spec) for evening in nights])


def forecast_at(alerts: pd.DataFrame, issue_time, spec: dict) -> pd.Series:
    """Probabilities per lead at one issue time, using only data known at that time."""
    issue_time = _utc(issue_time)
    table = build_table(alerts, issue_time)
    if issue_time not in table.index or not table.loc[issue_time, "is_night"]:
        raise ValueError("issue_time must be a quarter-hour between 18:00 and 05:45 Kyiv time")
    night = night_forecasts(table, table.loc[issue_time, "night"], spec)
    return night.loc[issue_time]


# ---- scoring ----

def brier(outcome, probability) -> float:
    outcome, probability = np.asarray(outcome, dtype=float), np.asarray(probability, dtype=float)
    return float(np.mean((probability - outcome) ** 2))


def skill(outcome, probability, reference) -> float:
    """Brier skill score: 1 - BS(probability) / BS(reference). Positive is better than the reference."""
    return 1.0 - brier(outcome, probability) / brier(outcome, reference)


def scored_frame(table: pd.DataFrame, forecasts: dict) -> pd.DataFrame:
    """Join outcomes, own state and both baselines to the model forecasts (keyed by method name)."""
    first = next(iter(forecasts.values()))
    frame = table.loc[first.index, ["night", "hour", "kyiv_on"]].copy()
    for lead in LEADS:
        frame[f"y_{lead}"] = table.loc[first.index, f"y_{lead}"]
        frame[f"a_{lead}"] = table.loc[first.index, f"rate_a_{lead}"]
        frame[f"b_{lead}"] = table.loc[first.index, f"rate_b_{lead}"]
        for name, predicted in forecasts.items():
            frame[f"{name}_{lead}"] = predicted[f"p_{lead}"]
    return frame


def _scorable(frame: pd.DataFrame, lead: int, methods, starts_only: bool) -> pd.DataFrame:
    columns = [f"y_{lead}"] + [f"{m}_{lead}" for m in methods]
    rows = frame[frame[columns].notna().all(axis=1)]
    return rows[~rows["kyiv_on"]] if starts_only else rows


def brier_table(frame: pd.DataFrame, methods, starts_only: bool) -> pd.DataFrame:
    """Brier score per lead and method, and the skill of each method against every one listed before it."""
    rows_out = []
    for lead in LEADS:
        rows = _scorable(frame, lead, methods, starts_only)
        y = rows[f"y_{lead}"]
        entry = {"lead_min": lead, "n": len(rows), "base_rate": float(y.mean())}
        for m in methods:
            entry[f"brier_{m}"] = brier(y, rows[f"{m}_{lead}"])
        for position, m in enumerate(methods):
            for reference in methods[:position]:
                entry[f"skill_{m}_vs_{reference}"] = skill(y, rows[f"{m}_{lead}"], rows[f"{reference}_{lead}"])
        rows_out.append(entry)
    return pd.DataFrame(rows_out).set_index("lead_min")


def skill_interval(frame, lead, method, reference, starts_only=True, draws=2000, seed=0):
    """Brier skill with a 95% interval from resampling whole nights."""
    rows = _scorable(frame, lead, [method, reference], starts_only)
    y = rows[f"y_{lead}"].to_numpy()
    err_m = (rows[f"{method}_{lead}"].to_numpy() - y) ** 2
    err_r = (rows[f"{reference}_{lead}"].to_numpy() - y) ** 2
    codes, _ = pd.factorize(rows["night"])
    n_nights = codes.max() + 1
    sum_m = np.bincount(codes, weights=err_m, minlength=n_nights)
    sum_r = np.bincount(codes, weights=err_r, minlength=n_nights)
    picks = np.random.default_rng(seed).integers(0, n_nights, size=(draws, n_nights))
    samples = 1.0 - sum_m[picks].sum(axis=1) / sum_r[picks].sum(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return 1.0 - err_m.sum() / err_r.sum(), float(low), float(high)


# ---- selection on data before the test period ----

def validation_evenings(cutoff, nights=VALIDATION_NIGHTS):
    """(first, last) evening of the validation period: the ``nights`` nights that end before ``cutoff``."""
    last = _utc(cutoff).tz_convert(KYIV_TZ).tz_localize(None).normalize() - pd.Timedelta(days=1)
    return last - pd.Timedelta(days=nights - 1), last


def mean_start_brier(frame: pd.DataFrame, method: str) -> float:
    """Selection criterion: mean over leads of the Brier score on alert-start rows."""
    scores = brier_table(frame, [method], starts_only=True)
    return float(scores[f"brier_{method}"].mean())


def select(alerts: pd.DataFrame, cutoff, model_candidates=None, baseline_c_candidates=None,
           nights=VALIDATION_NIGHTS) -> dict:
    """Choose the settings of baseline (c) and of the model, using data known at ``cutoff``.

    Each candidate is run night by night over the validation period and scored
    by ``mean_start_brier``.
    """
    model_candidates = MODEL_CANDIDATES if model_candidates is None else model_candidates
    baseline_c_candidates = BASELINE_C_CANDIDATES if baseline_c_candidates is None else baseline_c_candidates
    table = build_table(alerts, cutoff)
    first, last = validation_evenings(cutoff, nights)

    def best_of(candidates):
        runs = [rolling_forecasts(table, first, last, spec) for spec in candidates]
        scores = [mean_start_brier(scored_frame(table, {"m": run}), "m") for run in runs]
        summary = pd.DataFrame([{**spec, "val_brier": value} for spec, value in zip(candidates, scores)])
        winner = int(np.argmin(scores))
        return candidates[winner], runs[winner], summary

    baseline_c, run_c, table_c = best_of(baseline_c_candidates)
    model, run_model, table_model = best_of(model_candidates)
    return {
        "cutoff": _utc(cutoff),
        "validation_first_evening": first,
        "validation_last_evening": last,
        "baseline_c": baseline_c,
        "model": model,
        "baseline_c_table": table_c,
        "model_table": table_model,
        "validation_frame": scored_frame(table, {"c": run_c, "model": run_model}),
    }


# ---- lead-lag analysis ----

def lead_lag(alerts: pd.DataFrame, cutoff, oblasts=tuple(RING), max_lag_minutes=360,
             start=SWITCH_DONE) -> pd.DataFrame:
    """Cross-correlation between each oblast's alert state and Kyiv City's, by lag.

    State is sampled every 15 minutes from ``start`` to ``cutoff`` (1 = alert
    active). The mean for each quarter-hour of the day is subtracted from every
    series first, so that two series which merely share a daily cycle are not
    reported as related. A positive lag means the oblast leads: its state at
    time t is compared with Kyiv City's at t + lag.
    """
    cutoff = _utc(cutoff)
    visible = known_at(alerts, cutoff)
    grid = pd.date_range(_utc(start), cutoff.floor("15min"), freq=STEP)
    local = grid.tz_convert(KYIV_TZ)
    slot = np.asarray(local.hour * 4 + local.minute // 15)

    def state(rows):
        active, _ = _active_and_minutes(merge_intervals(rows["started_at"], rows["finished_at"]), grid)
        values = pd.Series(active.astype(float))
        return (values - values.groupby(slot).transform("mean")).to_numpy()

    kyiv = state(visible[visible["oblast"] == TARGET])
    relevant = visible[visible["level"].isin(FEATURE_LEVELS)]
    steps = np.arange(-(max_lag_minutes // 15), max_lag_minutes // 15 + 1)
    result = pd.DataFrame(index=pd.Index(steps * 15, name="lag_minutes"))
    for oblast in oblasts:
        other = state(relevant[relevant["oblast"] == oblast])
        result[oblast] = [_correlation(other, kyiv, int(step)) for step in steps]
    return result


def _correlation(leader, follower, step: int) -> float:
    """Correlation of leader[t] with follower[t + step]; 0 if either side is constant."""
    if step > 0:
        a, b = leader[:-step], follower[step:]
    elif step < 0:
        a, b = leader[-step:], follower[:step]
    else:
        a, b = leader, follower
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
