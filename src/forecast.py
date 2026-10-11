"""Next-day forecasts of daily alert-hours: baselines, models, rolling evaluation.

Leakage rule: every function that produces a forecast or a fitted quantity
takes the full ``history`` and a ``cutoff`` day and truncates the history to
``cutoff`` itself, as its first step (``history_until``). The forecast returned
is always for the day after ``cutoff``.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

TRAIN_START = "2024-01-01"
TEST_DAYS = 90
VALIDATION_DAYS = 90
ONE_DAY = pd.Timedelta(days=1)

# Candidate settings. The winner of each family is picked on a validation
# window that ends before the test period (see select_settings).
ETS_CANDIDATES = [
    {"trend": trend, "damped_trend": trend, "seasonal": seasonal, "log": log}
    for log in (False, True)
    for trend in (False, True)
    for seasonal in (None, 7)
]
SARIMA_CANDIDATES = [
    {"order": order, "seasonal_order": seasonal_order, "log": log}
    for log in (False, True)
    for order, seasonal_order in [
        ((1, 0, 0), (0, 0, 0, 0)),
        ((1, 0, 1), (0, 0, 0, 0)),
        ((0, 1, 1), (0, 0, 0, 0)),
        ((1, 0, 0), (1, 0, 0, 7)),
        ((1, 0, 1), (1, 0, 1, 7)),
        ((0, 1, 1), (0, 1, 1, 7)),
    ]
]


def history_until(history: pd.Series, cutoff, start=TRAIN_START) -> pd.Series:
    """The part of ``history`` in [start, cutoff], checked to be gap-free daily data."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff > history.index[-1]:
        raise ValueError(f"cutoff {cutoff.date()} is after the last observed day")
    part = history.loc[start:cutoff] if start is not None else history.loc[:cutoff]
    if part.empty or part.isna().any():
        raise ValueError("history is empty or has missing values before the cutoff")
    expected = pd.date_range(part.index[0], cutoff, freq="D")
    if not part.index.equals(expected):
        raise ValueError("history must have exactly one row per day up to the cutoff")
    return part.asfreq("D").astype(float)


def _bounded(hours: float) -> float:
    return float(min(max(hours, 0.0), 24.0))


# ---- baselines ----

def forecast_mean7(history, cutoff, start=TRAIN_START) -> float:
    """Mean of the last 7 days up to and including ``cutoff``."""
    past = history_until(history, cutoff, start)
    return float(past.iloc[-7:].mean())


def forecast_same_weekday(history, cutoff, start=TRAIN_START) -> float:
    """Value 7 days before the target day, i.e. 6 days before ``cutoff``."""
    past = history_until(history, cutoff, start)
    return float(past.iloc[-7])


# Two constant references, used only to interpret the scores: the median
# minimises absolute error and the mean minimises squared error, so a method
# that cannot beat them has found no day-to-day signal.

def forecast_training_median(history, cutoff, start=TRAIN_START) -> float:
    return float(history_until(history, cutoff, start).median())


def forecast_training_mean(history, cutoff, start=TRAIN_START) -> float:
    return float(history_until(history, cutoff, start).mean())


# ---- models ----

def _build(values: pd.Series, family: str, settings: dict):
    if family == "ets":
        return ExponentialSmoothing(
            values,
            trend=settings["trend"],
            damped_trend=settings["damped_trend"],
            seasonal=settings["seasonal"],
        )
    if family == "sarima":
        order = settings["order"]
        return SARIMAX(
            values,
            order=order,
            seasonal_order=settings["seasonal_order"],
            trend="c" if order[1] == 0 and settings["seasonal_order"][1] == 0 else None,
        )
    raise ValueError(f"unknown model family {family!r}")


def _fit(values: pd.Series, family: str, settings: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)  # exp overflow in the optimiser's search
        model = _build(values, family, settings)
        try:
            return model.fit(disp=False, maxiter=200)
        except np.linalg.LinAlgError:
            # the default optimiser occasionally steps onto a parameter vector for
            # which the stationary initialisation is singular; a derivative-free
            # optimiser avoids that
            return model.fit(disp=False, maxiter=2000, method="powell")


def _transform(values: pd.Series, settings: dict) -> pd.Series:
    return np.log1p(values) if settings["log"] else values


def _back_transform(value: float, settings: dict) -> float:
    # with log=True this is the median, not the mean, of the implied distribution
    return _bounded(np.expm1(value) if settings["log"] else value)


def forecast_model(history, cutoff, family, settings, start=TRAIN_START) -> float:
    """Fit ``family`` ("ets" or "sarima") on data up to ``cutoff``; forecast the next day."""
    past = _transform(history_until(history, cutoff, start), settings)
    prediction = _fit(past, family, settings).forecast(1).iloc[0]
    return _back_transform(prediction, settings)


def forecast_ets(history, cutoff, settings, start=TRAIN_START) -> float:
    return forecast_model(history, cutoff, "ets", settings, start)


def forecast_sarima(history, cutoff, settings, start=TRAIN_START) -> float:
    return forecast_model(history, cutoff, "sarima", settings, start)


# ---- choosing settings without touching the test period ----

def validation_forecasts(history, cutoff, family, settings, validation_days=VALIDATION_DAYS,
                         start=TRAIN_START) -> pd.Series:
    """One-day-ahead forecasts for the last ``validation_days`` days up to ``cutoff``.

    Parameters are estimated once, on the data before the validation window,
    and then held fixed while the window is fed in one day at a time. Nothing
    after ``cutoff`` is used.
    """
    past = _transform(history_until(history, cutoff, start), settings)
    fit_part, validation_part = past.iloc[:-validation_days], past.iloc[-validation_days:]
    fitted = _fit(fit_part, family, settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        extended = fitted.append(validation_part, refit=False)
    predictions = extended.predict(start=validation_part.index[0], end=validation_part.index[-1])
    return predictions.map(lambda value: _back_transform(value, settings))


def select_settings(history, cutoff, family, candidates, validation_days=VALIDATION_DAYS,
                    start=TRAIN_START):
    """Pick the candidate with the lowest validation MAE. Returns (settings, table)."""
    actual = history_until(history, cutoff, start).iloc[-validation_days:]
    rows = []
    for settings in candidates:
        predictions = validation_forecasts(history, cutoff, family, settings, validation_days, start)
        rows.append({**settings, "val_mae": mae(actual, predictions), "val_rmse": rmse(actual, predictions)})
    table = pd.DataFrame(rows)
    best = candidates[int(table["val_mae"].idxmin())]
    return best, table


# ---- rolling-origin evaluation ----

def rolling_forecasts(history, forecaster, first_day, last_day, **kwargs) -> pd.Series:
    """One-day-ahead forecasts for each day D in first_day..last_day.

    The forecast for D is ``forecaster(history, cutoff=D - 1 day, **kwargs)``;
    the forecaster is responsible for truncating ``history`` at the cutoff.
    """
    days = pd.date_range(first_day, last_day, freq="D")
    values = [forecaster(history, day - ONE_DAY, **kwargs) for day in days]
    return pd.Series(values, index=days, name="forecast")


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().mean())


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(((actual - predicted) ** 2).mean()))


def test_window(history, test_days=TEST_DAYS):
    """(first test day, last test day, last day usable for choosing settings)."""
    last_day = history.index[-1]
    first_day = last_day - (test_days - 1) * ONE_DAY
    return first_day, last_day, first_day - ONE_DAY


def evaluate(history, test_days=TEST_DAYS, validation_days=VALIDATION_DAYS, start=TRAIN_START) -> dict:
    """Select settings before the test period, then score every method on it."""
    first_day, last_day, selection_cutoff = test_window(history, test_days)
    ets_settings, ets_table = select_settings(
        history, selection_cutoff, "ets", ETS_CANDIDATES, validation_days, start)
    sarima_settings, sarima_table = select_settings(
        history, selection_cutoff, "sarima", SARIMA_CANDIDATES, validation_days, start)

    forecasts = pd.DataFrame({
        "mean of last 7 days": rolling_forecasts(history, forecast_mean7, first_day, last_day, start=start),
        "same weekday last week": rolling_forecasts(
            history, forecast_same_weekday, first_day, last_day, start=start),
        "exponential smoothing": rolling_forecasts(
            history, forecast_ets, first_day, last_day, settings=ets_settings, start=start),
        "SARIMA": rolling_forecasts(
            history, forecast_sarima, first_day, last_day, settings=sarima_settings, start=start),
        "reference: training median": rolling_forecasts(
            history, forecast_training_median, first_day, last_day, start=start),
        "reference: training mean": rolling_forecasts(
            history, forecast_training_mean, first_day, last_day, start=start),
    })
    actual = history.loc[first_day:last_day]
    scores = pd.DataFrame({
        "MAE": {name: mae(actual, column) for name, column in forecasts.items()},
        "RMSE": {name: rmse(actual, column) for name, column in forecasts.items()},
    })
    return {
        "test_first_day": first_day,
        "test_last_day": last_day,
        "selection_cutoff": selection_cutoff,
        "ets_settings": ets_settings,
        "sarima_settings": sarima_settings,
        "ets_validation": ets_table,
        "sarima_validation": sarima_table,
        "actual": actual,
        "forecasts": forecasts,
        "scores": scores,
    }


def mae_difference_interval(actual, candidate, reference, block=7, draws=5000, seed=0):
    """Mean difference in absolute error (candidate - reference) with a 95% interval.

    Uses a moving-block bootstrap over test days, since daily errors are
    autocorrelated. Negative values favour the candidate.
    """
    diff = ((actual - candidate).abs() - (actual - reference).abs()).to_numpy()
    n = len(diff)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    block_starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    samples = diff[(block_starts[:, :, None] + offsets).reshape(draws, -1)[:, :n]]
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
