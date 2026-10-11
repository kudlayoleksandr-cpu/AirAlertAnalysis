import numpy as np
import pandas as pd
import pytest

from src import forecast

START = "2025-01-01"
ETS = {"trend": False, "damped_trend": False, "seasonal": 7, "log": True}
SARIMA = {"order": (1, 0, 1), "seasonal_order": (1, 0, 0, 7), "log": False}

FORECASTERS = {
    "mean7": lambda history, cutoff: forecast.forecast_mean7(history, cutoff, start=START),
    "same_weekday": lambda history, cutoff: forecast.forecast_same_weekday(history, cutoff, start=START),
    "ets": lambda history, cutoff: forecast.forecast_ets(history, cutoff, ETS, start=START),
    "sarima": lambda history, cutoff: forecast.forecast_sarima(history, cutoff, SARIMA, start=START),
}


@pytest.fixture(scope="module")
def history():
    rng = np.random.default_rng(42)
    days = pd.date_range(START, periods=240, freq="D")
    weekly = 1.5 * np.sin(2 * np.pi * np.arange(240) / 7)
    values = np.clip(4 + weekly + rng.gamma(2.0, 1.0, 240), 0, 24)
    return pd.Series(values, index=days)


def _corrupt_after(history, cutoff, seed=7):
    """Replace every value after ``cutoff`` with unrelated numbers."""
    corrupted = history.copy()
    after = corrupted.index > pd.Timestamp(cutoff)
    assert after.sum() > 0
    corrupted[after] = np.random.default_rng(seed).uniform(0, 24, after.sum())
    return corrupted


# ---- leakage ----

@pytest.mark.parametrize("name", FORECASTERS)
def test_forecast_ignores_everything_after_the_cutoff(history, name):
    cutoff = history.index[199]
    original = FORECASTERS[name](history, cutoff)
    assert FORECASTERS[name](_corrupt_after(history, cutoff), cutoff) == original
    # also unchanged when the future is simply absent
    assert FORECASTERS[name](history.loc[:cutoff], cutoff) == original


@pytest.mark.parametrize("function", [forecast.forecast_training_median, forecast.forecast_training_mean])
def test_constant_references_ignore_everything_after_the_cutoff(history, function):
    cutoff = history.index[199]
    original = function(history, cutoff, start=START)
    assert function(_corrupt_after(history, cutoff), cutoff, start=START) == original
    assert function(history, history.index[150], start=START) != original


@pytest.mark.parametrize("name", FORECASTERS)
def test_forecast_does_depend_on_the_last_day_before_the_cutoff(history, name):
    # guards against the leakage test passing only because the forecast is constant
    cutoff = history.index[199]
    changed = history.copy()
    changed.loc[cutoff - pd.Timedelta(days=6):cutoff] += 5.0
    assert FORECASTERS[name](changed, cutoff) != FORECASTERS[name](history, cutoff)


def test_rolling_forecast_for_day_d_ignores_day_d_and_later(history):
    first, last = history.index[200], history.index[209]
    target = history.index[204]
    corrupted = _corrupt_after(history, target - pd.Timedelta(days=1))
    for name in ["mean7", "same_weekday", "ets"]:
        original = forecast.rolling_forecasts(history, FORECASTERS[name], first, last)
        altered = forecast.rolling_forecasts(corrupted, FORECASTERS[name], first, last)
        # forecasts up to and including the target day are identical ...
        pd.testing.assert_series_equal(original.loc[:target], altered.loc[:target])
        # ... and later ones, which may legitimately see the altered days, differ
        # (not same_weekday: within this window it only looks at days before the target)
        if name != "same_weekday":
            assert not np.allclose(original.loc[target + pd.Timedelta(days=1):],
                                   altered.loc[target + pd.Timedelta(days=1):])


def test_model_selection_ignores_everything_after_the_cutoff(history):
    cutoff = history.index[199]
    candidates = forecast.ETS_CANDIDATES[:4]
    best, table = forecast.select_settings(history, cutoff, "ets", candidates, validation_days=30, start=START)
    best2, table2 = forecast.select_settings(
        _corrupt_after(history, cutoff), cutoff, "ets", candidates, validation_days=30, start=START)
    assert best == best2
    pd.testing.assert_frame_equal(table, table2)


def test_validation_forecasts_ignore_data_after_the_cutoff_and_cover_the_window(history):
    cutoff = history.index[199]
    predictions = forecast.validation_forecasts(history, cutoff, "sarima", SARIMA, validation_days=30, start=START)
    altered = forecast.validation_forecasts(
        _corrupt_after(history, cutoff), cutoff, "sarima", SARIMA, validation_days=30, start=START)
    pd.testing.assert_series_equal(predictions, altered)
    assert predictions.index[0] == cutoff - pd.Timedelta(days=29)
    assert predictions.index[-1] == cutoff


def test_validation_forecast_for_a_day_does_not_use_that_day(history):
    cutoff = history.index[199]
    target = cutoff - pd.Timedelta(days=10)
    changed = history.copy()
    changed.loc[target] += 10.0
    a = forecast.validation_forecasts(history, cutoff, "ets", ETS, validation_days=30, start=START)
    b = forecast.validation_forecasts(changed, cutoff, "ets", ETS, validation_days=30, start=START)
    np.testing.assert_allclose(a.loc[:target], b.loc[:target], rtol=0, atol=1e-12)
    assert abs(a.loc[target + pd.Timedelta(days=1)] - b.loc[target + pd.Timedelta(days=1)]) > 1e-6


def test_evaluate_selects_before_the_test_window_and_scores_it(history, monkeypatch):
    # keep this fast: two candidates for one family, one for the other
    ets_other = {**ETS, "seasonal": None}
    monkeypatch.setattr(forecast, "ETS_CANDIDATES", [ETS, ets_other])
    monkeypatch.setattr(forecast, "SARIMA_CANDIDATES", [SARIMA])
    short = history.iloc[:120]
    result = forecast.evaluate(short, test_days=10, validation_days=20, start=START)
    assert result["selection_cutoff"] == result["test_first_day"] - pd.Timedelta(days=1)
    assert len(result["forecasts"]) == 10
    assert result["forecasts"].index.equals(result["actual"].index)
    assert set(result["scores"].columns) == {"MAE", "RMSE"}

    # the chosen settings must not change when the whole test window is altered
    altered = forecast.evaluate(
        _corrupt_after(short, result["selection_cutoff"]), test_days=10, validation_days=20, start=START)
    assert altered["ets_settings"] == result["ets_settings"]
    pd.testing.assert_frame_equal(altered["ets_validation"], result["ets_validation"])
    pd.testing.assert_frame_equal(altered["sarima_validation"], result["sarima_validation"])


# ---- behaviour of the baselines ----

def test_mean7_and_same_weekday_values():
    days = pd.date_range("2025-03-01", periods=14, freq="D")
    history = pd.Series(np.arange(14, dtype=float), index=days)
    cutoff = days[9]  # values 0..9 visible; target day is days[10]
    assert forecast.forecast_mean7(history, cutoff, start=None) == pytest.approx(np.mean([3, 4, 5, 6, 7, 8, 9]))
    assert forecast.forecast_same_weekday(history, cutoff, start=None) == 3.0
    assert (cutoff + pd.Timedelta(days=1)).dayofweek == days[3].dayofweek


def test_history_until_truncates_at_both_ends():
    days = pd.date_range("2023-12-25", periods=20, freq="D")
    history = pd.Series(1.0, index=days)
    part = forecast.history_until(history, "2024-01-05")
    assert part.index[0] == pd.Timestamp("2024-01-01")
    assert part.index[-1] == pd.Timestamp("2024-01-05")


def test_history_until_rejects_gaps_and_future_cutoffs():
    days = pd.date_range("2025-03-01", periods=10, freq="D")
    history = pd.Series(1.0, index=days)
    with pytest.raises(ValueError):
        forecast.history_until(history, "2025-03-20", start=None)
    with pytest.raises(ValueError):
        forecast.history_until(history.drop(days[4]), days[8], start=None)
    with_nan = history.copy()
    with_nan.iloc[3] = np.nan
    with pytest.raises(ValueError):
        forecast.history_until(with_nan, days[8], start=None)


def test_model_forecasts_are_bounded(history):
    cutoff = history.index[199]
    for name in ["ets", "sarima"]:
        assert 0 <= FORECASTERS[name](history, cutoff) <= 24


def test_metrics():
    actual = pd.Series([1.0, 2.0, 3.0])
    predicted = pd.Series([1.0, 4.0, 0.0])
    assert forecast.mae(actual, predicted) == pytest.approx(5 / 3)
    assert forecast.rmse(actual, predicted) == pytest.approx(np.sqrt(13 / 3))


def test_test_window_is_the_final_n_days():
    history = pd.Series(1.0, index=pd.date_range("2026-01-01", "2026-08-28", freq="D"))
    first, last, selection_cutoff = forecast.test_window(history, 90)
    assert last == pd.Timestamp("2026-08-28")
    assert first == pd.Timestamp("2026-05-31")
    assert len(pd.date_range(first, last)) == 90
    assert selection_cutoff == pd.Timestamp("2026-05-30")


def test_mae_difference_interval_sign():
    days = pd.date_range("2025-01-01", periods=90, freq="D")
    actual = pd.Series(5.0, index=days)
    good, bad = actual + 0.1, actual + 2.0
    mean, low, high = forecast.mae_difference_interval(actual, good, bad)
    assert mean == pytest.approx(-1.9)
    assert low <= mean <= high < 0
