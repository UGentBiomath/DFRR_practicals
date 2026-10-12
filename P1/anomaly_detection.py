"""
Help functions and methods for data inspection and anomaly detection.

Course of Digitalisation for Resource Recovery
@Saba Daneshgar - BIOMATH, UGent


"""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from statsmodels.tsa.stattools import adfuller


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _as_series(data: pd.Series | pd.DataFrame, column: str = "Value") -> pd.Series:
    """Return a numeric Series from a Series or single-variable DataFrame."""
    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, pd.DataFrame):
        if column in data.columns:
            series = data[column].copy()
        elif data.shape[1] == 1:
            series = data.iloc[:, 0].copy()
        else:
            raise ValueError(
                f"DataFrame must contain column {column!r} or have exactly one column."
            )
    else:
        raise TypeError("data must be a pandas Series or DataFrame")

    return pd.to_numeric(series, errors="coerce")


def _validate_window(window: int | None) -> int | None:
    if window in (None, 0):
        return None
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be None, 0, or an integer >= 2")
    return window


def _result_table(
    values: pd.Series,
    score: pd.Series,
    lower_bound: pd.Series,
    upper_bound: pd.Series,
    is_anomaly: pd.Series,
) -> pd.DataFrame:
    """Build the common detector output table."""
    return pd.DataFrame(
        {
            "value": values,
            "score": score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "is_anomaly": is_anomaly.fillna(False).astype(bool),
        },
        index=values.index,
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_sensor_data(
    variable: str,
    directory: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load a sensor CSV and optionally select a date range."""
    path = Path(directory) / f"{variable}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sensor file not found: {path}")

    data = pd.read_csv(path, sep=r";|,", engine="python")
    if "Time" not in data.columns:
        raise ValueError("CSV file must contain a 'Time' column")

    data["Time"] = pd.to_datetime(data["Time"])
    data = data.set_index("Time").sort_index()
    data = data[~data.index.duplicated(keep="first")]

    if start_date is not None or end_date is not None:
        data = data.loc[start_date:end_date]
    return data


def load_validated_anomalies(
    variable: str,
    directory: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load a validated anomaly dataset with Time and Value columns."""
    path = Path(directory) / f"{variable}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Validated dataset not found: {path}")

    data = pd.read_csv(path, sep=";", names=["Time", "Value"])
    data["Time"] = pd.to_datetime(data["Time"].astype(str).str.strip())
    data = data.set_index("Time")
    data = data[~data.index.duplicated(keep="first")].sort_index()

    if start_date is not None or end_date is not None:
        data = data.loc[start_date:end_date]
    return data


# -----------------------------------------------------------------------------
# Univariate methods
# -----------------------------------------------------------------------------

def iqr_detector(
    data: pd.Series | pd.DataFrame,
    window: int | None = None,
    threshold: float = 1.5,
    column: str = "Value",
) -> pd.DataFrame:
    """Detect anomalies using global or rolling interquartile-range bounds.

    Parameters
    ----------
    data : Series or DataFrame
        Univariate measurements.
    window : int or None, default None
        Rolling-window size. If None (or 0), global quartiles are used.
    threshold : float, default 1.5
        IQR multiplier used to define the lower and upper bounds.
    column : str, default "Value"
        Column to use when ``data`` is a DataFrame.
    """
    values = _as_series(data, column)
    window = _validate_window(window)
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    if window is None:
        q1 = pd.Series(values.quantile(0.25), index=values.index)
        q3 = pd.Series(values.quantile(0.75), index=values.index)
    else:
        min_periods = max(2, window // 2)
        q1 = values.rolling(window, min_periods=min_periods, center=True).quantile(0.25)
        q3 = values.rolling(window, min_periods=min_periods, center=True).quantile(0.75)

    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    mask = ((values < lower) | (values > upper)) & lower.notna() & upper.notna()

    distance = pd.concat([(q1 - values) / iqr, (values - q3) / iqr], axis=1).max(axis=1)
    score = distance.replace([np.inf, -np.inf], np.nan)
    return _result_table(values, score, lower, upper, mask)


def zscore_detector(
    data: pd.Series | pd.DataFrame,
    window: int | None = None,
    threshold: float = 3.0,
    column: str = "Value",
) -> pd.DataFrame:
    """Detect anomalies using global or rolling z-scores."""
    values = _as_series(data, column)
    window = _validate_window(window)
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    if window is None:
        mean = pd.Series(values.mean(), index=values.index)
        std = pd.Series(values.std(), index=values.index)
    else:
        min_periods = max(2, window // 2)
        mean = values.rolling(window, min_periods=min_periods, center=True).mean()
        std = values.rolling(window, min_periods=min_periods, center=True).std()

    safe_std = std.replace(0, np.nan)
    zscore = (values - mean) / safe_std
    lower = mean - threshold * std
    upper = mean + threshold * std
    mask = zscore.abs() >= threshold
    return _result_table(values, zscore, lower, upper, mask)


def modified_zscore_detector(
    data: pd.Series | pd.DataFrame,
    window: int | None = None,
    threshold: float = 3.5,
    column: str = "Value",
) -> pd.DataFrame:
    """Detect anomalies using the median and median absolute deviation (MAD)."""
    values = _as_series(data, column)
    window = _validate_window(window)
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    if window is None:
        median = pd.Series(values.median(), index=values.index)
        mad_value = np.median(np.abs(values.dropna() - values.median()))
        mad = pd.Series(mad_value, index=values.index)
    else:
        min_periods = max(2, window // 2)
        median = values.rolling(window, min_periods=min_periods, center=True).median()
        absolute_deviation = (values - median).abs()
        mad = absolute_deviation.rolling(
            window, min_periods=min_periods, center=True
        ).median()

    safe_mad = mad.replace(0, np.nan)
    score = 0.6745 * (values - median) / safe_mad

    zero_mad = mad.eq(0) & median.notna()
    score = score.mask(zero_mad & values.ne(median), np.inf)
    score = score.mask(zero_mad & values.eq(median), 0.0)


    bound_width = (threshold / 0.6745) * mad
    lower = median - bound_width
    upper = median + bound_width
    mask = score.abs() >= threshold
    return _result_table(values, score, lower, upper, mask)


def ewma_detector(
    data: pd.Series | pd.DataFrame,
    halflife: float = 10,
    threshold: float = 3.0,
    column: str = "Value",
) -> pd.DataFrame:
    """Detect anomalies using exponentially weighted mean and standard deviation."""
    values = _as_series(data, column)
    if halflife <= 0:
        raise ValueError("halflife must be > 0")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    mean = values.ewm(halflife=halflife, adjust=False).mean()
    std = values.ewm(halflife=halflife, adjust=False).std()
    safe_std = std.replace(0, np.nan)
    score = (values - mean) / safe_std
    lower = mean - threshold * std
    upper = mean + threshold * std
    mask = score.abs() >= threshold
    return _result_table(values, score, lower, upper, mask)


def shewhart_detector(data, window=None, threshold=3.0, column="Value",
                      statistic="mean", rolling=False, limits="fixed", limit_window=100):
    """Shewhart-style monitoring of individual values or subgroup statistics.

    With window=None, monitor individual observations (mean or median).
    With window>=2, monitor subgroup means, medians, or ranges. `rolling=True`
    makes overlapping trailing subgroups; otherwise groups do not overlap.

    `limits='fixed'` estimates one reference center/spread from all available
    chart statistics. `limits='rolling'` estimates them from the preceding
    `limit_window` chart statistics, excluding the current statistic to avoid
    self-influence. The first `limit_window` chart statistics have undefined
    limits and are not flagged.
    """
    values = _as_series(data, column)
    if statistic not in {"mean", "median", "range"}:
        raise ValueError("statistic must be 'mean', 'median', or 'range'")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    if window is not None and (not isinstance(window, int) or isinstance(window, bool) or window < 2):
        raise ValueError("window must be None or an integer >= 2")
    if statistic == "range" and window is None:
        raise ValueError("Range charts require window >= 2")
    if not isinstance(rolling, bool):
        raise ValueError("rolling must be True or False")
    if rolling and window is None:
        raise ValueError("rolling=True requires window >= 2")
    if limits not in {"fixed", "rolling"}:
        raise ValueError("limits must be 'fixed' or 'rolling'")
    if limits == "rolling" and (not isinstance(limit_window, int) or isinstance(limit_window, bool) or limit_window < 2):
        raise ValueError("limit_window must be an integer >= 2")

    if window is None:
        chart_values = values.copy()
    elif rolling:
        groups = values.rolling(window=window, min_periods=window)
        if statistic == "mean":
            chart_values = groups.mean()
        elif statistic == "median":
            chart_values = groups.median()
        else:
            chart_values = groups.max() - groups.min()
        chart_values.name = values.name
    else:
        points, timestamps = [], []
        for start in range(0, len(values) - window + 1, window):
            subgroup = values.iloc[start:start + window]
            if subgroup.isna().any():
                continue
            if statistic == "mean":
                point = subgroup.mean()
            elif statistic == "median":
                point = subgroup.median()
            else:
                point = subgroup.max() - subgroup.min()
            points.append(point)
            timestamps.append(values.index[start + window - 1])
        chart_values = pd.Series(points, index=timestamps, name=values.name, dtype=float)

    valid = chart_values.dropna()
    if valid.empty:
        raise ValueError("No valid observations or complete subgroups available")

    if limits == "fixed":
        center_value = float(valid.median() if statistic == "median" else valid.mean())
        spread_value = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
        center = pd.Series(center_value, index=chart_values.index)
        spread = pd.Series(spread_value if np.isfinite(spread_value) else 0.0,
                           index=chart_values.index)
    else:
        history = chart_values.shift(1).rolling(window=limit_window, min_periods=limit_window)
        center = history.median() if statistic == "median" else history.mean()
        spread = history.std(ddof=1)

    lower = center - threshold * spread
    if statistic == "range":
        lower = lower.clip(lower=0.0)
    upper = center + threshold * spread
    score = (chart_values - center) / spread.replace(0, np.nan)
    zero_spread = spread.eq(0) & center.notna()
    score = score.mask(zero_spread & chart_values.ne(center), np.inf)
    score = score.mask(zero_spread & chart_values.eq(center), 0.0)
    mask = ((chart_values < lower) | (chart_values > upper)).fillna(False)
    return _result_table(chart_values, score, lower, upper, mask)



def rate_of_change_detector(data, periods: int = 1, lower: float = -np.inf,
                            upper: float = np.inf, column="Value"):
    """Flag percentage rate-of-change values outside specified bounds."""
    values = _as_series(data, column)
    if periods < 1:
        raise ValueError("periods must be >= 1")
    roc = 100 * values.pct_change(periods=periods, fill_method=None)
    mask = ((roc < lower) | (roc > upper)).fillna(False)
    lower_s = pd.Series(lower, index=values.index, dtype=float)
    upper_s = pd.Series(upper, index=values.index, dtype=float)
    return _result_table(values, roc, lower_s, upper_s, mask)


def constant_value_detector(data, length: int, tolerance: float = 0.0, column="Value"):
    """Flag measurements belonging to flat runs of at least ``length`` points."""
    values = _as_series(data, column)
    if length < 2:
        raise ValueError("length must be >= 2")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    changed = values.diff().abs().gt(tolerance).fillna(True)
    group = changed.cumsum()
    run_length = values.groupby(group).transform("size")
    mask = run_length.ge(length)
    score = run_length.astype(float)
    nan_bound = pd.Series(np.nan, index=values.index)
    return _result_table(values, score, nan_bound, nan_bound, mask)


def physical_range_detector(data, lower: float, upper: float, column="Value"):
    """Flag values outside physically plausible lower and upper limits."""
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    values = _as_series(data, column)
    lower_s = pd.Series(lower, index=values.index, dtype=float)
    upper_s = pd.Series(upper, index=values.index, dtype=float)
    mask = ((values < lower_s) | (values > upper_s)).fillna(False)
    score = pd.concat([lower_s - values, values - upper_s], axis=1).max(axis=1).clip(lower=0)
    return _result_table(values, score, lower_s, upper_s, mask)


# -----------------------------------------------------------------------------
# ARIMA/ARMA
# -----------------------------------------------------------------------------

def _regularize_datetime_frequency(series: pd.Series) -> pd.Series:
    """Attach/infer frequency metadata for a regular DatetimeIndex.

    If pandas cannot infer a frequency, fall back to a RangeIndex for model
    fitting. This avoids statsmodels' date-index warning without changing the
    observations; callers can still map predictions back to the original index.
    """
    series = series.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    if series.index.freq is not None:
        return series

    inferred = pd.infer_freq(series.index)
    if inferred is not None:
        series.index = pd.DatetimeIndex(series.index, freq=inferred)
    return series


def arima_detector(data, order=(0, 0, 1), threshold=3.0, column="Value"):
    """Fit an ARIMA/ARMA model and detect unusually large residuals."""
    from statsmodels.tsa.arima.model import ARIMA

    values = _as_series(data, column).dropna().copy()
    original_index = values.index

    model_values = _regularize_datetime_frequency(values)


    if isinstance(model_values.index, pd.DatetimeIndex) and model_values.index.freq is None:
        model_values = pd.Series(model_values.to_numpy(), index=pd.RangeIndex(len(model_values)))

    fitted = ARIMA(model_values, order=order).fit()

    prediction_values = np.asarray(
        fitted.predict(start=0, end=len(model_values) - 1), dtype=float
    )
    prediction = pd.Series(prediction_values, index=original_index)
    residual = values - prediction

    residual_std = float(residual.std(ddof=1))
    if not np.isfinite(residual_std):
        residual_std = 0.0

    lower = prediction - threshold * residual_std
    upper = prediction + threshold * residual_std
    mask = residual.abs().gt(threshold * residual_std)

    return _result_table(values, residual, lower, upper, mask), fitted


def arima_order_search(data, max_p=5, max_q=5, column="Value"):
    """Search ARMA(p,q) orders using AIC."""
    from statsmodels.tsa.arima.model import ARIMA
    values = _as_series(data, column).dropna().copy()
    values = _regularize_datetime_frequency(values)
    if isinstance(values.index, pd.DatetimeIndex) and values.index.freq is None:
        values = pd.Series(values.to_numpy(), index=pd.RangeIndex(len(values)))
    return find_best_order_for_model(values, ARIMA, max_p=max_p, max_q=max_q, model_name="ARMA")



# -----------------------------------------------------------------------------
# Data inspection
# -----------------------------------------------------------------------------

def data_inspection(points: pd.DataFrame, variable: str) -> pd.Series:
    """Plot a time series, density and box plot; return summary statistics."""
    points[variable].plot(figsize=(12, 3), title=variable)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    sns.kdeplot(ax=axes[0], data=points[variable], fill=True)
    axes[0].set_title("Density plot")
    sns.boxplot(ax=axes[1], x=points[variable])
    axes[1].set_title("Box plot")
    plt.tight_layout()

    statistics = points[variable].describe()
    statistics.loc["skewness"] = points[variable].skew()
    return statistics




# -----------------------------------------------------------------------------
# Time-series utilities
# -----------------------------------------------------------------------------

def resample_data(points: pd.Series | pd.DataFrame, frequency: str, method: str):
    """Resample time-indexed data using a selected aggregation/fill method."""
    data = points[~points.index.duplicated(keep="first")]
    methods = {
        "last": lambda x: x.resample(frequency).last(),
        "bfill": lambda x: x.resample(frequency).bfill(),
        "ffill": lambda x: x.resample(frequency).ffill(),
        "mean": lambda x: x.resample(frequency).mean(),
        "min": lambda x: x.resample(frequency).min(),
        "max": lambda x: x.resample(frequency).max(),
        "sum": lambda x: x.resample(frequency).sum(),
    }
    if method not in methods:
        raise ValueError(f"method must be one of {list(methods)}")
    return methods[method](data)


def test_stationarity(data, column: str | None = None, significance: float = 0.05):
    """Classify a time series as stationary/non-stationary using the ADF test."""
    series = _as_series(data, column or "Value") if isinstance(data, pd.DataFrame) else data
    p_value = adfuller(pd.Series(series).dropna(), autolag="AIC")[1]
    return "Stationary" if p_value <= significance else "Non-Stationary"


def difference_series(data, order: int = 1, column: str = "Value") -> pd.Series:
    """Difference a time series by ``order`` periods."""
    values = _as_series(data, column)
    return values.diff(order)


def get_order_sets(n: int, n_per_set: int) -> list[list[int]]:
    values = list(range(n))
    return [values[i : i + n_per_set] for i in range(0, n, n_per_set)]


def find_aic_for_model(data, p, q, model: Callable, model_name="model"):
    """Fit one (p,0,q) model and return its AIC and order."""
    try:
        results = model(data, order=(p, 0, q)).fit()
        return results.aic, (p, 0, q)
    except Exception as exc:
        warnings.warn(f"{model_name}({p},0,{q}) failed: {exc}", RuntimeWarning)
        return np.nan, (p, 0, q)


def find_best_order_for_model(data, model: Callable, max_p: int, max_q: int, model_name="model"):
    """Search (p,0,q) orders and return the order with the minimum AIC."""
    combinations = [(p, q) for p in range(max_p + 1) for q in range(max_q + 1)]
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(find_aic_for_model)(data, p, q, model, model_name)
        for p, q in combinations
    )
    results_df = pd.DataFrame(results, columns=["aic", "order"]).dropna(subset=["aic"])
    if results_df.empty:
        raise RuntimeError("No candidate model could be fitted")
    best = results_df.loc[results_df["aic"].idxmin()]
    return best["aic"], best["order"], results_df


def find_anomalies(squared_errors, threshold_multiplier: float = 3.0):
    """Flag squared errors above mean + multiplier*standard deviation."""
    errors = np.asarray(squared_errors)
    threshold = errors.mean() + threshold_multiplier * errors.std()
    predictions = (errors >= threshold).astype(int)
    return predictions, threshold


def absolute_to_relative(
    data: pd.DataFrame,
    time_data: str = "index",
    unit: str = "d",
    inplace: bool = False,
    save_absolute: bool = True,
    decimals: int = 5,
):
    """Convert datetime values to elapsed time from the first observation."""
    if unit not in {"sec", "min", "hr", "d"}:
        raise ValueError("unit must be 'sec', 'min', 'hr', or 'd'")

    result = data if inplace else data.copy()
    times = result.index if time_data == "index" else pd.to_datetime(result[time_data])
    elapsed_seconds = (times - times[0]).total_seconds()
    divisors = {"sec": 1, "min": 60, "hr": 3600, "d": 86400}
    relative = np.asarray(elapsed_seconds) / divisors[unit]
    relative = np.round(relative, decimals)

    if time_data == "index":
        if save_absolute:
            result["time_abs"] = times
        result.index = relative
    else:
        if save_absolute:
            result["time_abs"] = times
        result[time_data] = relative

    return None if inplace else result



# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

PLOT_COLORS = {
    "measurement": "#2C5282", "limit": "#909090",
    "anomaly": "#D62728", "validated": "#009E73",
    "score": "#596579", "reference": "#909090",
}

def _plot_style(fig, title=None, height=None):
    """Apply the same Plotly styling to every interactive figure."""
    fig.update_layout(template="plotly_white", title=title, hovermode="closest",
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#F3F6FA",
                      font={"size": 13}, legend={"orientation": "v", "x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(255,255,255,0.85)"},
                      margin={"l": 60, "r": 190, "t": 65, "b": 65},
                      **({"height": height} if height else {}))
    fig.update_xaxes(showgrid=True, gridcolor="#DFE6EF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#DFE6EF", zeroline=False)
    return fig


def _detected_marker():
    return {"symbol": "x", "size": 10, "color": PLOT_COLORS["anomaly"], "line": {"width": 0}}


def _validated_marker():
    return {"symbol": "circle-open", "size": 13,
            "color": PLOT_COLORS["validated"], "line": {"width": 2.5}}

def plot_detector_result(
    result: pd.DataFrame,
    title: str = "Anomaly detection",
    show_score: bool = True,
    show_limits: bool = True,
    values: pd.Series | None = None,
):
    """Plot measurements, detected anomalies, optional bounds and scores.

    Parameters
    ----------
    result : pd.DataFrame
        Detector output containing 'is_anomaly' and optionally
        'value', 'score', 'lower_bound', and 'upper_bound'.
    title : str
        Figure title.
    show_score : bool
        Whether to plot the anomaly score in a second subplot.
    show_limits : bool
        Whether to display detector bounds, when available.
    values : pd.Series, optional
        External measurements to visualize instead of result['value'].
        Useful for multivariate detectors such as LOF and PCA.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if "is_anomaly" not in result.columns:
        raise ValueError("result must contain 'is_anomaly'.")

    if values is None:
        if "value" not in result.columns:
            raise ValueError(
                "result must contain 'value' unless values is provided."
            )
        values = result["value"]
    elif not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series.")

    flags = (
        result["is_anomaly"]
        .reindex(values.index, fill_value=False)
        .fillna(False)
        .astype(bool)
    )

    has_score = (
        show_score
        and "score" in result.columns
        and result["score"].notna().any()
    )

    rows = 2 if has_score else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=values.index,
            y=values,
            mode="lines",
            name="Measurements",
            line={"color": "#2C5282", "width": 1.6},
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=values.index[flags],
            y=values.loc[flags],
            mode="markers",
            marker=_detected_marker(),
            name="Detected anomaly",
        ),
        row=1, col=1,
    )


    if show_limits:
        if {"lower_bound", "upper_bound"}.issubset(result.columns):
            if result["lower_bound"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=result.index,
                        y=result["lower_bound"],
                        mode="lines",
                        name="Lower limit",
                        line={
                            "dash": "dash",
                            "color": "#909090",
                            "width": 1.5,
                        },
                    ),
                    row=1, col=1,
                )

            if result["upper_bound"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=result.index,
                        y=result["upper_bound"],
                        mode="lines",
                        name="Upper limit",
                        line={
                            "dash": "dash",
                            "color": "#909090",
                            "width": 1.5,
                        },
                    ),
                    row=1, col=1,
                )

    if has_score:
        fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["score"],
                mode="lines",
                name="Anomaly score",
                line={"color": "#6B7280"},
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Score", row=2, col=1)

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=rows, col=1)

    _plot_style(fig, title, 650 if rows == 2 else 430)

    return fig

# def plot_detector_result(result: pd.DataFrame, title: str = "Anomaly detection", show_score: bool = True, show_limits: bool = True, values: pd.Series | None = None,):
#     """Plot measurements, detector bounds/anomalies, and optionally anomaly score.
#
#     Works with the harmonized detector output. Returns a Plotly figure so the
#     caller can further customize it if desired.
#     """
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#
#     required = {"value", "is_anomaly"}
#     missing = required.difference(result.columns)
#     if missing:
#         raise ValueError(f"result is missing required columns: {sorted(missing)}")
#
#     has_score = show_score and "score" in result.columns and result["score"].notna().any()
#     rows = 2 if has_score else 1
#     fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08)
#
#     fig.add_trace(
#         go.Scatter(x=result.index, y=result["value"], mode="lines", name="Measurements", line={"color": "#2C5282", "width": 1.6}),
#         row=1, col=1,
#     )
#
#     anomalies = result[result["is_anomaly"]]
#     fig.add_trace(
#         go.Scatter(
#             x=anomalies.index,
#             y=anomalies["value"],
#             mode="markers",
#             marker=_detected_marker(),
#             name="Detected anomaly",
#         ),
#         row=1, col=1,
#     )
#
#     if show_limits:
#         if {"lower_bound", "upper_bound"}.issubset(result.columns):
#             if result["lower_bound"].notna().any():
#                 fig.add_trace(
#                     go.Scatter(x=result.index, y=result["lower_bound"], mode="lines", name="Lower limit", line={"dash": "dash", "color": "#909090", "width": 1.5}),
#                     row=1, col=1,
#                 )
#             if result["upper_bound"].notna().any():
#                 fig.add_trace(
#                     go.Scatter(x=result.index, y=result["upper_bound"], mode="lines", name="Upper limit", line={"dash": "dash", "color": "#909090", "width": 1.5}),
#                     row=1, col=1,
#                 )
#
#     if has_score:
#         fig.add_trace(
#             go.Scatter(x=result.index, y=result["score"], mode="lines", name="Anomaly score", line={"color": "#6B7280"}),
#             row=2, col=1,
#         )
#         fig.update_yaxes(title_text="Score", row=2, col=1)
#
#     fig.update_yaxes(title_text="Value", row=1, col=1)
#     fig.update_xaxes(title_text="Time", row=rows, col=1)
#     _plot_style(fig, title, 650 if rows == 2 else 430)
#     return fig


def plot_detection_summary(data, results: dict[str, pd.DataFrame], column: str = "Value"):
    """Plot overall anomalies and a binary detection raster for several methods."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    values = _as_series(data, column)
    masks = pd.DataFrame({name: result["is_anomaly"].reindex(values.index, fill_value=False)
                          for name, result in results.items()}, index=values.index)
    overall = masks.any(axis=1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Overall anomaly decision", "Detections by method"),
                        vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name="Measurements", line={"color": PLOT_COLORS["measurement"]}), row=1, col=1)
    fig.add_trace(go.Scatter(x=values.index[overall], y=values[overall], mode="markers",
                             marker=_detected_marker(), name="Detected anomaly"), row=1, col=1)

    method_colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD",
                     "#FF7F0E", "#17BECF", "#8C564B", "#E377C2",
                     "#7F7F7F", "#BCBD22"]
    for i, name in enumerate(masks.columns):
        idx = masks.index[masks[name]]
        fig.add_trace(go.Scatter(x=idx, y=np.full(len(idx), i), mode="markers", name=name,
                                 marker={"size": 7, "color": method_colors[i % len(method_colors)], "symbol": "x", "line": {"width": 0}}), row=2, col=1)

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(masks.columns))),
                     ticktext=list(masks.columns), row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    _plot_style(fig, height=650)
    return fig, masks, overall



# -----------------------------------------------------------------------------
# Multivariate methods
# -----------------------------------------------------------------------------

def correlation_matrix(data: pd.DataFrame):
    """Return the numeric correlation matrix for a DataFrame."""
    return data.corr(numeric_only=True)


def plot_correlation_matrix(data: pd.DataFrame):
    """Plot an interactive correlation heatmap."""
    import plotly.express as px
    corr = correlation_matrix(data)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Blues")
    return _plot_style(fig, "Correlation matrix", 550)


def lof_detector(data: pd.DataFrame, n_neighbors=20, contamination="auto"):
    """Apply Local Outlier Factor and return scores plus anomaly mask."""
    from sklearn.neighbors import LocalOutlierFactor
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = model.fit_predict(data)
    result = pd.DataFrame(index=data.index)
    result["score"] = -model.negative_outlier_factor_  # larger = more unusual
    result["is_anomaly"] = labels == -1
    result["upper_bound"] = float(-model.offset_)  # LOF score decision boundary
    return result


def plot_lof_neighborhood_scores(data: pd.DataFrame, result: pd.DataFrame,
                                 x_variable, y_variable, radius_scale=1000):
    """Plot LOF scores as open circles with area proportional to anomaly strength.

    Larger LOF scores mean more unusual observations. Circle areas are normalized
    between 0 and radius_scale.
    """
    import plotly.graph_objects as go

    scores = result["score"].reindex(data.index).astype(float)
    finite = np.isfinite(scores.to_numpy())
    if not finite.any():
        raise ValueError("No finite LOF scores to plot.")
    low, high = scores[finite].min(), scores[finite].max()
    strength = ((scores - low) / (high - low)) if high > low else scores * 0
    diameters = np.sqrt(strength.clip(lower=0)) * np.sqrt(radius_scale) * 1.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_variable], y=data[y_variable], mode="markers",
        name="Measurements", marker={"size": 5, "color": PLOT_COLORS["measurement"], "opacity": 0.65}))
    fig.add_trace(go.Scatter(
        x=data[x_variable], y=data[y_variable], mode="markers",
        name="LOF score (circle area)",
        marker={"symbol": "circle-open", "size": diameters.fillna(0).to_numpy(),
                "color": PLOT_COLORS["anomaly"], "line": {"width": 1.5}},
        customdata=scores.to_numpy(), hovertemplate="LOF score: %{customdata:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title=x_variable, yaxis_title=y_variable)
    return _plot_style(fig, "Local Outlier Factor: score magnitude", 520)


def plot_multivariate_anomalies(data: pd.DataFrame, result: pd.DataFrame,
                                x_variable, y_variable):
    """Scatter two variables and highlight observations flagged as anomalies."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[x_variable], y=data[y_variable], mode="markers", name="Measurements", marker={"size": 5, "color": PLOT_COLORS["measurement"], "opacity": 0.45}))
    mask = result["is_anomaly"].reindex(data.index, fill_value=False)
    fig.add_trace(go.Scatter(x=data.loc[mask, x_variable], y=data.loc[mask, y_variable],
                             mode="markers", marker=_detected_marker(), name="Detected anomaly"))
    fig.update_layout(xaxis_title=x_variable, yaxis_title=y_variable)
    _plot_style(fig, "Multivariate anomaly detection", 520)
    return fig


def pca_spe_detector(data: pd.DataFrame, explained_variance=0.90, confidence=0.98):
    """PCA reconstruction-error (SPE) detector with a KDE-derived control limit.

    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA(explained_variance)
    scores = pca.fit_transform(scaled)
    reconstructed = pca.inverse_transform(scores)
    spe = np.sqrt(np.sum((reconstructed - scaled) ** 2, axis=1)) / scaled.shape[1]

    x = spe.reshape(-1, 1)
    span = float(np.max(x) - np.min(x))
    bandwidth = max(span / 100, np.finfo(float).eps)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(x)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 1000)[:, None]
    density = np.exp(kde.score_samples(grid))
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    limit = float(grid[np.searchsorted(cdf, confidence), 0])

    result = pd.DataFrame({"score": spe, "is_anomaly": spe > limit}, index=data.index)
    info = {
        "n_components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "control_limit": limit,
        "pca": pca,
        "scaler": scaler,
        "scores": pd.DataFrame(scores, index=data.index,
                               columns=[f"PC{i+1}" for i in range(scores.shape[1])]),
        "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        "reconstructed_scaled": pd.DataFrame(reconstructed, index=data.index, columns=data.columns),
    }
    return result, info


def plot_score_result(result: pd.DataFrame, title="Anomaly score", threshold=None):
    """Plot a generic anomaly score and optional threshold."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["score"], mode="lines", name="Anomaly score", line={"color": PLOT_COLORS["score"]}))
    if threshold is None and "upper_bound" in result.columns:
        threshold = result["upper_bound"].dropna().iloc[0] if result["upper_bound"].notna().any() else None
    if threshold is not None:
        fig.add_hline(y=float(threshold), line_dash="dash", line_color=PLOT_COLORS["limit"], annotation_text="Threshold")
    anomalies = result[result["is_anomaly"]]
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["score"], mode="markers",
                             marker=_detected_marker(), name="Detected anomaly"))
    fig.update_layout(xaxis_title="Time", yaxis_title="Score")
    _plot_style(fig, title, 430)
    return fig

def plot_pca_scores(info: dict, result: pd.DataFrame, pc_x=1, pc_y=2):
    """Show the PCA score space with detected anomalies marked by red crosses.

    Parameters
    ----------
    pc_x, pc_y : int
        Principal-component numbers (starting at 1) for the two axes.
    """
    import plotly.graph_objects as go
    scores = info["scores"]
    xname, yname = f"PC{pc_x}", f"PC{pc_y}"
    if xname not in scores or yname not in scores:
        raise ValueError("Selected PCs are unavailable; choose component numbers present in info['scores'].")
    mask = result["is_anomaly"].reindex(scores.index, fill_value=False).astype(bool)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores[xname], y=scores[yname], mode="markers",
                             name="Measurements", marker={"size": 5, "color": PLOT_COLORS["measurement"], "opacity": 0.45}))
    fig.add_trace(go.Scatter(x=scores.loc[mask, xname], y=scores.loc[mask, yname],
                             mode="markers", name="Detected anomaly", marker=_detected_marker()))
    fig.update_layout(xaxis_title=xname, yaxis_title=yname)
    return _plot_style(fig, "PCA score plot", 520)


def plot_pca_explained_variance(info: dict):
    """Plot individual PCA variance bars and cumulative variance on a secondary axis."""
    import plotly.graph_objects as go
    ratio = np.asarray(info["explained_variance_ratio"], dtype=float)
    labels = [f"PC{i+1}" for i in range(len(ratio))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=ratio, name="Individual variance",
                         marker_color=PLOT_COLORS["measurement"], yaxis="y"))
    fig.add_trace(go.Scatter(x=labels, y=np.cumsum(ratio), mode="lines",
                             line={"shape": "hv", "color": PLOT_COLORS["validated"], "width": 2.5},
                             name="Cumulative variance", yaxis="y2"))
    fig.update_layout(
        xaxis_title="Principal component",
        yaxis={"title": "Individual explained variance", "tickformat": ".0%", "rangemode": "tozero"},
        yaxis2={"title": "Cumulative explained variance", "tickformat": ".0%",
                "range": [0, 1.05], "overlaying": "y", "side": "right", "showgrid": False},
    )
    return _plot_style(fig, "PCA explained variance", 430)


# -----------------------------------------------------------------------------
# Data validation
# -----------------------------------------------------------------------------

def build_ground_truth(index, anomaly_times, tolerance=None):
    """Build a Boolean anomaly label Series aligned to a measurement index.

    Parameters
    ----------
    index : pandas.Index
        Index of the measurements to evaluate.
    anomaly_times : array-like or pandas object
        Timestamps known to be anomalous.
    tolerance : str or pandas.Timedelta, optional
        If supplied, each validated timestamp is matched to the nearest
        measurement timestamp within this tolerance. Without a tolerance,
        only exact timestamp matches are labelled.
    """
    idx = pd.DatetimeIndex(index)
    times = pd.DatetimeIndex(getattr(anomaly_times, "index", anomaly_times))
    truth = pd.Series(False, index=idx, name="is_anomaly")

    if tolerance is None:
        truth.loc[truth.index.intersection(times)] = True
        return truth

    tol = pd.Timedelta(tolerance)
    positions = idx.get_indexer(times, method="nearest", tolerance=tol)
    positions = positions[positions >= 0]
    if len(positions):
        truth.iloc[np.unique(positions)] = True
    return truth


def evaluate_detector(y_true, y_pred) -> dict:
    """Return binary anomaly-detection metrics in a compact dictionary."""
    true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(y_pred).astype(int).ravel()
    cm = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "confusion_matrix": cm,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, zero_division=0),
        "recall": recall_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
    }


def show_evaluation(y_true, y_pred) -> dict:
    """Plot a confusion matrix, print a report, and return evaluation metrics."""
    result = evaluate_detector(y_true, y_pred)
    cm = result["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print(classification_report(y_true, y_pred, zero_division=0))
    return result


def evaluate_results(y_true, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate several harmonized detector results against known labels."""
    truth = pd.Series(y_true).astype(bool)
    rows = []
    for name, result in results.items():
        pred = result["is_anomaly"].reindex(truth.index, fill_value=False)
        metrics = evaluate_detector(truth, pred)
        rows.append({
            "method": name,
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TN": metrics["TN"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
        })
    return pd.DataFrame(rows).set_index("method")


def plot_validation_result(result: pd.DataFrame, y_true, title="Detector validation"):
    """Plot detector predictions together with known/validated anomalies."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    truth = pd.Series(y_true, index=getattr(y_true, "index", result.index)).astype(bool)
    truth = truth.reindex(result.index, fill_value=False)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(title, "Predicted vs known anomaly labels"),
                        vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=result.index, y=result["value"], mode="lines", name="Measurements", line={"color": "#2C5282", "width": 1.6}), row=1, col=1)

    if {"lower_bound", "upper_bound"}.issubset(result.columns):
        for column, label in (("lower_bound", "Lower limit"), ("upper_bound", "Upper limit")):
            bounds = result[column].replace([np.inf, -np.inf], np.nan)
            if bounds.notna().any():
                fig.add_trace(
                    go.Scatter(x=result.index, y=bounds, mode="lines",
                               name=label, line={"dash": "dash", "color": "#909090", "width": 1.5}),
                    row=1, col=1,
                )
    pred = result["is_anomaly"].fillna(False)
    fig.add_trace(go.Scatter(x=result.index[truth], y=result.loc[truth, "value"], mode="markers",
                             marker=_validated_marker(), name="Known anomaly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index[pred], y=result.loc[pred, "value"], mode="markers",
                             marker=_detected_marker(), name="Detected anomaly"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=pred.astype(int), mode="lines", name="Detected anomaly label", line={"color": "#D62728"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=truth.astype(int), mode="lines", name="Known anomaly label", line={"color": "#009E73"}), row=2, col=1)
    fig.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["Normal", "Anomaly"], row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    _plot_style(fig, height=650)
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix"):
    """Plot a 2x2 confusion matrix with counts."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    true = pd.Series(y_true).astype(bool)
    pred = pd.Series(y_pred).reindex(true.index, fill_value=False).astype(bool)

    cm = confusion_matrix(true, pred, labels=[False, True])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Anomaly"],
    )
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    disp.ax_.set_title(title)
    disp.figure_.set_facecolor("#FFFFFF")
    disp.ax_.set_facecolor("#F3F6FA")
    return disp.ax_


def roc_analysis(y_true, scores):
    """Return false-positive rate, true-positive rate and ROC AUC."""
    from sklearn.metrics import roc_curve, auc

    true = pd.Series(y_true).astype(bool)
    score = pd.Series(scores).reindex(true.index)
    valid = score.notna() & true.notna()

    fpr, tpr, thresholds = roc_curve(true.loc[valid].astype(int), score.loc[valid])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc=None, title="ROC curve"):
    """Plot an ROC curve and the no-skill diagonal."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F3F6FA")
    ax.grid(alpha=0.2)
    label = f"ROC (AUC = {roc_auc:.3f})" if roc_auc is not None else "ROC"
    ax.plot(fpr, tpr, label=label, color=PLOT_COLORS["measurement"], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=PLOT_COLORS["reference"], label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.subplots_adjust(right=0.76)
    return ax
