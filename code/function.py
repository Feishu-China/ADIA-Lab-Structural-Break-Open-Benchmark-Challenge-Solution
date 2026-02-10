import os
import time
import json
import joblib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from scipy.stats import f, ks_2samp, levene, ttest_ind, mannwhitneyu

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


EPS = 1e-8


def ensure_picture_dir_in_parent(folder_name: str = "picture") -> str:
    """
    Ensure the output directory exists at "../picture" and return its path.

    Args:
        folder_name: Folder name to create under the parent directory.

    Returns:
        The directory path (e.g., "../picture").
    """
    out_dir = os.path.join("..", folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def set_global_plot_style() -> None:
    """
    Set global matplotlib plotting style.

    Raises:
        ValueError: If rcParams update fails.
    """
    plt.rcParams.update({
        "figure.figsize": (12, 9),
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 16,
        "lines.linewidth": 1.1,
    })


def transform_series(series: np.ndarray, window: int = 5) -> dict[str, np.ndarray]:
    """
    Generate multiple deterministic transformations of a univariate time series.

    Args:
        series: One-dimensional numeric time series.
        window: Window size used for rolling statistics.

    Returns:
        A dictionary mapping transformation names to transformed series.
    """
    original = series.copy()

    mu = np.mean(series)
    sigma = np.std(series)
    zscore = (series - mu) / (sigma + 1e-8)

    cumulative_sum = np.cumsum(series)
    dense_rank = pd.Series(series).rank(method="dense").to_numpy()
    abs_value = np.abs(series)

    moving_avg = pd.Series(series).rolling(window=window, min_periods=1).mean().to_numpy()

    moving_std = pd.Series(series).rolling(window=window, min_periods=1).std().to_numpy()
    moving_std = np.nan_to_num(moving_std, nan=0.0)

    return {
        "Original Series": original,
        "Z-score Normalized": zscore,
        "Cumulative Summed": cumulative_sum,
        "Dense Ranked": dense_rank,
        "Absolute Value": abs_value,
        f"Moving Average (window={window})": moving_avg,
        f"Moving Std (window={window})": moving_std,
    }


def plot_true_breakpoint_transformations(
    X_train: pd.DataFrame,
    y_train,
    target_id: int = 10000,
    moving_window: int = 5,
    label_col: str = "structural_breakpoint",
    save_stem: str = "true_breakpoint_transformations",
    dpi: int = 450,
) -> tuple[str, str]:
    """
    Plot seven transformations for one sample and save as PNG and PDF.

    Args:
        X_train: Long-format feature data accessible via ``X_train.loc[target_id]``,
            containing columns ``value`` and ``period`` and indexed by ``time`` level.
        y_train: Labels as Series or DataFrame accessible via ``y_train.loc[target_id]``.
        target_id: Target id to visualize.
        moving_window: Window size for rolling transformations.
        label_col: Label column name if ``y_train`` is a DataFrame.
        save_stem: Output filename stem (no extension).
        dpi: DPI for saved PNG.

    Returns:
        A tuple (png_path, pdf_path).

    Raises:
        ValueError: If the extracted time series is empty.
    """
    set_global_plot_style()

    id_x_data = X_train.loc[target_id].sort_index(level="time")
    time_series = id_x_data["value"].to_numpy()
    time_axis = id_x_data.index.get_level_values("time").to_numpy()
    period_array = id_x_data["period"].to_numpy()

    if len(time_series) == 0:
        raise ValueError("Empty time series for the given target_id.")

    if isinstance(y_train, pd.DataFrame):
        is_breakpoint = bool(y_train.loc[target_id, label_col])
    else:
        is_breakpoint = bool(y_train.loc[target_id])

    breakpoint_time = None
    for i in range(1, len(period_array)):
        if period_array[i - 1] == 0 and period_array[i] == 1:
            breakpoint_time = time_axis[i]
            break

    transformed = transform_series(time_series, window=moving_window)
    names = list(transformed.keys())

    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)

    for idx, name in enumerate(names):
        axes[idx].plot(time_axis, transformed[name])
        axes[idx].grid(alpha=0.3, linestyle="--")

        if is_breakpoint and breakpoint_time is not None:
            axes[idx].axvline(
                x=breakpoint_time,
                color="red",
                linestyle="--",
                alpha=0.8,
                linewidth=2.8,
            )

        if idx == 0:
            suffix = " (True Breakpoint)" if is_breakpoint else " (No Breakpoint)"
            axes[idx].set_title(f"{name}{suffix}", pad=10)
        else:
            axes[idx].set_title(name, pad=10)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()

    out_dir = ensure_picture_dir_in_parent("picture")
    png_path = os.path.join(out_dir, save_stem + ".png")
    pdf_path = os.path.join(out_dir, save_stem + ".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    return png_path, pdf_path


def set_report_plot_style() -> None:
    """Apply a consistent matplotlib style for report figures.
    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    plt.rcParams.update({
        "figure.figsize": (12, 9),
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 16,
        "lines.linewidth": 2.8,
    })


def ensure_long_columns(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has required columns: id, time, value, period.
    
    Args:
        X_train: Input DataFrame that either has (id, time) as MultiIndex + (value, period) as columns,
                 or already contains all four required columns as regular columns.
    
    Returns:
        DataFrame with explicit columns: id, time, value, period (no MultiIndex).
    
    Raises:
        ValueError: If required columns (value, period) are missing initially, or if the final
                    DataFrame lacks any of the four required columns (id, time, value, period).
    """
    need_cols = {"value", "period"}
    miss = need_cols - set(X_train.columns)
    if miss:
        raise ValueError(f"X_train missing columns {miss}; required at least {need_cols}")

    if ("id" in X_train.index.names) and ("time" in X_train.index.names):
        X = X_train.reset_index()
    else:
        X = X_train.copy()

    required = {"id", "time", "value", "period"}
    miss2 = required - set(X.columns)
    if miss2:
        raise ValueError(f"Cannot construct required columns {required}. Missing {miss2}. Current columns={list(X.columns)}")

    return X


def ensure_y_series(y, index: pd.Index, label_col: str = "structural_breakpoint") -> pd.Series:
    """
    Convert labels to a 1D pandas Series aligned to the given id-level index.
    
    Args:
        y: Input labels in dict/DataFrame/np.ndarray/Series format.
        index: Target pandas Index to align the output Series with (id-level index).
        label_col: Name of the column containing labels if y is a DataFrame (default: "structural_breakpoint").
    
    Returns:
        pd.Series: 1D Series with int dtype (0/1) aligned to the input index.
    
    Raises:
        ValueError: If y is a DataFrame without the specified label_col (and has >1 columns).
        ValueError: If y is a 2D ndarray with more than 1 column.
        ValueError: If label alignment fails (missing ids in y for the input index).
    """
    if isinstance(y, dict):
        y = pd.Series(y)

    if isinstance(y, pd.DataFrame):
        if label_col not in y.columns:
            if y.shape[1] == 1:
                label_col = y.columns[0]
            else:
                raise ValueError(f"y is a DataFrame but column '{label_col}' not found.")
        y = y[label_col]

    if isinstance(y, np.ndarray):
        if y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(f"y ndarray has shape={y.shape}; expected (n,) or (n,1).")
            y = y[:, 0]
        y = pd.Series(y)

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    y = y.reindex(index)
    if y.isna().any():
        bad = y[y.isna()].index[:10].tolist()
        raise ValueError(f"Label alignment failed. Missing ids (examples): {bad}")

    return y.astype(int)


def compute_8_scores(v0: np.ndarray, v1: np.ndarray) -> dict[str, float]:
    """
    Compute eight statistical scores to compare two data segments (period 0 vs period 1).
    
    Args:
        v0: Numpy array of values from period 0.
        v1: Numpy array of values from period 1.
    
    Returns:
        dict[str, float]: Dictionary mapping statistical method names to their scalar scores.
    
    Raises:
        None
    """
    s: dict[str, float] = {}

    if len(v0) >= 2 and len(v1) >= 2:
        t_stat, _ = ttest_ind(v0, v1, equal_var=False)
        s["Welch's t-test"] = float(abs(t_stat))
    else:
        s["Welch's t-test"] = 0.0

    if len(v0) >= 1 and len(v1) >= 1:
        u_stat, _ = mannwhitneyu(v0, v1, alternative="two-sided")
        s["Mann–Whitney U"] = float(u_stat)
    else:
        s["Mann–Whitney U"] = 0.0

    if len(v0) >= 1 and len(v1) >= 1:
        ks_stat, _ = ks_2samp(v0, v1)
        s["KS test"] = float(ks_stat)
    else:
        s["KS test"] = 0.0

    s["Mean Difference"] = float(abs(np.mean(v1) - np.mean(v0))) if (len(v0) and len(v1)) else 0.0
    s["Variance Difference"] = float(abs(np.var(v1, ddof=1) - np.var(v0, ddof=1))) if (len(v0) >= 2 and len(v1) >= 2) else 0.0
    s["Std Difference"] = float(abs(np.std(v1, ddof=1) - np.std(v0, ddof=1))) if (len(v0) >= 2 and len(v1) >= 2) else 0.0
    s["Median Difference"] = float(abs(np.median(v1) - np.median(v0))) if (len(v0) and len(v1)) else 0.0

    if len(v0) and len(v1):
        iqr0 = float(np.percentile(v0, 75) - np.percentile(v0, 25))
        iqr1 = float(np.percentile(v1, 75) - np.percentile(v1, 25))
        s["IQR Difference"] = float(abs(iqr1 - iqr0))
    else:
        s["IQR Difference"] = 0.0

    return s


def plot_8_tests_train_roc_4x2(
    X_train: pd.DataFrame,
    y_train,
    save_stem: str = "statistical_tests_train_roc_4x2",
    dpi: int = 450,
    label_col: str = "structural_breakpoint",
) -> tuple[str, str]:
    """
    Generate ROC curves for eight statistical test–based scores on the training set.

    Args:
        X_train: Long-format training data containing at least columns ``id``, ``time``, ``period``, and ``value``.
        y_train: Training labels indexed by ``id`` or convertible to a label series aligned with ``X_train``.
        save_stem: Output filename stem (no extension). Files are saved into ``../picture``.
        dpi: Resolution (dots per inch) for the saved PNG figure.
        label_col: Name of the column in ``y_train`` that contains the binary label.

    Returns:
        A tuple ``(png_path, pdf_path)`` corresponding to the saved figure files.

    Raises:
        ValueError: If no valid samples are available for ROC computation.
    """
    X = ensure_long_columns(X_train)
    ids = pd.Index(X["id"].unique(), name="id")
    y = ensure_y_series(y_train, ids, label_col=label_col)

    method_names = [
        "Welch's t-test",
        "Mann–Whitney U",
        "KS test",
        "Mean Difference",
        "Variance Difference",
        "Std Difference",
        "Median Difference",
        "IQR Difference",
    ]

    scores_dict: dict[str, list[float]] = {m: [] for m in method_names}
    labels: list[int] = []

    for _id, df in X.groupby("id"):
        df = df.sort_values("time")
        v0 = df[df["period"] == 0]["value"].values
        v1 = df[df["period"] == 1]["value"].values

        if len(v0) < 2 or len(v1) < 2 or _id not in y.index:
            continue

        s = compute_8_scores(v0, v1)
        labels.append(int(y.loc[_id]))
        for m in method_names:
            scores_dict[m].append(s[m])

    labels_arr = np.asarray(labels, dtype=int)
    if len(labels_arr) == 0:
        raise ValueError("No valid samples for ROC plotting.")

    set_report_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()

    for i, m in enumerate(method_names):
        ax = axes[i]
        score = np.asarray(scores_dict[m], dtype=float)

        fpr, tpr, _ = roc_curve(labels_arr, score)
        auc_val = roc_auc_score(labels_arr, score)

        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "--", alpha=0.7)
        ax.set_title(m, fontsize=30, pad=10)
        ax.grid(True, alpha=0.35)

        ax.text(
            0.95, 0.05, f"AUC = {auc_val:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=22,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7, ec="none")
        )

    for j in range(len(method_names), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    out_dir = ensure_picture_dir_in_parent("picture")
    png_path = os.path.join(out_dir, save_stem + ".png")
    pdf_path = os.path.join(out_dir, save_stem + ".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    return png_path, pdf_path


def safe_std_series(x: pd.Series) -> float:
    """
    Compute standard deviation of a pandas Series with EPS fallback to avoid zero-division.
    
    Args:
        x: Input pandas Series for standard deviation calculation.
    
    Returns:
        float: Standard deviation of the Series, or EPS if std is zero.
    
    Raises:
        None
    """
    s = float(x.std())
    return s if s > 0 else EPS


def safe_div(a: float, b: float) -> float:
    """
    Safely compute division (a / b) with EPS added to denominator to avoid zero-division.
    
    Args:
        a: Numerator of the division operation.
        b: Denominator of the division operation.
    
    Returns:
        float: Result of a / (b + EPS).
    
    Raises:
        None
    """
    return float(a) / (float(b) + EPS)


def safe_iqr(arr: np.ndarray) -> float:
    """
    Compute interquartile range (IQR) of a numpy array with EPS fallback to avoid zero values.
    
    Args:
        arr: Input numpy array for IQR calculation.
    
    Returns:
        float: IQR (75th percentile - 25th percentile) of the array; 0.0 if array is empty;
               EPS if IQR is zero.
    
    Raises:
        None
    """
    if arr is None or len(arr) == 0:
        return 0.0
    q75, q25 = np.percentile(arr, [75, 25])
    v = float(q75 - q25)
    return v if v > 0 else EPS


def mad(arr: np.ndarray) -> float:
    """
    Compute median absolute deviation (MAD) with EPS fallback to avoid zero values.
    
    Args:
        arr: Input numpy array for MAD calculation.
    
    Returns:
        float: MAD of the array; 0.0 if array is empty; EPS if MAD is zero.
    
    Raises:
        None
    """
    if arr is None or len(arr) == 0:
        return 0.0
    m = np.median(arr)
    v = float(np.median(np.abs(arr - m)))
    return v if v > 0 else EPS


def plot_oof_roc(y_true, y_score, title: str = "OOF ROC") -> float:
    """
    Create and return the image save directory (parent folder's "picture" directory).
    
    Args:
        None
    
    Returns:
        str: Absolute path to the ../picture directory.
    
    Raises:
        OSError: If directory creation fails due to permission issues or invalid path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(12, 9))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    return float(auc)


class FirstFeatureGenerator:
    """s1_: global distribution statistics (no period split)."""

    def generate(self, X: pd.DataFrame, prefix: str = "s1_") -> pd.DataFrame:
        """
        Generate global distribution features from long-format DataFrame grouped by ID.
        
        Args:
            X: Long-format DataFrame containing at least "id" and "value" columns.
            prefix: Prefix to add to all generated feature names (default: "s1_").
        
        Returns:
            pd.DataFrame: Feature DataFrame indexed by "id", with generated statistical features.
        
        Raises:
            KeyError: If "id" or "value" columns are missing from input DataFrame.
        """
        agg = X.groupby("id")["value"].agg(
            mean="mean",
            median="median",
            max="max",
            min="min",
            std=lambda x: safe_std_series(x),
            skew="skew",
        )
        agg["mean_norm"] = agg["mean"] / (agg["std"] + EPS)
        agg["median_norm"] = agg["median"] / (agg["std"] + EPS)
        agg.columns = [prefix + c for c in agg.columns]
        return agg


class SecondFeatureGenerator:
    """s2_: temporal-shape features aggregated to one row per id."""

    def generate(self, X: pd.DataFrame, prefix: str = "s2_") -> pd.DataFrame:
        """
        Generate temporal-shape features from long-format time-series DataFrame.
        
        Args:
            X: Long-format DataFrame containing at least "id", "time", and "value" columns.
            prefix: Prefix to add to all generated feature names (default: "s2_").
        
        Returns:
            pd.DataFrame: Feature DataFrame indexed by "id", with all generated temporal-shape features.
        
        Raises:
            KeyError: If "id", "time", or "value" columns are missing from input DataFrame.
            ValueError: If input DataFrame is empty or has invalid numerical values.
        """
        X = X.sort_values(["id", "time"]).copy()
        g = X.groupby("id")["value"]

        X["z"] = (X["value"] - g.transform("mean")) / g.transform(lambda s: safe_std_series(s))
        X["abs"] = X["value"].abs()
        X["rank"] = g.rank(pct=True)

        X["roll_mean"] = g.rolling(16, min_periods=1).mean().reset_index(0, drop=True)
        X["roll_std"] = g.rolling(16, min_periods=1).std().reset_index(0, drop=True).fillna(0)

        feats = ["value", "z", "abs", "rank", "roll_mean", "roll_std"]
        agg = X.groupby("id")[feats].agg(["mean", "median", "max", "min", "std", "skew"])
        agg.columns = ["_".join(c) for c in agg.columns]

        for f_name in feats:
            denom = agg[f"{f_name}_std"].replace(0, EPS)
            agg[f"{f_name}_mean_norm"] = agg[f"{f_name}_mean"] / denom

        agg.columns = [prefix + c for c in agg.columns]
        return agg


class BreakpointCompareFeatureGenerator:
    """s3_: explicit pre/post (period 0 vs 1) comparison features."""

    def _one_id(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute pre/post comparison features for a single ID's DataFrame.
        
        Args:
            df: DataFrame for a single ID containing "period" and "value" columns.
        
        Returns:
            pd.Series: 20-dimensional feature vector for the ID (0.0 if insufficient data).
        
        Raises:
            KeyError: If "period" or "value" columns are missing from input DataFrame.
        """
        a = df.loc[df.period == 0, "value"].values
        b = df.loc[df.period == 1, "value"].values
        if len(a) < 2 or len(b) < 2:
            return pd.Series([0.0] * 20)

        pre_mean, post_mean = float(np.mean(a)), float(np.mean(b))
        pre_median, post_median = float(np.median(a)), float(np.median(b))
        pre_std, post_std = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))

        pre_iqr, post_iqr = float(safe_iqr(a)), float(safe_iqr(b))
        pre_mad, post_mad = float(mad(a)), float(mad(b))

        pre_p10, pre_p90 = np.percentile(a, [10, 90])
        post_p10, post_p90 = np.percentile(b, [10, 90])

        mean_diff = post_mean - pre_mean
        mean_absdiff = abs(mean_diff)
        mean_ratio = safe_div(post_mean, pre_mean)

        median_diff = post_median - pre_median
        median_absdiff = abs(median_diff)
        median_ratio = safe_div(post_median, pre_median)

        std_ratio = safe_div(post_std, pre_std)
        iqr_ratio = safe_div(post_iqr, pre_iqr)
        mad_ratio = safe_div(post_mad, pre_mad)

        pre_spread = float(pre_p90 - pre_p10)
        post_spread = float(post_p90 - post_p10)
        spread_ratio = safe_div(post_spread, pre_spread)

        return pd.Series([
            pre_mean, post_mean, mean_diff, mean_absdiff, mean_ratio,
            pre_median, post_median, median_diff, median_absdiff, median_ratio,
            pre_std, post_std, std_ratio,
            pre_iqr, post_iqr, iqr_ratio,
            pre_mad, post_mad, mad_ratio,
            spread_ratio
        ])

    def generate(self, X: pd.DataFrame, prefix: str = "s3_") -> pd.DataFrame:
        """
        Generate pre/post comparison features from long-format DataFrame grouped by ID.
        
        Args:
            X: Long-format DataFrame containing at least "id", "period", and "value" columns.
            prefix: Prefix to add to all generated feature names (default: "s3_").
        
        Returns:
            pd.DataFrame: Feature DataFrame indexed by "id", with all pre/post comparison features.
        
        Raises:
            KeyError: If "id", "period", or "value" columns are missing from input DataFrame.
            ValueError: If input DataFrame is empty or has invalid numerical values.
        """
        out = X.groupby("id").apply(self._one_id)
        out.columns = [
            "pre_mean", "post_mean", "mean_diff", "mean_absdiff", "mean_ratio",
            "pre_median", "post_median", "median_diff", "median_absdiff", "median_ratio",
            "pre_std", "post_std", "std_ratio",
            "pre_iqr", "post_iqr", "iqr_ratio",
            "pre_mad", "post_mad", "mad_ratio",
            "spread_ratio",
        ]
        out.columns = [prefix + c for c in out.columns]
        return out


class FourthFeatureGenerator:
    """s4_: statistical hypothesis test features (F/Levene/KS/Welch's t-test)."""

    def _test(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute statistical hypothesis test features for a single ID's DataFrame.
        
        Args:
            df: DataFrame for a single ID containing "period" and "value" columns.
        
        Returns:
            pd.Series: 8-dimensional feature vector (0.0 if insufficient data).
        
        Raises:
            KeyError: If "period" or "value" columns are missing from input DataFrame.
        """
        a = df.loc[df.period == 0, "value"].values
        b = df.loc[df.period == 1, "value"].values
        if len(a) < 2 or len(b) < 2:
            return pd.Series([0.0] * 8)

        v1 = np.var(a, ddof=1)
        v2 = max(np.var(b, ddof=1), EPS)
        f_stat = float(v1 / v2)
        f_p = float(2 * min(
            f.cdf(f_stat, len(a) - 1, len(b) - 1),
            1 - f.cdf(f_stat, len(a) - 1, len(b) - 1)
        ))

        lev_stat, lev_p = levene(a, b)
        ks_stat, ks_p = ks_2samp(a, b)
        t_stat, t_p = ttest_ind(a, b, equal_var=False)

        return pd.Series([float(f_stat), float(f_p),
                          float(lev_stat), float(lev_p),
                          float(ks_stat), float(ks_p),
                          float(t_stat), float(t_p)])

    def generate(self, X: pd.DataFrame, prefix: str = "s4_") -> pd.DataFrame:
        """
        Generate statistical hypothesis test features from long-format DataFrame grouped by ID.
        
        Args:
            X: Long-format DataFrame containing at least "id", "period", and "value" columns.
            prefix: Prefix to add to all generated feature names (default: "s4_").
        
        Returns:
            pd.DataFrame: Feature DataFrame indexed by "id", with all hypothesis test features.
        
        Raises:
            KeyError: If "id", "period", or "value" columns are missing from input DataFrame.
            ValueError: If input DataFrame is empty or has invalid numerical values.
        """
        out = X.groupby("id").apply(self._test)
        out.columns = ["f_stat", "f_p", "lev_stat", "lev_p", "ks_stat", "ks_p", "t_stat", "t_p"]
        out.columns = [prefix + c for c in out.columns]
        return out


def generate_features_split(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate two distinct feature blocks for base and meta models from long-format input data.
    
    Args:
        X: Long-format DataFrame containing at least "id", "period", "time", and "value" columns.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - X1: Compact global statistics features (s1_) for TabPFN base model (cleaned of inf/NaN).
            - XS: Richer combined features (s2_ + s3_ + s4_) for meta model (cleaned of inf/NaN).
            Both DataFrames are indexed by "id" with overlapping indices only.
    
    Raises:
        KeyError: If required columns ("id", "period", "time", "value") are missing from input DataFrame.
        ValueError: If no overlapping indices exist between X1 and XS feature sets.
    """
    X = X.copy()
    s1 = FirstFeatureGenerator().generate(X, "s1_")
    s2 = SecondFeatureGenerator().generate(X, "s2_")
    s3 = BreakpointCompareFeatureGenerator().generate(X, "s3_")
    s4 = FourthFeatureGenerator().generate(X, "s4_")

    X1 = s1.replace([np.inf, -np.inf], 0).fillna(0)
    XS = s2.join([s3, s4], how="outer").replace([np.inf, -np.inf], 0).fillna(0)

    common = X1.index.intersection(XS.index)
    return X1.loc[common], XS.loc[common]


def ensure_picture_dirs(folder_name: str = "picture") -> Tuple[str, str]:
    """Create picture folders in current dir and parent dir.

    Args:
        folder_name: Folder name to create in current directory and parent directory.

    Returns:
        A tuple (current_dir_path, parent_dir_path).

    Raises:
        ValueError: If folder_name is empty.
    """
    if not folder_name:
        raise ValueError("folder_name must be non-empty")
    cur_dir = os.path.join(".", folder_name)
    par_dir = os.path.join("..", folder_name)
    os.makedirs(cur_dir, exist_ok=True)
    os.makedirs(par_dir, exist_ok=True)
    return cur_dir, par_dir


def save_and_copy_figure(fig: plt.Figure, png_path: str, pdf_path: str, dpi: int = 450) -> Tuple[str, str, str, str]:
    """Save a figure to current folder and copy to parent folder.

    Args:
        fig: Matplotlib figure.
        png_path: Target PNG path in current folder.
        pdf_path: Target PDF path in current folder.
        dpi: DPI for PNG.

    Returns:
        A tuple (cur_png, cur_pdf, parent_png, parent_pdf).

    Raises:
        ValueError: If paths do not end with required extensions.
    """
    if not png_path.endswith(".png"):
        raise ValueError("png_path must end with .png")
    if not pdf_path.endswith(".pdf"):
        raise ValueError("pdf_path must end with .pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    parent_png = os.path.join("..", os.path.relpath(png_path, "."))
    parent_pdf = os.path.join("..", os.path.relpath(pdf_path, "."))

    os.makedirs(os.path.dirname(parent_png), exist_ok=True)
    os.makedirs(os.path.dirname(parent_pdf), exist_ok=True)

    shutil.copy2(png_path, parent_png)
    shutil.copy2(pdf_path, parent_pdf)

    return png_path, pdf_path, parent_png, parent_pdf


class TabPFNThenBoostPipeline:
    """
    Two-stage classification pipeline with TabPFN base model and meta model.
    """
    def __init__(self, tabpfn_model: Any, meta_model: Any, meta_cols: List[str]):
        """
        Initialize the pipeline with pre-trained models and meta feature columns.
        
        Args:
            tabpfn_model: Trained TabPFN base model with predict_proba method.
            meta_model: Trained meta model with predict_proba method.
            meta_cols: List of column names for meta model feature input.
        
        Returns:
            None
        """
        self.tabpfn_model = tabpfn_model
        self.meta_model = meta_model
        self.meta_cols = meta_cols

    def predict_proba(self, X_long: pd.DataFrame) -> np.ndarray:
        """
        Generate binary classification probabilities for long-format input data.
        
        Args:
            X_long: Long-format DataFrame with required columns for feature generation.
        
        Returns:
            np.ndarray: 2D array of probabilities (class 0, class 1) with shape (n_samples, 2).
        """
        X1, XS = generate_features_split(X_long)
        X1 = X1.replace([np.inf, -np.inf], 0).fillna(0)
        XS = XS.replace([np.inf, -np.inf], 0).fillna(0)

        x1 = self.tabpfn_model.predict_proba(X1)[:, 1]

        X_meta = XS.copy()
        X_meta["x1"] = x1

        X_meta = X_meta.reindex(columns=self.meta_cols, fill_value=0.0)
        proba = get_score_proba_1(self.meta_model, X_meta)
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X_long: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary classification predictions (0/1) based on probability threshold.
        
        Args:
            X_long: Long-format DataFrame with required columns for feature generation.
            threshold: Probability threshold for class 1 prediction (0.0 <= threshold <= 1.0).
        
        Returns:
            np.ndarray: 1D array of binary predictions (0/1) with shape (n_samples,).
        """
        return (self.predict_proba(X_long)[:, 1] >= threshold).astype(int)


def plot_roc_bundle(
    y_true: np.ndarray,
    score_map: Dict[str, np.ndarray],
    save_stem: str,
    dpi: int = 450,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Plot ROC curves for multiple models and save PNG/PDF to ./picture and copy to ../picture.

    Args:
        y_true: True labels (0/1), shape (n,).
        score_map: Mapping model_name -> score array (n,).
        save_stem: File stem without extension.
        dpi: DPI for PNG.
        title: Optional title.

    Returns:
        A dict with keys:
            "auc": {name: auc},
            "cur_png", "cur_pdf", "parent_png", "parent_pdf".

    Raises:
        ValueError: If y_true and any score length mismatch.
    """
    y_true = np.asarray(y_true).ravel()
    for k, v in score_map.items():
        v = np.asarray(v).ravel()
        if len(v) != len(y_true):
            raise ValueError(f"Length mismatch for {k}: len(score)={len(v)} vs len(y)={len(y_true)}")

    set_report_plot_style()
    fig = plt.figure(figsize=(12, 9))

    auc_map: Dict[str, float] = {}
    for name, score in score_map.items():
        score = np.asarray(score).ravel()
        fpr, tpr, _ = roc_curve(y_true, score)
        auc_val = float(roc_auc_score(y_true, score))
        auc_map[name] = auc_val
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "--", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend(loc="lower right")
    if title:
        plt.title(title)
    plt.tight_layout()

    cur_pic, _ = ensure_picture_dirs("picture")
    cur_png = os.path.join(cur_pic, f"{save_stem}.png")
    cur_pdf = os.path.join(cur_pic, f"{save_stem}.pdf")

    cur_png, cur_pdf, parent_png, parent_pdf = save_and_copy_figure(fig, cur_png, cur_pdf, dpi=dpi)
    plt.show()

    return {
        "auc": auc_map,
        "cur_png": cur_png,
        "cur_pdf": cur_pdf,
        "parent_png": parent_png,
        "parent_pdf": parent_pdf,
    }


def get_score_proba_1(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get positive-class probability scores.

    Args:
        model: Fitted model supporting predict_proba or decision_function.
        X: Features.

    Returns:
        A score array (n,).

    Raises:
        ValueError: If neither predict_proba nor decision_function is available.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Model must support predict_proba or decision_function")


def oof_predict_scores(model: Any, X: pd.DataFrame, y: pd.Series, n_folds: int = 10, seed: int = 42) -> np.ndarray:
    """Compute strict OOF scores.

    Args:
        model: Sklearn-compatible estimator (cloneable).
        X: Features.
        y: Labels (0/1) aligned with X.
        n_folds: Number of folds.
        seed: Random seed.

    Returns:
        OOF scores array (n,).

    Raises:
        ValueError: If n_folds < 2.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    for tr, va in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = get_score_proba_1(m, X.iloc[va])

    return oof


def fit_predict_scores(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Fit on full training data and predict scores on test.

    Args:
        model: Estimator.
        X_train: Train features.
        y_train: Train labels.
        X_test: Test features.

    Returns:
        Test score array (n_test,).
    """
    m = clone(model)
    m.fit(X_train, y_train)
    return get_score_proba_1(m, X_test)


def build_meta_models(seed: int = 42) -> Dict[str, Any]:
    """Build three boosting models: LightGBM, XGBoost, CatBoost.

    Args:
        seed: Random seed.

    Returns:
        Dict mapping names to estimators.

    Raises:
        ImportError: If xgboost or catboost is not installed.
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed but is required for the 3-model setup")
    if not _HAS_CATBOOST:
        raise ImportError("catboost is not installed but is required for the 3-model setup")

    models: Dict[str, Any] = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=1400,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.75,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=1600,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=2500,
            learning_rate=0.02,
            depth=8,
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=seed,
        ),
    }
    return models


def tabpfn_oof_scores(X: pd.DataFrame, y: pd.Series, n_folds: int = 10, seed: int = 42) -> np.ndarray:
    """Compute strict OOF probabilities for TabPFN.

    Args:
        X: Features (recommended compact X1 from s1_).
        y: Labels aligned with X.
        n_folds: Number of folds.
        seed: Random seed.

    Returns:
        OOF probability scores (n,).
    """
    from tabpfn import TabPFNClassifier

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    print(f"[TabPFN OOF] start | folds={n_folds} | n={len(X)}")
    t_all = time.time()

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        print(f"[TabPFN OOF] fold {fold:02d}/{n_folds} | tr={len(tr)} va={len(va)} ...", flush=True)

        m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]

        dt = time.time() - t0
        print(f"[TabPFN OOF] fold {fold:02d}/{n_folds} done | {dt/60:.1f} min", flush=True)

    print(f"[TabPFN OOF] all done | total={(time.time()-t_all)/60:.1f} min", flush=True)
    return oof


def tabpfn_fit_predict_scores(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Fit TabPFN on full train and predict probabilities on test.

    Args:
        X_train: Train features (compact X1).
        y_train: Train labels.
        X_test: Test features (compact X1).

    Returns:
        Probability scores (n_test,).
    """
    from tabpfn import TabPFNClassifier

    m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    m.fit(X_train, y_train)
    return m.predict_proba(X_test)[:, 1]


def run_tabpfn_x1_plus_s234_three_models(
    X_train: pd.DataFrame,
    y_train: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    n_folds: int = 10,
    seed: int = 42,
    dpi: int = 450,
    save_stem_train: str = "roc_train_oof_meta_3boost",
    save_stem_test: str = "roc_test_meta_3boost",
) -> Dict[str, Any]:
    """Run TabPFN-on-s1 then train three meta boosting models on (s2+s3+s4+x1) and plot ROC on train/test.

    Args:
        X_train: Long-format train data for feature generators.
        y_train: Train labels.
        X_test: Long-format test data.
        y_test: Test labels.
        n_folds: Number of folds for strict OOF.
        seed: Random seed.
        dpi: DPI for PNG.
        save_stem_train: File stem for train OOF ROC.
        save_stem_test: File stem for test ROC.

    Returns:
        Dict containing:
            "train_auc", "test_auc",
            "train_plot_paths", "test_plot_paths",
            "base_oof", "base_test",
            "meta_oof_scores", "meta_test_scores",
            "X_meta_train", "X_meta_test", "y_train_aligned", "y_test_aligned".
    """
    t0 = time.time()

    X1_tr, XS_tr = generate_features_split(X_train)
    y_tr = ensure_y_series(y_train, X1_tr.index)

    X1_te, XS_te = generate_features_split(X_test)
    y_te = ensure_y_series(y_test, X1_te.index)

    X1_tr = X1_tr.replace([np.inf, -np.inf], 0).fillna(0)
    XS_tr = XS_tr.replace([np.inf, -np.inf], 0).fillna(0)
    X1_te = X1_te.replace([np.inf, -np.inf], 0).fillna(0)
    XS_te = XS_te.replace([np.inf, -np.inf], 0).fillna(0)

    y_tr = ensure_y_series(y_train, X1_tr.index)
    y_te = ensure_y_series(y_test, X1_te.index)

    X1_cols = X1_tr.columns.intersection(X1_te.columns)
    XS_cols = XS_tr.columns.intersection(XS_te.columns)

    X1_tr, X1_te = X1_tr[X1_cols], X1_te[X1_cols]
    XS_tr, XS_te = XS_tr[XS_cols], XS_te[XS_cols]

    base_oof = tabpfn_oof_scores(X1_tr, y_tr, n_folds=n_folds, seed=seed)
    base_test = tabpfn_fit_predict_scores(X1_tr, y_tr, X1_te)

    X_meta_train = XS_tr.copy()
    X_meta_train["x1"] = base_oof

    X_meta_test = XS_te.copy()
    X_meta_test["x1"] = base_test

    models = build_meta_models(seed=seed)

    meta_oof_scores: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        meta_oof_scores[name] = oof_predict_scores(model, X_meta_train, y_tr, n_folds=n_folds, seed=seed)

    train_plot = plot_roc_bundle(
        y_true=y_tr.values,
        score_map=meta_oof_scores,
        save_stem=save_stem_train,
        dpi=dpi,
        title=None,
    )

    meta_test_scores: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        meta_test_scores[name] = fit_predict_scores(model, X_meta_train, y_tr, X_meta_test)

    test_plot = plot_roc_bundle(
        y_true=y_te.values,
        score_map=meta_test_scores,
        save_stem=save_stem_test,
        dpi=dpi,
        title=None,
    )

    train_auc = {k: float(roc_auc_score(y_tr.values, v)) for k, v in meta_oof_scores.items()}
    test_auc = {k: float(roc_auc_score(y_te.values, v)) for k, v in meta_test_scores.items()}

    runtime_sec = float(time.time() - t0)

    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_plot_paths": train_plot,
        "test_plot_paths": test_plot,
        "base_oof": base_oof,
        "base_test": base_test,
        "meta_oof_scores": meta_oof_scores,
        "meta_test_scores": meta_test_scores,
        "X_meta_train": X_meta_train,
        "X_meta_test": X_meta_test,
        "y_train_aligned": y_tr,
        "y_test_aligned": y_te,
        "runtime_sec": runtime_sec,
    }


def train_and_save_tabpfn_plus_3boost_pipelines(
    X_train: pd.DataFrame,
    y_train: Any,
    seed: int = 42,
    save_dir: str = "models_tabpfn_plus_boost",
) -> Dict[str, str]:
    """
    Train TabPFN base model + meta boost models, wrap into pipelines and save to disk.
    
    Args:
        X_train: Long-format training DataFrame with required columns for feature generation.
        y_train: Training target (any format compatible with ensure_y_series).
        seed: Random seed for reproducibility (default: 42).
        save_dir: Directory path to save trained pipelines (default: "models_tabpfn_plus_boost").
    
    Returns:
        Dict[str, str]: Mapping of model names to their saved pickle file paths.
    
    Raises:
        OSError: If save directory cannot be created/written to.
        KeyError: If required columns are missing from X_train.
        ValueError: If X_train/y_train have mismatched indices after processing.
    """
    os.makedirs(save_dir, exist_ok=True)

    X1_tr, XS_tr = generate_features_split(X_train)
    y_tr = ensure_y_series(y_train, X1_tr.index)

    X1_tr = X1_tr.replace([np.inf, -np.inf], 0).fillna(0)
    XS_tr = XS_tr.replace([np.inf, -np.inf], 0).fillna(0)

    X_meta_train = XS_tr.copy()

    from tabpfn import TabPFNClassifier
    tab_model = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    x1_train = tabpfn_oof_scores(X1_tr, y_tr, n_folds=10, seed=seed)
    tab_model.fit(X1_tr, y_tr)

    X_meta_train["x1"] = x1_train

    meta_cols = list(X_meta_train.columns)

    models = build_meta_models(seed=seed)

    saved: Dict[str, str] = {}
    for name, model in models.items():
        m = clone(model)
        m.fit(X_meta_train[meta_cols], y_tr)

        pipe = TabPFNThenBoostPipeline(
            tabpfn_model=tab_model,
            meta_model=m,
            meta_cols=meta_cols,
        )

        pkl_path = os.path.join(save_dir, f"TabPFN_plus_{name}.pkl")
        joblib.dump(pipe, pkl_path)
        saved[name] = pkl_path

    return saved


def save_pipelines_from_run_output(
    out: dict,
    X_train: pd.DataFrame,
    y_train,
    seed: int = 42,
    save_dir: str = "models_tabpfn_plus_boost",
) -> Dict[str, str]:
    """
    Save TabPFN+3Boost pipelines by reusing the already-computed run output `out`.

    Args:
        out: Output dict returned by `run_tabpfn_x1_plus_s234_three_models`.
        X_train: Long-format training data (same as used in the run).
        y_train: Training labels.
        seed: Random seed for models.
        save_dir: Directory to save pkl pipelines.

    Returns:
        A dict mapping model name -> saved pkl path.

    Raises:
        ValueError: If required keys are missing in `out`.
    """
    need_keys = ["X_meta_train", "y_train_aligned", "base_oof"]
    miss = [k for k in need_keys if k not in out]
    if miss:
        raise ValueError(f"out missing keys: {miss}")

    os.makedirs(save_dir, exist_ok=True)

    X_meta_train = out["X_meta_train"].copy()
    y_tr = out["y_train_aligned"].copy()
    x1_oof = np.asarray(out["base_oof"]).ravel()

    if "x1" not in X_meta_train.columns:
        X_meta_train["x1"] = x1_oof

    meta_cols = list(X_meta_train.columns)

    X1_tr, _ = generate_features_split(X_train)
    X1_tr = X1_tr.replace([np.inf, -np.inf], 0).fillna(0)
    X1_tr = X1_tr.reindex(index=y_tr.index)

    from tabpfn import TabPFNClassifier
    tab_model = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    tab_model.fit(X1_tr, y_tr)

    models = build_meta_models(seed=seed)

    saved: Dict[str, str] = {}
    for name, model in models.items():
        m = clone(model)
        m.fit(X_meta_train[meta_cols], y_tr)

        pipe = TabPFNThenBoostPipeline(
            tabpfn_model=tab_model,
            meta_model=m,
            meta_cols=meta_cols,
        )

        pkl_path = os.path.join(save_dir, f"TabPFN_plus_{name}.pkl")
        joblib.dump(pipe, pkl_path)
        saved[name] = pkl_path

    return saved


def save_roc_png_pdf(
    y_true,
    y_score,
    filename_base: str,
    dpi: int = 450,
    save_dir: str = "model_artifacts",
    legend_fontsize: int = 26,
) -> float:
    """
    Generate and save ROC curve as PNG/PDF files, return ROC-AUC score.
    
    Args:
        y_true: True binary labels (array-like, any shape), flattened to 1D for calculation.
        y_score: Target scores (array-like, any shape) - probabilities/decision scores, flattened to 1D.
        filename_base: Base name for saved files (without extension, e.g., "roc_curve").
        dpi: Resolution (dots per inch) for PNG file (default: 450).
        save_dir: Directory to save PNG/PDF files (default: "model_artifacts").
        legend_fontsize: Font size for AUC legend in plot (default: 26).
    
    Returns:
        float: ROC-AUC score (area under the ROC curve) for input labels/scores.
    
    Raises:
        OSError: If save directory cannot be created/written to.
        ValueError: If y_true/y_score have mismatched lengths, or y_true is not binary.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(12, 9))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", alpha=0.8)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=legend_fontsize)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    png_path = os.path.join(save_dir, f"{filename_base}.png")
    pdf_path = os.path.join(save_dir, f"{filename_base}.pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    return float(auc)


def save_last_created_figure_as_png_pdf(
    plot_callable,
    png_path: str,
    pdf_path: str,
    dpi: int = 300,
):
    """
    Execute plotting function and save the last created matplotlib figure as PNG/PDF.
    
    Temporarily override plt.show()/plt.close() to prevent internal figure closure,
    capture the last new figure created by the callable, save it to specified paths,
    then restore original plt.show()/plt.close() and display the figure.
    
    Args:
        plot_callable: Callable function that generates matplotlib figure (no input args).
        png_path: Full path to save PNG file (e.g., "../picture/plot.png").
        pdf_path: Full path to save PDF file (e.g., "../picture/plot.pdf").
        dpi: Resolution (dots per inch) for PNG file (default: 300).
    
    Returns:
        Tuple[str, str]: (PNG file path, PDF file path) where figures were saved.
    
    Raises:
        OSError: If save paths are unwritable or directories do not exist.
        IndexError: If no new figures are created by plot_callable.
        TypeError: If plot_callable is not a callable function.
    """
    old_show = plt.show
    old_close = plt.close
    plt.show = lambda *a, **k: None
    plt.close = lambda *a, **k: None

    before = set(plt.get_fignums())
    plot_callable()
    after = set(plt.get_fignums())

    new_figs = sorted(list(after - before))
    fig = plt.figure(new_figs[-1]) if len(new_figs) else plt.gcf()

    fig.canvas.draw()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.show = old_show
    plt.close = old_close

    plt.figure(fig.number)
    plt.show()

    return png_path, pdf_path


def build_feature_df_for_train_test(
    X_train: pd.DataFrame,
    y_train: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    label_col: str = "structural_breakpoint",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Generate engineered feature tables for train/test sets with aligned labels (0/1).
    
    Combines compact (X1) and rich (XS) features from generate_features_split, cleans invalid values,
    aligns train/test feature columns, and ensures labels are 0/1 series aligned to feature indices.
    
    Args:
        X_train: Long-format training DataFrame (contains id/period/time/value columns).
        y_train: Training labels (any format compatible with ensure_y_series).
        X_test: Long-format test DataFrame (same structure as X_train).
        y_test: Test labels (any format compatible with ensure_y_series).
        label_col: Column name for label if y_train/y_test are DataFrames (default: "structural_breakpoint").
    
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train_feat: Cleaned train feature table (one row per id, common columns only).
            - y_train_aligned: Aligned train labels (0/1, same index as X_train_feat).
            - X_test_feat: Cleaned test feature table (one row per id, same columns as X_train_feat).
            - y_test_aligned: Aligned test labels (0/1, same index as X_test_feat).
    
    Raises:
        KeyError: If required columns are missing from X_train/X_test, or label_col not found in y DataFrames.
        ValueError: If no common columns exist between train/test feature tables, or label alignment fails.
        OSError: If feature generation functions encounter file/system errors.
    """
    X1_tr, XS_tr = generate_features_split(X_train)
    X_train_feat = X1_tr.join(XS_tr, how="inner")
    X_train_feat = X_train_feat.replace([np.inf, -np.inf], 0).fillna(0)
    y_train_aligned = ensure_y_series(y_train, X_train_feat.index, label_col=label_col)

    X1_te, XS_te = generate_features_split(X_test)
    X_test_feat = X1_te.join(XS_te, how="inner")
    X_test_feat = X_test_feat.replace([np.inf, -np.inf], 0).fillna(0)
    y_test_aligned = ensure_y_series(y_test, X_test_feat.index, label_col=label_col)

    common_cols = X_train_feat.columns.intersection(X_test_feat.columns)
    X_train_feat = X_train_feat[common_cols]
    X_test_feat = X_test_feat[common_cols]

    return X_train_feat, y_train_aligned, X_test_feat, y_test_aligned


def get_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Get continuous scores for ROC/AUC evaluation.

    Args:
        model: A fitted estimator.
        X: Feature matrix.

    Returns:
        A one-dimensional array of continuous scores.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def oof_auc_roc(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 10,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute strict out-of-fold ROC/AUC on training data.

    Args:
        model: An sklearn-compatible estimator (cloneable).
        X: Training features.
        y: Training labels (0/1).
        n_folds: Number of folds for StratifiedKFold.
        seed: Random seed.

    Returns:
        A tuple (auc, fpr, tpr, oof_scores).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    for tr, va in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = get_score(m, X.iloc[va])

    auc = roc_auc_score(y, oof)
    fpr, tpr, _ = roc_curve(y, oof)
    return auc, fpr, tpr, oof


def test_auc_roc(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit on full training data and evaluate ROC/AUC on hold-out test data.

    Args:
        model: An sklearn-compatible estimator.
        X_train: Training features.
        y_train: Training labels (0/1).
        X_test: Test features.
        y_test: Test labels (0/1).

    Returns:
        A tuple (auc, fpr, tpr, test_scores).
    """
    model.fit(X_train, y_train)
    pred = get_score(model, X_test)
    auc = roc_auc_score(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, pred)
    return auc, fpr, tpr, pred


def tabpfn_oof_auc_roc(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 10,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute strict out-of-fold ROC/AUC for TabPFN.

    Args:
        X: Training features.
        y: Training labels (0/1).
        n_folds: Number of folds for StratifiedKFold.
        seed: Random seed.

    Returns:
        A tuple (auc, fpr, tpr, oof_scores).
    """
    from tabpfn import TabPFNClassifier

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    for tr, va in skf.split(X, y):
        m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]

    auc = roc_auc_score(y, oof)
    fpr, tpr, _ = roc_curve(y, oof)
    return auc, fpr, tpr, oof


def tabpfn_test_auc_roc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit TabPFN on full training data and evaluate ROC/AUC on hold-out test data.

    Args:
        X_train: Training features.
        y_train: Training labels (0/1).
        X_test: Test features.
        y_test: Test labels (0/1).

    Returns:
        A tuple (auc, fpr, tpr, test_scores).
    """
    from tabpfn import TabPFNClassifier

    m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    m.fit(X_train, y_train)
    pred = m.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, pred)
    return auc, fpr, tpr, pred


def plot_roc_curves(
    curves: List[Tuple[str, float, np.ndarray, np.ndarray]],
    save_base: Optional[str] = None,
    dpi: int = 450,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Plot ROC curves and optionally save the figure as both PNG and PDF.

    Args:
        curves: List of (name, auc, fpr, tpr).
        save_base: Optional output path without extension.
        dpi: DPI for saved PNG.

    Returns:
        A tuple (png_path, pdf_path). If save_base is None, returns (None, None).
    """
    fig = plt.figure()
    for name, auc, fpr, tpr in curves:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], "--", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()

    png_path = None
    pdf_path = None
    if save_base is not None:
        png_path = save_base + ".png"
        pdf_path = save_base + ".pdf"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.show()
    return png_path, pdf_path


def build_models(seed: int = 42) -> Dict[str, Any]:
    """
    Build candidate models for comparison.

    Args:
        seed: Random seed.

    Returns:
        A dictionary mapping model names to estimators.
    """
    models: Dict[str, Any] = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=False, random_state=seed)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            max_depth=10,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=8,
            colsample_bytree=0.6,
            subsample=0.9,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        ),
    }

    if _HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=seed,
        )

    return models


def run_compare_models_train_oof_and_test(
    X_train: pd.DataFrame,
    y_train: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    n_folds: int = 10,
    seed: int = 42,
    include_tabpfn: bool = True,
    label_col: str = "structural_breakpoint",
    save_train_stem: str = "roc_train_oof",
    save_test_stem: str = "roc_test",
    dpi: int = 450,
) -> Dict[str, Any]:
    """
    Compare models on training OOF ROC and hold-out test ROC using engineered features.

    Requirements:
        generate_features_split(X) and ensure_y_series(y, index, label_col=...) must be defined.

    Args:
        X_train: Long-format training data.
        y_train: Training labels.
        X_test: Long-format test data.
        y_test: Test labels.
        n_folds: Number of folds for OOF evaluation.
        seed: Random seed.
        include_tabpfn: Whether to include TabPFN.
        label_col: Label column name if y is a DataFrame.
        save_train_stem: Output filename stem for training ROC figure.
        save_test_stem: Output filename stem for test ROC figure.
        dpi: DPI for saved figures.

    Returns:
        A dictionary containing feature tables, predictions, and used model names.
    """
    set_report_plot_style()
    out_dir = ensure_picture_dir_in_parent("picture")
    train_base = os.path.join(out_dir, save_train_stem)
    test_base = os.path.join(out_dir, save_test_stem)

    Xtr, ytr, Xte, yte = build_feature_df_for_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_col=label_col,
    )

    models = build_models(seed=seed)

    train_curves: List[Tuple[str, float, np.ndarray, np.ndarray]] = []
    train_oof_preds: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(name)
        t0 = time.time()
        auc, fpr, tpr, oof_pred = oof_auc_roc(model, Xtr, ytr, n_folds=n_folds, seed=seed)
        train_curves.append((name, auc, fpr, tpr))
        train_oof_preds[name] = oof_pred
        _ = time.time() - t0

    if include_tabpfn:
        t0 = time.time()
        tab_auc, tab_fpr, tab_tpr, tab_oof = tabpfn_oof_auc_roc(Xtr, ytr, n_folds=n_folds, seed=seed)
        train_curves.append(("TabPFN", tab_auc, tab_fpr, tab_tpr))
        train_oof_preds["TabPFN"] = tab_oof
        _ = time.time() - t0

    train_png, train_pdf = plot_roc_curves(
        train_curves,
        save_base=train_base,
        dpi=dpi,
    )

    test_curves: List[Tuple[str, float, np.ndarray, np.ndarray]] = []
    test_preds: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        t0 = time.time()
        m = clone(model)
        auc, fpr, tpr, pred = test_auc_roc(m, Xtr, ytr, Xte, yte)
        test_curves.append((name, auc, fpr, tpr))
        test_preds[name] = pred
        _ = time.time() - t0

    if include_tabpfn:
        t0 = time.time()
        tab_auc, tab_fpr, tab_tpr, tab_pred = tabpfn_test_auc_roc(Xtr, ytr, Xte, yte)
        test_curves.append(("TabPFN", tab_auc, tab_fpr, tab_tpr))
        test_preds["TabPFN"] = tab_pred
        _ = time.time() - t0

    test_png, test_pdf = plot_roc_curves(
        test_curves,
        save_base=test_base,
        dpi=dpi,
    )

    used_models = list(models.keys()) + (["TabPFN"] if include_tabpfn else [])

    return {
        "X_train_feat": Xtr,
        "y_train": ytr,
        "X_test_feat": Xte,
        "y_test": yte,
        "models": models,
        "train_oof_preds": train_oof_preds,
        "test_preds": test_preds,
        "used_models": used_models,
        "train_roc_png": train_png,
        "train_roc_pdf": train_pdf,
        "test_roc_png": test_png,
        "test_roc_pdf": test_pdf,
    }


def ensure_picture_dir() -> str:
    """
    Resolve an image output directory.

    Args:
        folder_name: Name of the image folder.

    Returns:
        Absolute path of the resolved folder.

    Raises:
        OSError: If neither current nor parent directory can be used.
    """
    out_dir = os.path.join("..", "picture")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_final_model_train_test_roc_from_out(
    out: dict,
    model_name: str,
    save_train_stem: str,
    save_test_stem: str,
    dpi: int = 450,
) -> Dict[str, Any]:
    """
    Plot ROC curves for a specific model from out dict (no training/prediction).

    Args:
        out: Output dict from run_tabpfn_x1_plus_s234_three_models.
        model_name: Model name (e.g., "CatBoost", "LightGBM", "XGBoost").
        save_train_stem: Filename stem for training ROC.
        save_test_stem: Filename stem for test ROC.
        dpi: DPI for PNG.

    Returns:
        Dict with train_auc, test_auc, train_png, train_pdf, test_png, test_pdf.
    """
    required_keys = ["meta_oof_scores", "meta_test_scores", "y_train_aligned", "y_test_aligned"]
    missing = [k for k in required_keys if k not in out]
    if missing:
        raise KeyError(f"Missing required keys in out dict: {missing}")

    train_scores = out["meta_oof_scores"][model_name]
    test_scores = out["meta_test_scores"][model_name]
    y_train_aligned = out["y_train_aligned"]
    y_test_aligned = out["y_test_aligned"]

    set_report_plot_style()
    plt.rcParams.update({"legend.fontsize": 22})
    out_dir = ensure_picture_dir()

    fig_train = plt.figure(figsize=(12, 9))
    fpr_train, tpr_train, _ = roc_curve(y_train_aligned, train_scores)
    auc_train = roc_auc_score(y_train_aligned, train_scores)
    plt.plot(fpr_train, tpr_train, label=f"AUC = {auc_train:.4f}")
    plt.plot([0, 1], [0, 1], "--", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    train_png = os.path.join(out_dir, f"{save_train_stem}.png")
    train_pdf = os.path.join(out_dir, f"{save_train_stem}.pdf")
    plt.savefig(train_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(train_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig_train)

    fig_test = plt.figure(figsize=(12, 9))
    fpr_test, tpr_test, _ = roc_curve(y_test_aligned, test_scores)
    auc_test = roc_auc_score(y_test_aligned, test_scores)
    plt.plot(fpr_test, tpr_test, label=f"AUC = {auc_test:.4f}")
    plt.plot([0, 1], [0, 1], "--", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    test_png = os.path.join(out_dir, f"{save_test_stem}.png")
    test_pdf = os.path.join(out_dir, f"{save_test_stem}.pdf")
    plt.savefig(test_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(test_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig_test)

    print(f"[{model_name}] Train OOF AUC = {auc_train:.4f}")
    print(f"[{model_name}] Test AUC      = {auc_test:.4f}")

    return {
        "train_auc": float(auc_train),
        "test_auc": float(auc_test),
        "train_png": train_png,
        "train_pdf": train_pdf,
        "test_png": test_png,
        "test_pdf": test_pdf,
    }


def plot_feature_importance(model, X_meta: pd.DataFrame, k: int, stem: str) -> None:
    """
    Plot feature importance.

    Args:
        model: Fitted model with feature importances.
        X_meta: Feature table.
        k: Top-k features.
        stem: Output filename stem (no extension).

    Raises:
        AttributeError: If model has no importance attribute.
    """
    set_report_plot_style()

    out_dir = ensure_picture_dir()
    feats = list(X_meta.columns)

    if hasattr(model, "get_feature_importance"):
        imp = np.asarray(model.get_feature_importance(), dtype=float)
    else:
        imp = np.asarray(model.feature_importances_, dtype=float)

    idx = np.argsort(imp)[::-1][:k]
    names = [feats[i] for i in idx]
    vals = imp[idx]

    plt.figure(figsize=(12, 0.45 * len(names)))
    plt.barh(names, vals)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.35)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{stem}_feature_importance_top{k}.png"), dpi=450, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{stem}_feature_importance_top{k}.pdf"), bbox_inches="tight")
    plt.show()


def plot_shap(model, X_meta: pd.DataFrame, k: int, stem: str) -> None:
    """
    Plot SHAP bar and beeswarm.

    Args:
        model: Fitted tree model.
        X_meta: Feature table.
        k: Top-k features to display.
        stem: Output filename stem (no extension).

    Raises:
        ImportError: If shap is not installed.
    """
    set_report_plot_style()

    out_dir = ensure_picture_dir()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_meta)
    sv = sv[1] if isinstance(sv, list) else sv

    plt.figure(figsize=(12, 9))
    shap.summary_plot(sv, X_meta, plot_type="bar", max_display=k, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}_shap_bar.png"), dpi=450, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{stem}_shap_bar.pdf"), bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12, 9))
    shap.summary_plot(sv, X_meta, max_display=k, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}_shap_beeswarm.png"), dpi=450, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{stem}_shap_beeswarm.pdf"), bbox_inches="tight")
    plt.show()


def top_k_features_from_model(model, feature_names, k):
    """
    Extract top-k feature names from a fitted model.

    Args:
        model: Fitted model with feature importances.
        feature_names: Names aligned with importance vector.
        k: Number of features.

    Returns:
        A list of top-k feature names.

    Raises:
        ValueError: If model does not support feature importances.
    """
    if hasattr(model, "get_feature_importance"):
        imp = np.asarray(model.get_feature_importance(), dtype=float)
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        raise ValueError("The model does not support the calculation of feature importance.")
    imp = np.nan_to_num(imp, nan=0.0)
    idx = np.argsort(imp)[::-1][:k]
    return [feature_names[i] for i in idx]


def plot_top8_feature_distributions(X_meta: pd.DataFrame, y: pd.Series, model, stem: str) -> None:
    """
    Plot distribution of top 8 important features (0/1 label groups) with consistent styling.
    
    Generates 4x2 subplot grid of feature distributions (histograms) for negative (y=0) and positive (y=1) labels,
    saves plots as high-resolution PNG/PDF to specified directory, and displays the figure.
    
    Args:
        X_meta: Feature DataFrame (one row per sample, columns as feature names).
        y: Binary target series (0/1) aligned with X_meta index.
        model: Fitted model with feature importance/coefficient attributes (compatible with top_k_features_from_model).
        stem: Base filename for saved plots (without extension, e.g., "train" or "test").
    
    Returns:
        None
    
    Raises:
        KeyError: If top 8 feature names from model are not present in X_meta columns.
        ValueError: If y contains non-binary values (not 0/1), or X_meta/y indices are misaligned.
        OSError: If plot directory is unwritable or file save fails.
        IndexError: If model returns fewer than 8 features.
    """
    set_report_plot_style()

    out_dir = ensure_picture_dir()
    top8 = top_k_features_from_model(model, list(X_meta.columns), 8)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()

    for i, col in enumerate(top8):
        ax = axes[i]
        ax.hist(X_meta.loc[y == 0, col], bins=30, alpha=0.6)
        ax.hist(X_meta.loc[y == 1, col], bins=30, alpha=0.6)
        ax.set_title(col)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}_top8_feature_dist_4x2.png"), dpi=450, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{stem}_top8_feature_dist_4x2.pdf"), bbox_inches="tight")
    plt.show()


def export_model_figures(
    out: dict,
    model_name: str,
    model_path: str,
    stem: str,
    k_features: int = 25,
) -> None:
    """
    Export all figures for a specific model (feature importance, SHAP, distributions, ROC).

    Requirements:
        The following callables must already be defined/imported in your runtime:
          - plot_feature_importance(model, X_meta, k, stem)
          - plot_shap(model, X_meta, k, stem)
          - plot_top8_feature_distributions(X_meta, y, model, stem)
          - plot_final_model_train_test_roc_from_out(out, model_name, save_train_stem, save_test_stem, dpi)

        Each plotting function is expected to save into the directory returned by resolve_picture_dir()
        (i.e., "./picture" first, then "../picture" fallback). If your plotting functions currently
        use their own ensure_picture_dir(), replace it with resolve_picture_dir() inside them.

    Args:
        out: Output dict from run_tabpfn_x1_plus_s234_three_models.
        model_name: Model name key used in out (e.g., "CatBoost").
        model_path: Path to saved pipeline pickle.
        stem: Filename stem for saved figures.
        k_features: Number of top features to plot.
        dpi: DPI for ROC PNG.

    Returns:
        A dict with resolved picture directory and the ROC file stems used.

    Raises:
        KeyError: If out lacks required keys.
        FileNotFoundError: If model_path does not exist.
        NameError: If required plotting functions are not defined.
    """
    pipe = joblib.load(model_path)
    X_meta = out["X_meta_train"]
    y_tr = out["y_train_aligned"].astype(int)

    plot_feature_importance(pipe.meta_model, X_meta, k_features, stem)
    plot_shap(pipe.meta_model, X_meta, k_features, stem)
    plot_top8_feature_distributions(X_meta, y_tr, pipe.meta_model, stem)

    plot_final_model_train_test_roc_from_out(
        out=out,
        model_name=model_name,
        save_train_stem=f"{stem}_train_roc_oof",
        save_test_stem=f"{stem}_test_roc",
        dpi=450,
    )
