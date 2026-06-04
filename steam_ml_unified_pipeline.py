"""
================================================================================
STEAM GAMES ML PREDICTION PIPELINE - UNIFIED SCRIPT
================================================================================

PROJECT INFORMATION:
    Script Name:    Steam ML Unified Pipeline
    Version:        1.0.0
    Created:        2026-05-31
    
TEAM MEMBERS:
    Lead Developer: [Your Name]
    Data Scientist: [Name]
    ML Engineer:    [Name]
    
DESCRIPTION:
    Comprehensive Machine Learning pipeline for predicting Steam game success
    rates. Includes data collection, cleaning, feature engineering, model
    training, and evaluation.
    
USAGE:
    python steam_ml_unified_pipeline.py [mode]
    
    Modes:
        full          - Complete pipeline with training (default)
        prepare       - Data preparation only (no model training)
        predict       - Interactive prediction CLI
        exploration   - Data exploration only
        cleaning      - Data cleaning only
        training      - Model training only (requires prepared data)
        evaluation    - Model evaluation only (requires trained model)

================================================================================
"""

from __future__ import annotations

import json
import math
import shutil
import re
import time
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TARGET = "Is_highly_rated"
DEFAULT_KAGGLE_DATASET = "fronkongames/steam-games-dataset"
TARGET_MODE = "trend_playtime"  # options: 'default', 'metacritic_only', 'trend_playtime'
TARGET_FALLBACK_MODE = "default"
TARGET_MINIMUM_POSITIVES = 250
TARGET_MINIMUM_POSITIVE_RATIO = 0.005

# Regex patterns
NUMERIC_SANITIZE_RE = re.compile(r"[^0-9\.,\-]+")
CLEAN_QUOTES_RE = re.compile(r"^[\'\"]+|[\'\"]+$")
SEPARATOR_RE = re.compile(r"\s*[,;]\s*")
SPLIT_RE = re.compile(r"[,;]")
NUMBER_RE = re.compile(r"\d[\d,]*")

RAW_COLUMNS = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU", "Required age",
    "Price", "Discount", "DLC count", "About the game", "Supported languages",
    "Full audio languages", "Reviews", "Header image", "Website", "Support url",
    "Support email", "Windows", "Mac", "Linux", "Metacritic score", "Metacritic url",
    "User score", "Positive", "Negative", "Score rank", "Achievements", "Recommendations",
    "Notes", "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks", "Developers", "Publishers",
    "Categories", "Genres", "Tags", "Screenshots", "Movies",
]

FINAL_COLUMNS = [
    "AppID", "Name", "Genres", "Release_year", "Days_since_release", "Platform_count",
    "Is_multiplatform", "Price", "Is_free", "Total_reviews", "Review_ratio", TARGET,
    "Log_owners", "Has_achievements", "Log_total_reviews", "Genre_count",
]

MODEL_FEATURES = [
    "Release_year", "Days_since_release", "Platform_count", "Is_multiplatform",
    "Price", "Is_free", "Log_owners", "Has_achievements", "Genre_count",
]

# CLI Prediction prompts
PREDICTION_PROMPTS = {
    "Release_year": "Rok wydania",
    "Days_since_release": "Dni od premiery",
    "Platform_count": "Liczba platform",
    "Price": "Cena USD",
    "Is_free": "Czy darmowa? 1/0",
    "Total_reviews": "Laczna liczba recenzji",
    "Review_ratio": "Udzial pozytywnych recenzji 0-1",
    "Log_owners": "Log wlascicieli, np. 10",
    "Has_achievements": "Czy ma osiagniecia? 1/0",
    "Log_total_reviews": "Log recenzji, np. 5",
    "Genre_count": "Liczba gatunkow",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs() -> None:
    """Create necessary directories if they don't exist."""
    for path in (DATA_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().replace(microsecond=0).isoformat()


def log_message(message: str) -> None:
    """Print timestamped log message."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    """Format seconds to human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.1f}s"


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save dictionary as JSON file."""
    ensure_dirs()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_raw_csv() -> Path:
    """Find the latest raw CSV file in data directory."""
    files = sorted(DATA_DIR.glob("games_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    files = [p for p in files if p.name not in {"games_cleaned.csv", "games_engineered.csv"}]
    if not files:
        raise FileNotFoundError("Nie znaleziono surowego pliku data/games_YYYYMMDD_HHMMSS.csv")
    return files[0]


def download_kaggle_dataset(dataset: str = DEFAULT_KAGGLE_DATASET) -> Path:
    """Download dataset from Kaggle."""
    ensure_dirs()
    log_message(f"Pobieranie datasetu Kaggle: {dataset}")
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Brakuje biblioteki kagglehub. Uruchom: pip install -r requirements.txt") from exc

    downloaded_dir = Path(kagglehub.dataset_download(dataset))
    csv_files = sorted(downloaded_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"Dataset {dataset} nie zawiera pliku CSV.")

    source = csv_files[0]
    output = DATA_DIR / f"games_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy2(source, output)
    log_message(f"Zapisano surowe dane: {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
    save_json(
        {
            "timestamp": timestamp(),
            "dataset": dataset,
            "downloaded_directory": str(downloaded_dir),
            "source_csv": str(source),
            "output_csv": str(output),
            "size_bytes": output.stat().st_size,
        },
        REPORTS_DIR / "01_data_collection_report.json",
    )
    return output


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load raw CSV data."""
    source = path or latest_raw_csv()
    header = source.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "DiscountDLC count" in header:
        return pd.read_csv(source, names=RAW_COLUMNS, header=0, low_memory=False)
    return pd.read_csv(source, low_memory=False)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case."""
    df = df.rename(columns=lambda name: str(name).strip())
    rename = {
        "Metacritic score": "Metacritic_score",
        "User score": "User_score",
        "Release date": "Release_date",
        "Estimated owners": "Estimated_owners",
        "Required age": "Required_age",
        "Discount": "Discount",
        "DLC count": "DLC_count",
        "Peak CCU": "Peak_CCU",
        "Average playtime forever": "Average_playtime_forever",
        "Average playtime two weeks": "Average_playtime_two_weeks",
        "Median playtime forever": "Median_playtime_forever",
        "Median playtime two weeks": "Median_playtime_two_weeks",
    }
    return df.rename(columns=rename)


def clean_text_value(value: Any) -> str | float:
    """Clean text values."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"[]", "nan", "none"}:
        return np.nan
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
        text = text.replace('"', "").replace("'", "")
    text = CLEAN_QUOTES_RE.sub("", text)
    text = SEPARATOR_RE.sub(", ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_numeric_value(value: Any, default: float = np.nan) -> float:
    """Parse numeric value with error handling."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return default
    text = NUMERIC_SANITIZE_RE.sub("", text)
    if not text:
        return default
    if "." in text and "," in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".") if text.count(",") == 1 else text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return default


def split_count(value: Any) -> int:
    """Count number of items in comma/semicolon-separated string."""
    if pd.isna(value):
        return 0
    return len([part for part in SPLIT_RE.split(str(value)) if part.strip()])


def parse_owner_midpoint(value: Any) -> float:
    """Extract numeric midpoint from owner estimate range."""
    if pd.isna(value):
        return 0.0
    numbers = [int(num.replace(",", "")) for num in NUMBER_RE.findall(str(value))]
    if not numbers:
        return 0.0
    return float(numbers[0]) if len(numbers) == 1 else float(sum(numbers[:2]) / 2)


def build_default_target(df: pd.DataFrame) -> pd.Series:
    """Build target variable using default logic."""
    return (
        ((df["Review_ratio"] >= 0.70) & (df["Total_reviews"] >= 20))
        | (df["Metacritic_score"] >= 75)
    ).astype(int)


def build_metacritic_only_target(df: pd.DataFrame) -> pd.Series:
    """Build target variable using Metacritic scores only."""
    return (df["Metacritic_score"] >= 75).astype(int)


def build_trend_playtime_target(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """Build target variable based on playtime trend."""
    ratio = df["Average_playtime_two_weeks"] / (df["Average_playtime_forever"] + 1e-6)
    df["Recent_playtime_ratio"] = ratio
    eligible = df[(df["Average_playtime_two_weeks"] > 0) & (df["Average_playtime_forever"] > 0)]
    cutoff = float(eligible["Recent_playtime_ratio"].quantile(0.90)) if len(eligible) else float(ratio.quantile(0.90))
    target = ((ratio >= cutoff) & (ratio > 0)).astype(int)
    return target, cutoff


def resolve_duplicate_appid(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate AppIDs, keeping the best one."""
    if "AppID" not in df.columns:
        return df, 0
    temp = df.copy()
    temp["Positive"] = pd.to_numeric(temp.get("Positive", 0), errors="coerce").fillna(0).astype(int)
    temp["Negative"] = pd.to_numeric(temp.get("Negative", 0), errors="coerce").fillna(0).astype(int)
    temp["Total_reviews"] = temp["Positive"] + temp["Negative"]
    temp["Metacritic_score"] = pd.to_numeric(temp.get("Metacritic_score", np.nan), errors="coerce")
    temp["User_score"] = pd.to_numeric(temp.get("User_score", np.nan), errors="coerce")
    temp = temp.sort_values(
        ["AppID", "Total_reviews", "Metacritic_score", "User_score"],
        ascending=[True, False, False, False],
    )
    before = len(temp)
    temp = temp.drop_duplicates(subset=["AppID"], keep="first").drop(columns=["Total_reviews"])
    return temp, before - len(temp)


def bool_to_int(series: pd.Series) -> pd.Series:
    """Convert boolean-like values to integers."""
    return series.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)


def remove_extreme_outliers(
    df: pd.DataFrame, columns: list[str], iqr_multiplier: float = 1.5
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove rows with extreme outlier values using IQR method."""
    counts: dict[str, int] = {}
    keep_mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            counts[col] = 0
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        col_mask = series.between(lower, upper) | series.isna()
        counts[col] = int((~col_mask).sum())
        keep_mask &= col_mask
    filtered = df[keep_mask].copy()
    removed_total = int((~keep_mask).sum())
    return filtered, {"per_column": counts, "rows_removed": removed_total}


def platform_label(row: pd.Series) -> str:
    """Generate platform label from platform columns."""
    platforms = []
    if int(row.get("Has_windows", 0) or 0) == 1:
        platforms.append("Windows")
    if int(row.get("Has_mac", 0) or 0) == 1:
        platforms.append("Mac")
    if int(row.get("Has_linux", 0) or 0) == 1:
        platforms.append("Linux")
    return " + ".join(platforms) if platforms else "Unknown"


# ============================================================================
# DATA PROCESSING CLASSES
# ============================================================================

class DataCleaner:
    """Clean raw data for ML pipeline."""

    def __init__(self, source: Path | None = None):
        self.source = source or latest_raw_csv()
        self.report: dict[str, Any] = {"timestamp": timestamp(), "source": str(self.source)}

    def run(self) -> pd.DataFrame:
        """Execute data cleaning pipeline."""
        ensure_dirs()
        log_message(f"Wczytywanie danych surowych: {self.source}")
        df = normalize_column_names(load_raw_data(self.source))
        start_rows = len(df)
        self.report["input_shape"] = list(df.shape)
        log_message(f"Dane surowe: {df.shape[0]:,} wierszy, {df.shape[1]} kolumn".replace(",", " "))

        if "AppID" in df.columns:
            duplicates = int(df.duplicated(subset=["AppID"]).sum())
        else:
            duplicates = int(df.duplicated().sum())
        self.report["duplicate_rows"] = duplicates

        required = [c for c in ["AppID", "Name", "Release_date"] if c in df.columns]
        before_required = len(df)
        df = df.dropna(subset=required)
        self.report["missing_required_removed"] = before_required - len(df)

        df["Release_date"] = pd.to_datetime(df["Release_date"].astype(str).str.strip(), errors="coerce")
        before_date = len(df)
        today = pd.Timestamp.today().normalize()
        df = df[df["Release_date"].notna() & (df["Release_date"] <= today)]
        self.report["invalid_release_date_removed"] = before_date - len(df)

        numeric_defaults = {
            "Price": 0,
            "Positive": 0,
            "Negative": 0,
            "Estimated_owners": np.nan,
            "Metacritic_score": np.nan,
            "User_score": np.nan,
            "Achievements": 0,
            "Recommendations": 0,
            "Peak_CCU": 0,
            "Required_age": 0,
            "DLC_count": 0,
        }
        for column, default in numeric_defaults.items():
            if column in df.columns:
                df[column] = df[column].map(lambda value: parse_numeric_value(value, default)).fillna(default)

        if "AppID" in df.columns:
            df, resolved_duplicates = resolve_duplicate_appid(df)
            self.report["duplicates_removed"] = resolved_duplicates
        else:
            self.report["duplicates_removed"] = 0

        if "Price" in df.columns:
            df["Price"] = df["Price"].clip(lower=0).round(2)
            df = df[df["Price"] <= 1000]

        df["Total_reviews"] = (
            pd.to_numeric(df.get("Positive", 0), errors="coerce").fillna(0).astype(int)
            + pd.to_numeric(df.get("Negative", 0), errors="coerce").fillna(0).astype(int)
        )
        df["Owner_midpoint"] = df.get("Estimated_owners", pd.Series(np.nan, index=df.index)).map(parse_owner_midpoint)
        df["Days_since_release"] = (today - df["Release_date"]).dt.days.clip(lower=0).fillna(0).astype(int)

        outlier_cols = [c for c in ["Price", "Total_reviews", "Days_since_release", "Estimated_owners"] if c in df.columns]
        if outlier_cols:
            df, outlier_info = remove_extreme_outliers(df, outlier_cols, iqr_multiplier=1.5)
            self.report["outliers_removed"] = outlier_info
        else:
            self.report["outliers_removed"] = {"per_column": {}, "rows_removed": 0}

        for column in ["Positive", "Negative", "Metacritic_score", "User_score", "Achievements"]:
            if column in df.columns:
                df[column] = df[column].clip(lower=0)

        if "Metacritic_score" in df.columns:
            df["Metacritic_score"] = df["Metacritic_score"].clip(0, 100)
        if "User_score" in df.columns:
            df["User_score"] = df["User_score"].clip(0, 100)

        for column in ["Windows", "Mac", "Linux"]:
            if column in df.columns:
                df[column] = bool_to_int(df[column])

        for column in ["Name", "Developers", "Publishers", "Categories", "Genres", "Tags", "Reviews"]:
            if column in df.columns:
                df[column] = df[column].map(clean_text_value)

        before_no_signal = len(df)
        total_reviews = df.get("Positive", 0) + df.get("Negative", 0)
        review_signal = total_reviews >= 5
        score_signal = (
            df.get("Metacritic_score", pd.Series(np.nan, index=df.index)).notna()
            | df.get("User_score", pd.Series(np.nan, index=df.index)).notna()
        )
        genres_available = (
            df.get("Genres", pd.Series(np.nan, index=df.index)).notna()
            if "Genres" in df.columns
            else pd.Series(True, index=df.index)
        )
        useful_mask = genres_available & (review_signal | score_signal)
        df = df[useful_mask].copy()
        self.report["no_ml_signal_removed"] = before_no_signal - len(df)

        self.report["output_shape"] = list(df.shape)
        self.report["rows_removed_total"] = start_rows - len(df)
        self.report["missing_after_cleaning"] = df.isna().sum().loc[lambda s: s > 0].to_dict()
        df.to_csv(DATA_DIR / "games_cleaned.csv", index=False)
        save_json(self.report, REPORTS_DIR / "03_data_cleaning_report.json")
        log_message(
            f"Czyszczenie zakonczone: zostalo {len(df):,}, usunieto {start_rows - len(df):,} rekordow".replace(",", " ")
        )
        return df


class FeatureEngineer:
    """Engineer features for ML models."""

    def __init__(self, input_path: Path = DATA_DIR / "games_cleaned.csv"):
        self.input_path = input_path
        self.report: dict[str, Any] = {"timestamp": timestamp(), "source": str(input_path)}

    def run(self) -> pd.DataFrame:
        """Execute feature engineering."""
        ensure_dirs()
        log_message(f"Feature engineering: wczytywanie {self.input_path}")
        df = normalize_column_names(pd.read_csv(self.input_path, low_memory=False))
        input_cols = len(df.columns)
        today = pd.Timestamp.today().normalize()
        df["Release_date"] = pd.to_datetime(df["Release_date"], errors="coerce")

        df["Release_year"] = df["Release_date"].dt.year.fillna(today.year).astype(int)
        df["Release_month"] = df["Release_date"].dt.month.fillna(1).astype(int)
        df["Release_quarter"] = df["Release_date"].dt.quarter.fillna(1).astype(int)
        df["Days_since_release"] = (today - df["Release_date"]).dt.days.clip(lower=0).fillna(0).astype(int)
        df["Is_recent"] = (df["Days_since_release"] <= 365).astype(int)

        required_numeric = [
            "Windows", "Mac", "Linux", "Positive", "Negative", "Metacritic_score",
            "User_score", "Achievements", "Price", "Average_playtime_forever",
            "Average_playtime_two_weeks", "Median_playtime_forever", "Median_playtime_two_weeks",
        ]
        for column in required_numeric:
            if column not in df.columns:
                df[column] = 0
        df[required_numeric] = df[required_numeric].apply(pd.to_numeric, errors="coerce").fillna(0)

        df["Total_reviews"] = (df["Positive"] + df["Negative"]).astype(int)
        df["Review_ratio"] = np.where(df["Total_reviews"] > 0, df["Positive"] / df["Total_reviews"], 0.0)
        review_q75 = float(df["Total_reviews"].quantile(0.75))
        df["Is_heavily_reviewed"] = (df["Total_reviews"] > review_q75).astype(int)
        df["Log_total_reviews"] = np.log1p(df["Total_reviews"])

        df["Has_metacritic"] = (df["Metacritic_score"] > 0).astype(int)
        df["Metacritic_category"] = pd.cut(
            df["Metacritic_score"],
            bins=[-0.01, 49, 74, 89, 100],
            labels=["Poor", "Fair", "Good", "Excellent"],
        ).astype(str)
        df.loc[df["Metacritic_score"] <= 0, "Metacritic_category"] = "Missing"
        df["User_score_normalized"] = np.where(df["User_score"] > 10, df["User_score"] / 100, df["User_score"] / 10)
        df["User_score_normalized"] = df["User_score_normalized"].clip(0, 1)

        target_mode = TARGET_MODE
        if target_mode == "metacritic_only":
            df[TARGET] = build_metacritic_only_target(df)
        elif target_mode == "trend_playtime":
            df[TARGET], cutoff = build_trend_playtime_target(df)
            df["Recent_playtime_quantile_cutoff"] = cutoff
            minimum_positive = max(TARGET_MINIMUM_POSITIVES, int(len(df) * TARGET_MINIMUM_POSITIVE_RATIO))
            if int(df[TARGET].sum()) < minimum_positive:
                log_message(
                    f"Trend target ma za malo pozytywnych ({int(df[TARGET].sum())}), powrot do {TARGET_FALLBACK_MODE}."
                )
                target_mode = TARGET_FALLBACK_MODE
                df[TARGET] = build_default_target(df)
        else:
            df[TARGET] = build_default_target(df)

        df["Target_mode_used"] = target_mode
        if df[TARGET].nunique() < 2:
            log_message("Uwaga: wygenerowano tylko jedną klasę docelową. Sprawdź TARGET_MODE lub dane wejściowe.")

        df["Log_achievements"] = np.log1p(df["Achievements"].clip(lower=0))
        df["Has_achievements"] = (df["Achievements"] > 0).astype(int)
        df["Owner_midpoint"] = df.get("Estimated_owners", pd.Series(0, index=df.index)).map(parse_owner_midpoint)
        df["Log_owners"] = np.log1p(df["Owner_midpoint"])
        df["Genre_count"] = df.get("Genres", pd.Series(np.nan, index=df.index)).map(split_count).clip(lower=0)
        df["Category_count"] = df.get("Categories", pd.Series(np.nan, index=df.index)).map(split_count).clip(lower=0)

        df["Is_free"] = (df["Price"] <= 0).astype(int)
        df["Price_category"] = pd.cut(
            df["Price"],
            bins=[-0.01, 0, 10, 30, 60, math.inf],
            labels=["Free", "Budget", "Standard", "Premium", "AAA"],
        ).astype(str)
        df["Log_price"] = np.log1p(df["Price"].clip(lower=0))

        for column in ["Metacritic_category", "Price_category"]:
            dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
            df = pd.concat([df, dummies], axis=1)

        for column in ["Price", "Metacritic_score", "User_score", "Positive", "Negative", "Review_ratio", "Achievements", "Days_since_release"]:
            std = df[column].std()
            if std and not np.isnan(std):
                df[f"{column}_normalized"] = (df[column] - df[column].mean()) / std
            else:
                df[f"{column}_normalized"] = 0.0

        df["Price_Rating_ratio"] = df["Price"] / (df["Metacritic_score"] + 1)
        df["Rating_Review_score"] = (df["Metacritic_score"] / 100) * df["Review_ratio"]
        df["Owners_Review_ratio"] = df["Log_owners"] * df["Review_ratio"]

        self.report["output_shape"] = list(df.shape)
        self.report["target_distribution"] = df[TARGET].value_counts(dropna=False).to_dict()
        df.to_csv(DATA_DIR / "games_engineered.csv", index=False)
        save_json(self.report, REPORTS_DIR / "04_feature_engineering_report.json")
        target_counts = df[TARGET].value_counts(normalize=True).mul(100).round(2).to_dict()
        log_message(f"Feature engineering zakonczony: {input_cols} -> {len(df.columns)} kolumn; target %: {target_counts}")
        return df


class DataExporter:
    """Export processed data for ML training."""

    def __init__(self, input_path: Path = DATA_DIR / "games_engineered.csv"):
        self.input_path = input_path

    def run(self) -> pd.DataFrame:
        """Execute data export and splitting."""
        ensure_dirs()
        log_message(f"Eksport danych ML: wczytywanie {self.input_path}")
        df = pd.read_csv(self.input_path, low_memory=False)
        for column in FINAL_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        final = df[FINAL_COLUMNS].copy()
        final["Genres"] = final["Genres"].fillna("Unknown")
        final["Name"] = final["Name"].fillna("Unknown")
        for column in MODEL_FEATURES + [TARGET]:
            final[column] = pd.to_numeric(final[column], errors="coerce").fillna(0)
        final = final.replace([np.inf, -np.inf], 0).dropna(subset=[TARGET])

        stratify_final = final[TARGET] if final[TARGET].nunique() > 1 else None
        if stratify_final is None:
            log_message("Uwaga: target zawiera tylko jedną klasę. Podzial bez stratify.")
        train, temp = train_test_split(final, test_size=0.30, random_state=RANDOM_STATE, stratify=stratify_final)
        stratify_temp = temp[TARGET] if temp[TARGET].nunique() > 1 else None
        val, test = train_test_split(temp, test_size=0.50, random_state=RANDOM_STATE, stratify=stratify_temp)
        final.to_csv(PROCESSED_DIR / "games_final.csv", index=False)
        train.to_csv(PROCESSED_DIR / "games_train.csv", index=False)
        val.to_csv(PROCESSED_DIR / "games_val.csv", index=False)
        test.to_csv(PROCESSED_DIR / "games_test.csv", index=False)
        try:
            final.to_parquet(PROCESSED_DIR / "games_final.parquet", index=False)
        except Exception:
            pass

        manifest = {
            "timestamp": timestamp(),
            "data_shape": list(final.shape),
            "features": FINAL_COLUMNS,
            "model_features": MODEL_FEATURES,
            "target": TARGET,
            "feature_groups": {
                "identifiers": ["AppID", "Name"],
                "temporal": ["Release_year", "Days_since_release"],
                "platform": ["Platform_count"],
                "reviews": ["Total_reviews", "Review_ratio", "Log_total_reviews"],
                "scores": [TARGET],
                "content": ["Log_owners", "Has_achievements", "Genre_count"],
                "price": ["Price", "Is_free"],
                "metadata": ["Genres"],
            },
            "split": {"train": len(train), "val": len(val), "test": len(test)},
            "class_distribution": final[TARGET].value_counts().to_dict(),
        }
        save_json(manifest, PROCESSED_DIR / "dataset_manifest.json")
        pd.DataFrame({"column_name": FINAL_COLUMNS}).to_csv(PROCESSED_DIR / "columns_documentation.csv", index=False)
        log_message(f"Split gotowy: train={len(train):,}, val={len(val):,}, test={len(test):,}".replace(",", " "))
        return final


class DataValidator:
    """Validate processed data quality."""

    def __init__(self, input_path: Path = DATA_DIR / "games_engineered.csv"):
        self.input_path = input_path

    def run(self) -> dict[str, Any]:
        """Execute data validation."""
        ensure_dirs()
        log_message(f"Walidacja danych: {self.input_path}")
        df = pd.read_csv(self.input_path, low_memory=False)
        numeric = df.select_dtypes(include=[np.number])
        outliers = []
        for column in numeric.columns:
            q1 = numeric[column].quantile(0.25)
            q3 = numeric[column].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                continue
            mask = (numeric[column] < q1 - 1.5 * iqr) | (numeric[column] > q3 + 1.5 * iqr)
            outliers.append({"column": column, "outlier_count": int(mask.sum()), "outlier_percent": float(mask.mean() * 100)})
        outlier_df = pd.DataFrame(outliers, columns=["column", "outlier_count", "outlier_percent"])
        if not outlier_df.empty:
            outlier_df = outlier_df.sort_values("outlier_count", ascending=False)
        outlier_df.to_csv(REPORTS_DIR / "outlier_report_iqr.csv", index=False)
        report = {
            "timestamp": timestamp(),
            "shape": list(df.shape),
            "missing_values": df.isna().sum().loc[lambda s: s > 0].to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "numeric_columns": len(numeric.columns),
            "text_columns": len(df.select_dtypes(include=["object"]).columns),
            "target_distribution": df[TARGET].value_counts().to_dict() if TARGET in df.columns else {},
        }
        save_json(report, REPORTS_DIR / "02_validation_report.json")
        log_message(
            f"Walidacja zakonczona: missing kolumn={len(report['missing_values'])}, duplikaty={report['duplicates']}, outlier raport={len(outliers)} kolumn"
        )
        return report


# ============================================================================
# EXPLORATION & ANALYSIS FUNCTIONS
# ============================================================================

def run_exploration(source: Path | None = None) -> dict[str, Any]:
    """Run data exploration on raw dataset."""
    ensure_dirs()
    log_message("Eksploracja danych surowych")
    df = load_raw_data(source)
    report = {
        "timestamp": timestamp(),
        "source": str(source or latest_raw_csv()),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_values": df.isna().sum().loc[lambda s: s > 0].to_dict(),
        "numeric_summary": df.describe(include=[np.number]).to_dict(),
    }
    save_json(report, REPORTS_DIR / "01_exploration_summary.json")

    plot_df = normalize_column_names(df.copy())
    if "Price" in plot_df.columns:
        plot_df["Price"] = pd.to_numeric(plot_df["Price"], errors="coerce").fillna(0).clip(0, 200)
        plt.figure(figsize=(10, 6))
        sns.histplot(plot_df["Price"], bins=50)
        plt.title("Price distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "price_distribution.png", dpi=150)
        plt.close()
    log_message(f"Eksploracja zakonczona: shape={report['shape']}, raport=reports/01_exploration_summary.json")
    return report


def comprehensive_analysis(input_path: Path = PROCESSED_DIR / "games_final.csv") -> dict[str, Any]:
    """Perform comprehensive analysis of processed dataset."""
    ensure_dirs()
    log_message(f"Analiza kompleksowa: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    if "Platform_count" in df.columns:
        engineered_path = DATA_DIR / "games_engineered.csv"
        if engineered_path.exists() and "AppID" in df.columns:
            platform_cols = ["AppID", "Has_windows", "Has_mac", "Has_linux"]
            engineered_cols = pd.read_csv(engineered_path, nrows=0).columns
            if all(column in engineered_cols for column in platform_cols):
                platforms = pd.read_csv(engineered_path, usecols=platform_cols)
                df = df.merge(platforms.drop_duplicates("AppID"), on="AppID", how="left")
                df["Platform_label"] = df.apply(platform_label, axis=1)
        if "Platform_label" not in df.columns:
            df["Platform_label"] = df["Platform_count"].map(
                {1: "1 platform", 2: "2 platforms", 3: "Windows + Mac + Linux"}
            ).fillna("Unknown")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(numeric_only=True)
    strong = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) > 0.7:
                strong.append({"left": left, "right": right, "correlation": float(value)})

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(x=TARGET, data=df)
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distributions_target.png", dpi=150)
    plt.close()

    features = [c for c in MODEL_FEATURES if c in df.columns][:9]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, column in zip(axes.ravel(), features):
        if column == "Platform_count" and "Platform_label" in df.columns:
            order = df["Platform_label"].value_counts().index.tolist()
            sns.countplot(data=df, y="Platform_label", order=order, ax=ax)
            ax.set_xlabel("Games")
            ax.set_ylabel("")
            ax.set_title("Platforms")
        else:
            sns.histplot(df[column], ax=ax, bins=40)
            ax.set_title(column)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distributions_features.png", dpi=150)
    plt.close()

    report = {
        "timestamp": timestamp(),
        "shape": list(df.shape),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().loc[lambda s: s > 0].to_dict(),
        "target_distribution": df[TARGET].value_counts(normalize=False).to_dict(),
        "target_percent": (df[TARGET].value_counts(normalize=True) * 100).round(2).to_dict(),
        "strong_correlations_count": len(strong),
        "strong_correlations_top": sorted(strong, key=lambda x: abs(x["correlation"]), reverse=True)[:25],
    }
    save_json(report, REPORTS_DIR / "comprehensive_analysis.json")
    log_message(
        f"Analiza kompleksowa zakonczona: target={report['target_percent']}, silne korelacje={report['strong_correlations_count']}"
    )
    return report


# ============================================================================
# MODEL TRAINING & EVALUATION CLASSES
# ============================================================================

@dataclass
class PreparedData:
    """Container for prepared training data."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler


class AdvancedModelTrainer:
    """Train and compare multiple ML models."""

    def prepare_data(self) -> PreparedData:
        """Prepare data for training."""
        train = pd.read_csv(PROCESSED_DIR / "games_train.csv")
        val = pd.read_csv(PROCESSED_DIR / "games_val.csv")
        test = pd.read_csv(PROCESSED_DIR / "games_test.csv")
        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(train[MODEL_FEATURES].fillna(0)), columns=MODEL_FEATURES)
        x_val = pd.DataFrame(scaler.transform(val[MODEL_FEATURES].fillna(0)), columns=MODEL_FEATURES)
        x_test = pd.DataFrame(scaler.transform(test[MODEL_FEATURES].fillna(0)), columns=MODEL_FEATURES)
        return PreparedData(train, val, test, x_train, x_val, x_test, train[TARGET], val[TARGET], test[TARGET], scaler)

    def define_models(self) -> dict[str, Any]:
        """Define ML models to train."""
        models: dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=15, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32, 16), alpha=0.0005, early_stopping=True, max_iter=120, random_state=RANDOM_STATE),
        }
        try:
            from lightgbm import LGBMClassifier

            models["LightGBM"] = LGBMClassifier(n_estimators=120, learning_rate=0.1, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        except Exception:
            pass
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss")
        except Exception:
            pass
        return models

    def run(self) -> dict[str, Any]:
        """Execute model training."""
        ensure_dirs()
        log_message("Przygotowanie danych do treningu")
        data = self.prepare_data()
        log_message(
            f"Dane treningowe: train={len(data.train):,}, val={len(data.val):,}, test={len(data.test):,}, cechy={len(MODEL_FEATURES)}".replace(",", " ")
        )
        if data.y_train.nunique() < 2:
            raise RuntimeError("Trening wymaga co najmniej dwoch klas docelowych. Sprawdz TARGET_MODE lub dane.")
        weights = compute_sample_weight("balanced", data.y_train)
        results: dict[str, Any] = {}
        trained: dict[str, Any] = {}
        models = self.define_models()
        log_message(f"Modele do treningu: {', '.join(models.keys())}")
        for name, model in models.items():
            log_message(f"Start treningu modelu: {name}")
            start = time.time()
            try:
                try:
                    model.fit(data.x_train, data.y_train, sample_weight=weights)
                except TypeError:
                    model.fit(data.x_train, data.y_train)
                y_pred = model.predict(data.x_test)
                y_proba = self._predict_proba(model, data.x_test)
                results[name] = self.metrics(data.y_test, y_pred, y_proba, time.time() - start)
                trained[name] = model
                row = results[name]
                log_message(
                    f"{name}: ROC-AUC={row['roc_auc']:.4f}, F1={row['f1_score']:.4f}, ACC={row['accuracy']:.4f}, czas={format_seconds(row['training_time_seconds'])}"
                )
            except Exception as exc:
                results[name] = {"error": str(exc), "training_time_seconds": time.time() - start}
                log_message(f"{name}: BLAD treningu: {exc}")

        scored = {name: row for name, row in results.items() if "roc_auc" in row}
        if not scored:
            raise RuntimeError("Zaden model nie zakonczyl treningu poprawnie.")
        best_name = max(scored, key=lambda name: scored[name]["roc_auc"])
        artifact = {
            "model": trained[best_name],
            "scaler": data.scaler,
            "feature_names": MODEL_FEATURES,
            "target": TARGET,
            "best_model_name": best_name,
            "metrics": scored[best_name],
        }
        joblib.dump(artifact, MODELS_DIR / "best_model.joblib")
        joblib.dump(trained, MODELS_DIR / "all_models.joblib")
        report = {
            "timestamp": timestamp(),
            "data_info": {"train_size": len(data.train), "val_size": len(data.val), "test_size": len(data.test), "n_features": len(MODEL_FEATURES)},
            "class_distribution": data.train[TARGET].value_counts().to_dict(),
            "best_model": best_name,
            "models_performance": results,
        }
        save_json(report, REPORTS_DIR / "07_model_training_advanced.json")
        self.plot_model_comparison(results)
        log_message(f"Najlepszy model: {best_name} (ROC-AUC={scored[best_name]['roc_auc']:.4f})")
        return report

    @staticmethod
    def _predict_proba(model: Any, x_test: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from model."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test)
            if proba.ndim == 1:
                return proba
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        if hasattr(model, "decision_function"):
            values = model.decision_function(x_test)
            return values[:, 1] if getattr(values, "ndim", 1) > 1 else values
        return np.zeros(len(x_test), dtype=float)

    @staticmethod
    def metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, elapsed: float) -> dict[str, float]:
        """Calculate model metrics."""
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc_auc = float("nan")
        try:
            average_precision = float(average_precision_score(y_true, y_proba))
        except Exception:
            average_precision = float("nan")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "training_time_seconds": float(elapsed),
        }

    @staticmethod
    def plot_model_comparison(results: dict[str, Any]) -> None:
        """Plot model comparison charts."""
        rows = [{"model": k, **v} for k, v in results.items() if "roc_auc" in v]
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values("roc_auc", ascending=True)
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision", "training_time_seconds"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, metric in zip(axes.ravel(), metrics):
            sns.barplot(data=df, x=metric, y="model", ax=ax)
            ax.set_title(metric)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "models_comparison.png", dpi=150)
        plt.close()

        plt.figure(figsize=(9, 5))
        sns.barplot(data=df, x="roc_auc", y="model", hue="model", palette="viridis", legend=False)
        plt.title("ROC-AUC comparison")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "roc_auc_comparison.png", dpi=150)
        plt.close()


class AdvancedModelEvaluator:
    """Evaluate trained ML models."""

    def run(self) -> dict[str, Any]:
        """Execute model evaluation."""
        ensure_dirs()
        log_message("Ewaluacja najlepszego modelu")
        artifact = joblib.load(MODELS_DIR / "best_model.joblib")
        log_message(f"Zaladowano model: {artifact['best_model_name']}")
        test = pd.read_csv(PROCESSED_DIR / "games_test.csv")
        x_test = pd.DataFrame(
            artifact["scaler"].transform(test[artifact["feature_names"]].fillna(0)),
            columns=artifact["feature_names"],
        )
        y_true = test[TARGET]
        model = artifact["model"]
        y_pred = model.predict(x_test)
        y_proba = AdvancedModelTrainer._predict_proba(model, x_test)

        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm)
        self.plot_roc_pr(y_true, y_proba)
        self.plot_feature_importance(model, artifact["feature_names"])
        self.plot_prediction_distribution(y_true, y_proba)
        self.plot_error_analysis(y_true, y_pred, y_proba)
        report = {
            "timestamp": timestamp(),
            "best_model": artifact["best_model_name"],
            "metrics": AdvancedModelTrainer.metrics(y_true, y_pred, y_proba, 0.0),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }
        save_json(report, REPORTS_DIR / "08_model_evaluation_advanced.json")
        metrics = report["metrics"]
        log_message(
            f"Ewaluacja zakonczona: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}, confusion_matrix={report['confusion_matrix']}"
        )
        return report

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion matrix")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "confusion_matrix_advanced.png", dpi=150)
        plt.close()

    @staticmethod
    def plot_roc_pr(y_true: pd.Series, y_proba: np.ndarray) -> None:
        """Plot ROC and Precision-Recall curves."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], linestyle="--")
        axes[0].set_title("ROC curve")
        axes[1].plot(recall, precision)
        axes[1].set_title("Precision-Recall curve")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "roc_pr_curves_advanced.png", dpi=150)
        plt.close()

    @staticmethod
    def plot_feature_importance(model: Any, features: list[str]) -> None:
        """Plot feature importance."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            return
        df = pd.DataFrame({"feature": features, "importance": importance}).sort_values("importance", ascending=False).head(15)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=df, x="importance", y="feature")
        plt.title("Feature importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance_advanced.png", dpi=150)
        plt.close()

    @staticmethod
    def plot_prediction_distribution(y_true: pd.Series, y_proba: np.ndarray) -> None:
        """Plot prediction probability distributions."""
        df = pd.DataFrame({"target": y_true, "probability": y_proba})
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(data=df, x="probability", hue="target", bins=40, ax=axes[0])
        sns.boxplot(data=df, x="target", y="probability", ax=axes[1])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "prediction_distributions.png", dpi=150)
        plt.close()

    @staticmethod
    def plot_error_analysis(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
        """Plot error analysis."""
        labels = np.where(
            (y_true == 1) & (y_pred == 1),
            "TP",
            np.where((y_true == 0) & (y_pred == 0), "TN", np.where(y_pred == 1, "FP", "FN")),
        )
        df = pd.DataFrame({"error_type": labels, "probability": y_proba})
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(data=df, x="error_type", order=["TP", "TN", "FP", "FN"], ax=axes[0])
        sns.boxplot(data=df, x="error_type", y="probability", order=["TP", "TN", "FP", "FN"], ax=axes[1])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "error_analysis.png", dpi=150)
        plt.close()


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def run_full_pipeline(train_models: bool = True) -> None:
    """Run complete ML pipeline."""
    pipeline_start = time.time()
    ensure_dirs()
    log_message("=" * 78)
    log_message("START PIPELINE ML-PROJECT-STEAM")
    log_message(f"Tryb: {'pelny trening + ewaluacja' if train_models else 'przygotowanie danych bez treningu'}")
    stages: list[tuple[str, Callable[[], Any]]] = [
        ("Eksploracja danych surowych", run_exploration),
        ("Czyszczenie danych", lambda: DataCleaner().run()),
        ("Inzynieria cech", lambda: FeatureEngineer().run()),
        ("Walidacja danych po feature engineeringu", lambda: DataValidator().run()),
        ("Eksport i podzial train/val/test", lambda: DataExporter().run()),
        ("Analiza kompleksowa gotowego datasetu", comprehensive_analysis),
    ]
    if train_models:
        stages.extend(
            [
                ("Trening i porownanie modeli", lambda: AdvancedModelTrainer().run()),
                ("Ewaluacja najlepszego modelu", lambda: AdvancedModelEvaluator().run()),
            ]
        )

    try:
        source = latest_raw_csv()
        log_message(f"Znaleziono lokalny surowy CSV: {source}")
    except FileNotFoundError:
        stages.insert(0, ("Pobieranie danych z Kaggle", download_kaggle_dataset))

    total = len(stages)
    for index, (name, fn) in enumerate(stages, start=1):
        run_pipeline_stage(index, total, name, fn)

    log_message(f"KONIEC PIPELINE. Czas calkowity: {format_seconds(time.time() - pipeline_start)}")
    log_message("=" * 78)


def summarize_result(result: Any) -> str:
    """Summarize result from pipeline stage."""
    if isinstance(result, pd.DataFrame):
        return f"DataFrame: {len(result):,} wierszy, {len(result.columns)} kolumn".replace(",", " ")
    if not isinstance(result, dict):
        return "OK"
    if "output_shape" in result:
        return f"output_shape={result['output_shape']}"
    if "shape" in result:
        extra = ""
        if "target_percent" in result:
            extra = f", target%={result['target_percent']}"
        return f"shape={result['shape']}{extra}"
    if "split" in result:
        return f"split={result['split']}, klasy={result.get('class_distribution', {})}"
    if "best_model" in result and "models_performance" in result:
        best = result["best_model"]
        metrics = result["models_performance"].get(best, {})
        return f"best_model={best}, roc_auc={metrics.get('roc_auc')}, f1={metrics.get('f1_score')}"
    if "metrics" in result:
        metrics = result["metrics"]
        return f"roc_auc={metrics.get('roc_auc')}, f1={metrics.get('f1_score')}, accuracy={metrics.get('accuracy')}"
    if "source" in result:
        return f"source={result['source']}"
    return json.dumps(result, ensure_ascii=False)[:500]


def run_pipeline_stage(index: int, total: int, name: str, fn: Callable[[], Any]) -> Any:
    """Run a single pipeline stage."""
    log_message("-" * 78)
    log_message(f"[{index}/{total}] START: {name}")
    start = time.time()
    try:
        result = fn()
    except Exception as exc:
        log_message(f"[{index}/{total}] BLAD: {name}: {exc}")
        raise
    elapsed = time.time() - start
    log_message(f"[{index}/{total}] KONIEC: {name} ({format_seconds(elapsed)})")
    log_message(f"[{index}/{total}] WYNIK: {summarize_result(result)}")
    return result


def predict_from_values(values: dict[str, float]) -> dict[str, Any]:
    """Make prediction from feature values."""
    artifact = joblib.load(MODELS_DIR / "best_model.joblib")
    row = pd.DataFrame([{feature: values.get(feature, 0.0) for feature in artifact["feature_names"]}])
    x = pd.DataFrame(artifact["scaler"].transform(row), columns=artifact["feature_names"])
    probability = float(artifact["model"].predict_proba(x)[0, 1])
    return {
        "prediction": int(probability >= 0.5),
        "label": "WYSOKA OCENA" if probability >= 0.5 else "NISKA/SREDNIA OCENA",
        "probability": probability,
        "model": artifact["best_model_name"],
    }


def run_step(name: str, fn: Callable[[], Any]) -> None:
    """Run a single pipeline step."""
    result = run_pipeline_stage(1, 1, name, fn)
    if isinstance(result, dict):
        log_message("Pelny wynik etapu:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:3000], flush=True)


# ============================================================================
# INTERACTIVE CLI PREDICTION
# ============================================================================

def read_float(feature: str) -> float:
    """Read float value from user input."""
    raw = input(f"{PREDICTION_PROMPTS.get(feature, feature)}: ").strip().replace(",", ".")
    return float(raw) if raw else 0.0


def interactive_predict() -> None:
    """Run interactive prediction CLI."""
    print("\n" + "=" * 78)
    print("PREDYKCJA SUKCESU GRY STEAM")
    print("=" * 78 + "\n")
    try:
        values = {feature: read_float(feature) for feature in MODEL_FEATURES}
        result = predict_from_values(values)
        print(f"\n{'-' * 78}")
        print(f"Model: {result['model']}")
        print(f"Wynik: {result['label']}")
        print(f"Prawdopodobienstwo wysokiej oceny: {result['probability']:.2%}")
        print("-" * 78 + "\n")
    except FileNotFoundError:
        print("Błąd: Brak wytrenowanego modelu. Uruchom najpierw 'python steam_ml_unified_pipeline.py full'")
    except Exception as e:
        print(f"Błąd: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def print_usage() -> None:
    """Print usage information."""
    print("\n" + "=" * 78)
    print("STEAM GAMES ML PREDICTION PIPELINE - UNIFIED SCRIPT")
    print("=" * 78)
    print("\nUSAGE: python steam_ml_unified_pipeline.py [mode]\n")
    print("MODES:")
    print("  full          - Complete pipeline with training (default)")
    print("  prepare       - Data preparation only (no model training)")
    print("  predict       - Interactive prediction CLI")
    print("  exploration   - Data exploration only")
    print("  cleaning      - Data cleaning only")
    print("  training      - Model training only (requires prepared data)")
    print("  evaluation    - Model evaluation only (requires trained model)")
    print("=" * 78 + "\n")


def main() -> None:
    """Main entry point."""
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "full"

    if mode == "full":
        run_full_pipeline(train_models=True)
    elif mode == "prepare":
        run_full_pipeline(train_models=False)
    elif mode == "predict":
        interactive_predict()
    elif mode == "exploration":
        run_step("Eksploracja danych surowych", run_exploration)
    elif mode == "cleaning":
        run_step("Czyszczenie danych", lambda: DataCleaner().run())
    elif mode == "training":
        ensure_dirs()
        log_message("=" * 78)
        log_message("TRENING MODELI")
        run_pipeline_stage(1, 2, "Trening i porownanie modeli", lambda: AdvancedModelTrainer().run())
        log_message("=" * 78)
    elif mode == "evaluation":
        ensure_dirs()
        log_message("=" * 78)
        log_message("EWALUACJA MODELU")
        run_pipeline_stage(1, 2, "Ewaluacja najlepszego modelu", lambda: AdvancedModelEvaluator().run())
        log_message("=" * 78)
    else:
        print(f"Nieznany tryb: {mode}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
