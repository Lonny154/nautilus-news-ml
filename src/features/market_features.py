"""Clean raw market data and create market-derived features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_MARKET_COLUMNS = {
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_market_files(
    input_dir: str | Path = "data/raw/market",
) -> pd.DataFrame:
    """Load and combine all market CSV files from a directory."""

    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No market CSV files were found in {input_path.resolve()}."
        )

    frames: list[pd.DataFrame] = []

    for file_path in files:
        frame = pd.read_csv(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    missing_columns = REQUIRED_MARKET_COLUMNS - set(combined.columns)

    if missing_columns:
        raise ValueError(
            "Market data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return combined


def build_market_features(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Clean market bars and calculate time-series features."""

    frame = market_data.copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame = frame.dropna(
        subset=[
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    frame["symbol"] = frame["symbol"].astype(str).str.upper()

    frame = frame.sort_values(["symbol", "timestamp"])
    frame = frame.drop_duplicates(
        subset=["symbol", "timestamp"],
        keep="last",
    )

    grouped = frame.groupby("symbol", group_keys=False)

    # Price-change features
    frame["return_1h"] = grouped["close"].pct_change()
    frame["return_3h"] = grouped["close"].pct_change(periods=3)
    frame["return_6h"] = grouped["close"].pct_change(periods=6)

    frame["log_return_1h"] = grouped["close"].transform(
        lambda values: np.log(values / values.shift(1))
    )

    # Intrabar price behavior
    frame["high_low_range"] = (
        frame["high"] - frame["low"]
    ) / frame["open"]

    frame["open_close_return"] = (
        frame["close"] - frame["open"]
    ) / frame["open"]

    # Volume features
    frame["volume_change_1h"] = grouped["volume"].pct_change()

    frame["volume_rolling_mean_6h"] = grouped["volume"].transform(
        lambda values: values.rolling(
            window=6,
            min_periods=1,
        ).mean()
    )

    frame["relative_volume_6h"] = (
        frame["volume"] / frame["volume_rolling_mean_6h"]
    )

    # Rolling volatility
    frame["volatility_6h"] = grouped["return_1h"].transform(
        lambda values: values.rolling(
            window=6,
            min_periods=2,
        ).std()
    )

    frame["volatility_12h"] = grouped["return_1h"].transform(
        lambda values: values.rolling(
            window=12,
            min_periods=2,
        ).std()
    )

    # Calendar features
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek

    # Prediction target:
    # return from the current close to the next bar close
    frame["future_return_1h"] = grouped["close"].shift(-1) / frame["close"] - 1

    frame["target_up_1h"] = np.where(
        frame["future_return_1h"].isna(),
        np.nan,
        (frame["future_return_1h"] > 0).astype(int),
    )

    frame = frame.replace([np.inf, -np.inf], np.nan)

    return frame.reset_index(drop=True)


def process_market_data(
    input_dir: str | Path = "data/raw/market",
    output_path: str | Path = (
        "data/processed/features/market_features.csv"
    ),
) -> Path:
    """Load, transform, and save processed market features."""

    raw_market = load_market_files(input_dir)
    market_features = build_market_features(raw_market)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    market_features.to_csv(destination, index=False)

    print(f"Processed {len(market_features):,} market bars.")
    print(f"Saved market features to {destination}")

    return destination