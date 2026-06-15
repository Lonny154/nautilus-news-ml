"""Run the complete market-news feature engineering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.market_features import process_market_data
from src.features.merge_features import save_combined_features
from src.features.news_features import process_news_data


def validate_combined_features(
    file_path: str | Path,
) -> None:
    """Run basic validation checks on the final feature dataset."""

    frame = pd.read_csv(file_path)

    required_columns = {
        "timestamp",
        "symbol",
        "close",
        "return_1h",
        "article_count",
        "sentiment_mean",
        "future_return_1h",
        "target_up_1h",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            "Combined dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    timestamps = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Combined dataset contains invalid timestamps."
        )

    duplicate_count = frame.duplicated(
        subset=["symbol", "timestamp"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Combined dataset contains {duplicate_count} "
            "duplicate symbol-timestamp rows."
        )

    print("\nValidation summary")
    print("------------------")
    print(f"Rows: {len(frame):,}")
    print(f"Symbols: {frame['symbol'].nunique():,}")
    print(f"Start: {timestamps.min()}")
    print(f"End: {timestamps.max()}")
    print(
        "Rows with news:",
        f"{(frame['article_count'] > 0).sum():,}",
    )
    print(
        "Average articles per populated bar:",
        round(
            frame.loc[
                frame["article_count"] > 0,
                "article_count",
            ].mean(),
            2,
        ),
    )
    print(
        "Rows with usable target:",
        f"{frame['future_return_1h'].notna().sum():,}",
    )


def run_pipeline() -> Path:
    """Run all feature engineering stages."""

    print("Step 1: Processing market data")
    process_market_data()

    print("\nStep 2: Processing news data")
    process_news_data()

    print("\nStep 3: Merging market and news features")
    combined_path = save_combined_features()

    print("\nStep 4: Validating final dataset")
    validate_combined_features(combined_path)

    return combined_path


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Build combined market and news features."
        )
    )

    parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()