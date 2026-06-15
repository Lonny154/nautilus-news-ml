"""Merge processed market bars with delayed hourly news features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NEWS_FEATURE_COLUMNS = [
    "article_count",
    "sentiment_mean",
    "sentiment_std",
    "sentiment_min",
    "sentiment_max",
    "positive_article_share",
    "negative_article_share",
    "neutral_article_share",
    "unique_source_count",
    "article_count_rolling_6h",
    "sentiment_rolling_mean_6h",
    "news_volume_change",
]


def merge_market_and_news(
    market_data: pd.DataFrame,
    news_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge each market bar with the latest news available at that time.

    The news processor sets available_at to one hour after the
    publication bucket, preventing future news from leaking backward.
    """

    market = market_data.copy()
    news = news_data.copy()

    market["timestamp"] = pd.to_datetime(
        market["timestamp"],
        utc=True,
        errors="coerce",
    )

    news["available_at"] = pd.to_datetime(
        news["available_at"],
        utc=True,
        errors="coerce",
    )

    market = market.dropna(subset=["timestamp"])
    news = news.dropna(subset=["available_at"])

    market = market.sort_values("timestamp")
    news = news.sort_values("available_at")

    combined = pd.merge_asof(
        market,
        news,
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )

    existing_news_columns = [
        column
        for column in NEWS_FEATURE_COLUMNS
        if column in combined.columns
    ]

    # No earlier news means no news was known yet.
    combined[existing_news_columns] = combined[
        existing_news_columns
    ].fillna(0.0)

    combined["has_news"] = (
        combined.get("article_count", 0) > 0
    ).astype(int)

    combined["hours_since_news_bucket"] = (
        combined["timestamp"] - combined["available_at"]
    ).dt.total_seconds() / 3600

    combined["hours_since_news_bucket"] = (
        combined["hours_since_news_bucket"]
        .fillna(-1.0)
    )

    combined = combined.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    return combined


def save_combined_features(
    market_path: str | Path = (
        "data/processed/features/market_features.csv"
    ),
    news_path: str | Path = (
        "data/processed/features/hourly_news_features.csv"
    ),
    output_path: str | Path = (
        "data/processed/features/combined_features.csv"
    ),
) -> Path:
    """Load processed features, merge them, and save the result."""

    market = pd.read_csv(market_path)
    news = pd.read_csv(news_path)

    combined = merge_market_and_news(market, news)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(destination, index=False)

    print(f"Created {len(combined):,} combined feature rows.")
    print(f"Saved combined features to {destination}")

    return destination