"""Clean GDELT articles and create hourly news features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


REQUIRED_NEWS_COLUMNS = {
    "timestamp",
    "title",
}


def load_news_files(
    input_dir: str | Path = "data/raw/news",
) -> pd.DataFrame:
    """Load and combine all news CSV files from a directory."""

    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No news CSV files were found in {input_path.resolve()}."
        )

    frames: list[pd.DataFrame] = []

    for file_path in files:
        frame = pd.read_csv(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    missing_columns = REQUIRED_NEWS_COLUMNS - set(combined.columns)

    if missing_columns:
        raise ValueError(
            "News data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return combined


def score_sentiment(
    news_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add VADER sentiment scores to article titles."""

    frame = news_data.copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    frame["title"] = frame["title"].fillna("").astype(str)

    frame = frame.dropna(subset=["timestamp"])
    frame = frame[frame["title"].str.strip().ne("")]

    if "url" in frame.columns:
        frame = frame.drop_duplicates(
            subset=["url"],
            keep="last",
        )
    else:
        frame = frame.drop_duplicates(
            subset=["timestamp", "title"],
            keep="last",
        )

    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = frame["title"].apply(
        analyzer.polarity_scores
    )

    frame["sentiment_negative"] = sentiment_scores.apply(
        lambda score: score["neg"]
    )

    frame["sentiment_neutral"] = sentiment_scores.apply(
        lambda score: score["neu"]
    )

    frame["sentiment_positive"] = sentiment_scores.apply(
        lambda score: score["pos"]
    )

    frame["sentiment_compound"] = sentiment_scores.apply(
        lambda score: score["compound"]
    )

    frame["is_positive"] = (
        frame["sentiment_compound"] >= 0.05
    ).astype(int)

    frame["is_negative"] = (
        frame["sentiment_compound"] <= -0.05
    ).astype(int)

    frame["is_neutral"] = (
        frame["sentiment_compound"].between(
            -0.05,
            0.05,
            inclusive="neither",
        )
    ).astype(int)

    return frame.sort_values("timestamp").reset_index(drop=True)


def aggregate_hourly_news(
    scored_news: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate articles by publication hour.

    News published during an hour becomes available at the beginning
    of the next hour. This helps prevent look-ahead bias.
    """

    frame = scored_news.copy()

    frame["publication_hour"] = frame["timestamp"].dt.floor("h")

    aggregation: dict[str, tuple[str, str]] = {
        "article_count": ("title", "count"),
        "sentiment_mean": ("sentiment_compound", "mean"),
        "sentiment_std": ("sentiment_compound", "std"),
        "sentiment_min": ("sentiment_compound", "min"),
        "sentiment_max": ("sentiment_compound", "max"),
        "positive_article_share": ("is_positive", "mean"),
        "negative_article_share": ("is_negative", "mean"),
        "neutral_article_share": ("is_neutral", "mean"),
    }

    if "source" in frame.columns:
        aggregation["unique_source_count"] = (
            "source",
            "nunique",
        )

    hourly = (
        frame.groupby("publication_hour", as_index=False)
        .agg(**aggregation)
        .sort_values("publication_hour")
    )

    hourly["sentiment_std"] = hourly["sentiment_std"].fillna(0.0)

    # News from 10:00 through 10:59 can first be used at 11:00.
    hourly["available_at"] = (
        hourly["publication_hour"] + pd.Timedelta(hours=1)
    )

    # Rolling news-volume and sentiment context
    hourly["article_count_rolling_6h"] = (
        hourly["article_count"]
        .rolling(window=6, min_periods=1)
        .sum()
    )

    hourly["sentiment_rolling_mean_6h"] = (
        hourly["sentiment_mean"]
        .rolling(window=6, min_periods=1)
        .mean()
    )

    hourly["news_volume_change"] = hourly["article_count"].pct_change()

    hourly = hourly.replace([np.inf, -np.inf], np.nan)

    return hourly.reset_index(drop=True)


def process_news_data(
    input_dir: str | Path = "data/raw/news",
    article_output_path: str | Path = (
        "data/processed/features/scored_news.csv"
    ),
    hourly_output_path: str | Path = (
        "data/processed/features/hourly_news_features.csv"
    ),
) -> tuple[Path, Path]:
    """Load, score, aggregate, and save news features."""

    raw_news = load_news_files(input_dir)
    scored_news = score_sentiment(raw_news)
    hourly_news = aggregate_hourly_news(scored_news)

    article_destination = Path(article_output_path)
    hourly_destination = Path(hourly_output_path)

    article_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored_news.to_csv(article_destination, index=False)
    hourly_news.to_csv(hourly_destination, index=False)

    print(f"Scored {len(scored_news):,} news articles.")
    print(f"Created {len(hourly_news):,} hourly news rows.")
    print(f"Saved scored articles to {article_destination}")
    print(f"Saved hourly features to {hourly_destination}")

    return article_destination, hourly_destination