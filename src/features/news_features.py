import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_sentiment_scores(news_df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df = news_df.copy()

    df["text"] = (
        df["title"].fillna("") + " " + df.get("description", "").fillna("")
    )

    scores = df["text"].apply(analyzer.polarity_scores)
    df["sentiment_compound"] = scores.apply(lambda x: x["compound"])
    df["sentiment_positive"] = scores.apply(lambda x: x["pos"])
    df["sentiment_negative"] = scores.apply(lambda x: x["neg"])

    return df


def aggregate_news_daily(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby("date").agg(
        news_count=("title", "count"),
        avg_sentiment=("sentiment_compound", "mean"),
        min_sentiment=("sentiment_compound", "min"),
        max_sentiment=("sentiment_compound", "max"),
        negative_news_share=("sentiment_negative", "mean"),
        positive_news_share=("sentiment_positive", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    return daily