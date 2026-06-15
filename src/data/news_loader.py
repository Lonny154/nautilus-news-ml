from pathlib import Path
import pandas as pd


def load_news_csv(path: str = "data/raw/news/world_news.csv") -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"News file not found: {path}")

    df = pd.read_csv(path)

    required_cols = ["timestamp", "source", "title"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "title"])

    df["date"] = df["timestamp"].dt.date
    df["title"] = df["title"].astype(str)

    if "description" not in df.columns:
        df["description"] = ""

    if "url" not in df.columns:
        df["url"] = ""

    return df


if __name__ == "__main__":
    news = load_news_csv()
    print(news.head())
    print(news.info())