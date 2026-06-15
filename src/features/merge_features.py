import pandas as pd


def create_daily_labels(market_df: pd.DataFrame) -> pd.DataFrame:
    df = market_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["next_close"] = df["Close"].shift(-1)
    df["target_green_tomorrow"] = (df["next_close"] > df["Close"]).astype(int)

    return df.dropna(subset=["next_close"])