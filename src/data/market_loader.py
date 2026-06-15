import yfinance as yf
from pathlib import Path


def download_market_data(symbol: str, start: str, end: str, output_path: str) -> None:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    df = df.reset_index()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    download_market_data(
        symbol="SPY",
        start="2018-01-01",
        end="2026-01-01",
        output_path="data/raw/market/spy_daily.csv",
    )