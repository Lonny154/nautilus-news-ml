"""Download historical stock bars from Alpaca."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def fetch_alpaca_bars(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Hour",
    output_dir: str | Path = "data/raw/market",
) -> Path:
    """Download historical bars from Alpaca and save them as CSV."""

    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in the .env file."
        )

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    params: dict[str, Any] = {
        "symbols": symbol.upper(),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "adjustment": "all",
        "feed": "iex",
        "limit": 10_000,
        "sort": "asc",
    }

    rows: list[dict[str, Any]] = []

    while True:
        response = requests.get(
            ALPACA_BARS_URL,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        symbol_bars = payload.get("bars", {}).get(symbol.upper(), [])

        for bar in symbol_bars:
            rows.append(
                {
                    "timestamp": bar["t"],
                    "symbol": symbol.upper(),
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                    "trade_count": bar.get("n"),
                    "vwap": bar.get("vw"),
                }
            )

        next_page_token = payload.get("next_page_token")

        if not next_page_token:
            break

        params["page_token"] = next_page_token

    if not rows:
        raise ValueError(
            f"Alpaca returned no bars for {symbol} between {start} and {end}."
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["symbol", "timestamp"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_start = start[:10]
    safe_end = end[:10]
    file_path = (
        output_path
        / f"{symbol.upper()}_{timeframe}_{safe_start}_{safe_end}.csv"
    )

    frame.to_csv(file_path, index=False)

    print(f"Downloaded {len(frame):,} market bars.")
    print(f"Saved market data to {file_path}")

    return file_path