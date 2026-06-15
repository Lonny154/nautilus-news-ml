"""Download real news articles from the GDELT DOC 2.0 API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

import re

import time

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def _create_retrying_session() -> requests.Session:
    """Create an HTTP session that retries temporary API failures."""

    retry = Retry(
        total=5,
        connect=3,
        read=3,
        status=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "nautilus-news-ml/0.1 "
                "(educational market-news research project)"
            )
        }
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

def fetch_gdelt_news(
    query: str,
    start: str,
    end: str,
    max_records: int = 100,
    output_dir: str | Path = "data/raw/news",
) -> Path:
    """Fetch matching articles from GDELT and save them as CSV."""

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
        "startdatetime": _format_gdelt_datetime(start),
        "enddatetime": _format_gdelt_datetime(end),
    }

    session = _create_retrying_session()

    response = session.get(
        GDELT_DOC_URL,
        params=params,
        timeout=60,
    )   

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "not provided")
        raise RuntimeError(
            "GDELT rate limit remained active after retries. "
            f"Retry-After header: {retry_after}. "
         "Run the request again later or reduce max_records."
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_preview = response.text[:500]
        raise RuntimeError(
            f"GDELT request failed with HTTP {response.status_code}. "
            f"Response: {response_preview}"
        ) from exc

    payload: dict[str, Any] = response.json()
    articles = payload.get("articles", [])

    if not articles:
        raise ValueError(
            f"GDELT returned no articles for query: {query!r}"
        )

    rows = []

    for article in articles:
        rows.append(
            {
                "timestamp": article.get("seendate"),
                "source": article.get("domain"),
                "title": article.get("title"),
                "url": article.get("url"),
                "language": article.get("language"),
                "source_country": article.get("sourcecountry"),
                "query": query,
            }
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    frame = frame.dropna(subset=["timestamp", "title"])
    frame = frame.drop_duplicates(subset=["url"])
    frame = frame.sort_values("timestamp")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_query = (
        query.lower()
        .replace(" ", "_")
        .replace('"', "")
        .replace("(", "")
        .replace(")", "")
    )

    file_path = output_path / f"gdelt_{safe_query}_{start}_{end}.csv"
    frame.to_csv(file_path, index=False)

    print(f"Downloaded {len(frame):,} news articles.")
    print(f"Saved news data to {file_path}")

    return file_path


def _format_gdelt_datetime(value: str) -> str:
    """Convert YYYY-MM-DD into GDELT's YYYYMMDDHHMMSS format."""

    cleaned = value.replace("-", "").replace(":", "").replace("T", "")
    cleaned = cleaned.replace("Z", "").replace(" ", "")

    if len(cleaned) == 8:
        cleaned += "000000"

    if len(cleaned) != 14:
        raise ValueError(
            "GDELT dates must use YYYY-MM-DD or a full UTC datetime."
        )

    return cleaned