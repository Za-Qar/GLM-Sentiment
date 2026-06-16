from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from .config import Company, ensure_parent_dir


PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def collect_yfinance_prices(
    companies: Iterable[Company],
    start_date: date,
    end_date: date,
    output_path: str | Path,
):
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("pandas and yfinance are required for price collection. Install requirements.txt first.") from exc

    tickers = [company.ticker for company in companies]
    raw = yf.download(
        tickers=tickers,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    prices = normalize_yfinance_download(raw, tickers)
    ensure_parent_dir(output_path)
    prices.to_csv(output_path, index=False)
    return prices


def normalize_yfinance_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict] = []

    if raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            ticker_frame = raw[ticker].copy()
            rows.extend(frame_to_rows(ticker_frame, ticker))
    else:
        rows.extend(frame_to_rows(raw.copy(), tickers[0]))

    return pd.DataFrame(rows, columns=PRICE_COLUMNS)


def frame_to_rows(frame: pd.DataFrame, ticker: str) -> list[dict]:
    import pandas as pd

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    normalized = frame.rename(columns=rename_map)
    normalized = normalized.reset_index()
    date_column = "Date" if "Date" in normalized.columns else normalized.columns[0]

    rows: list[dict] = []
    for _, row in normalized.iterrows():
        rows.append(
            {
                "date": pd.to_datetime(row[date_column]).date().isoformat(),
                "ticker": ticker,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "adj_close": row.get("adj_close"),
                "volume": row.get("volume"),
            }
        )
    return rows
