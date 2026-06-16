from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .config import Company, ensure_parent_dir


HEADLINE_COLUMNS = [
    "headline_id",
    "source",
    "published_at_utc",
    "published_at_london",
    "ticker",
    "company_name",
    "headline_text",
    "mapping_confidence",
]


@dataclass(frozen=True)
class HeadlineRecord:
    headline_id: str
    source: str
    published_at_utc: str
    published_at_london: str
    ticker: str
    company_name: str
    headline_text: str
    mapping_confidence: str


def collect_gdelt_headlines(
    companies: Iterable[Company],
    start_date: date,
    end_date: date,
    output_path: str | Path,
    max_records_per_month: int = 250,
    pause_seconds: float = 0.25,
) -> list[HeadlineRecord]:
    records: list[HeadlineRecord] = []
    seen_ids: set[str] = set()

    for company in companies:
        for month_start, month_end in month_windows(start_date, end_date):
            articles = fetch_company_month(company, month_start, month_end, max_records_per_month)
            for article in articles:
                record = article_to_record(article, company)
                if record is None or record.headline_id in seen_ids:
                    continue
                seen_ids.add(record.headline_id)
                records.append(record)
            time.sleep(pause_seconds)

    write_headlines(records, output_path)
    return records


def fetch_company_month(company: Company, start_date: date, end_date: date, max_records: int) -> list[dict]:
    alias_query = " OR ".join(f'"{alias}"' for alias in company.aliases)
    query = f"({alias_query})"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sourcelang": "english",
        "sort": "hybrid",
        "maxrecords": str(max_records),
        "startdatetime": gdelt_datetime(start_date, start=True),
        "enddatetime": gdelt_datetime(end_date, start=False),
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "GLM-Sentiment dissertation prototype"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return list(payload.get("articles", []))


def article_to_record(article: dict, company: Company) -> HeadlineRecord | None:
    title = str(article.get("title", "")).strip()
    if not title:
        return None

    published_utc = parse_gdelt_seen_date(str(article.get("seendate", "")))
    published_london = published_utc.astimezone(ZoneInfo("Europe/London"))
    source = str(article.get("domain", "") or article.get("sourceCountry", "") or "GDELT").strip()
    headline_id = stable_headline_id(company.ticker, published_utc.isoformat(), title)
    return HeadlineRecord(
        headline_id=headline_id,
        source=source,
        published_at_utc=published_utc.isoformat(),
        published_at_london=published_london.isoformat(),
        ticker=company.ticker,
        company_name=company.company_name,
        headline_text=title,
        mapping_confidence="high",
    )


def parse_gdelt_seen_date(value: str) -> datetime:
    cleaned = value.strip()
    formats = ("%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S")
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def stable_headline_id(ticker: str, published_at_utc: str, title: str) -> str:
    raw = f"{ticker}|{published_at_utc}|{title}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def gdelt_datetime(value: date, start: bool) -> str:
    suffix = "000000" if start else "235959"
    return value.strftime("%Y%m%d") + suffix


def month_windows(start_date: date, end_date: date) -> Iterable[tuple[date, date]]:
    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        window_start = max(current, start_date)
        window_end = min(next_month - timedelta(days=1), end_date)
        yield window_start, window_end
        current = next_month


def write_headlines(records: Iterable[HeadlineRecord], output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADLINE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
