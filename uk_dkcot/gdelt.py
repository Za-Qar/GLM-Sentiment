from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests

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
    """
    Args:
        headline_id          : stable identifier for deduplication and traceability.
        source               : GDELT source domain or country fallback.
        published_at_utc     : publication timestamp in UTC from GDELT.
        published_at_london  : same timestamp converted to UK market time.
        ticker               : yfinance ticker of the mapped company.
        company_name         : official company name used for reporting.
        headline_text        : headline text used as the sentiment-model input.
        mapping_confidence   : confidence flag for the company-headline mapping.
    """

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
    max_records_per_window: int = 250,
    pause_seconds: float = 30.0,
    months_per_window: int = 3,
) -> list[HeadlineRecord]:
    """
    Args:
        companies             : selected ten-company universe from the project config.
        start_date            : first date in the fixed experiment window.
        end_date              : final date in the fixed experiment window.
        output_path           : CSV path where raw headline rows are written.
        max_records_per_window : maximum GDELT articles requested per company/date window.
        pause_seconds          : delay between requests to respect GDELT throttling.
        months_per_window      : number of calendar months per GDELT request window.

    Returns:
        Raw GDELT headline records mapped to companies by configured aliases.
    """

    records: list[HeadlineRecord] = []
    seen_ids: set[str] = set()

    for company in companies:
        for window_start, window_end in collection_windows(start_date, end_date, months_per_window):
            print(f"Collecting {company.ticker} {window_start} to {window_end}", flush=True)
            articles = fetch_company_window(company, window_start, window_end, max_records_per_window)
            for article in articles:
                record = article_to_record(article, company)
                if record is None or record.headline_id in seen_ids:
                    continue
                seen_ids.add(record.headline_id)
                records.append(record)
            print(f"Collected {len(records)} total headline rows so far", flush=True)
            time.sleep(pause_seconds)

    write_headlines(records, output_path)
    return records


def fetch_company_window(company: Company, start_date: date, end_date: date, max_records: int) -> list[dict]:
    """
    Args:
        company     : company whose aliases are searched in GDELT.
        start_date  : first date of the request window.
        end_date    : final date of the request window.
        max_records : maximum number of articles to request from GDELT.

    Returns:
        Raw article dictionaries returned by the GDELT DOC API.
    """

    aliases = gdelt_search_aliases(company)
    alias_query = " OR ".join(format_gdelt_term(alias) for alias in aliases)
    query = f"({alias_query})" if len(aliases) > 1 else alias_query
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sourcelang": "english",
        # datedesc is lighter than hybrid ranking and keeps output time-oriented,
        # which suits the later next-trading-day alignment.
        "sort": "datedesc",
        "maxrecords": str(max_records),
        "startdatetime": gdelt_datetime(start_date, start=True),
        "enddatetime": gdelt_datetime(end_date, start=False),
    }
    # GDELT serves the same DOC API over HTTP; in testing, Python HTTPS calls
    # were repeatedly throttled while HTTP respected the normal rate limit.
    url = "http://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)
    try:
        payload = fetch_json_with_retries(url)
    except Exception as exc:
        message = f"GDELT request failed for {company.ticker} {start_date} to {end_date} using query {query!r}"
        raise RuntimeError(message) from exc
    return list(payload.get("articles", []))


def format_gdelt_term(alias: str) -> str:
    """
    Args:
        alias : cleaned company alias to place in the GDELT query.

    Returns:
        A GDELT query term, quoting only multi-word or punctuated aliases.
    """

    return alias if alias.replace("-", "").isalnum() else f'"{alias}"'


def gdelt_search_aliases(company: Company) -> list[str]:
    """
    Args:
        company : company whose configured aliases need to be made safe for GDELT.

    Returns:
        Company-name aliases suitable for GDELT search syntax.
    """

    aliases: list[str] = []
    ticker_base = company.ticker.split(".")[0].lower()

    for alias in company.aliases:
        cleaned = alias.strip()
        # GDELT can reject quoted phrases containing very short tokens such as
        # "plc", so the suffix is removed only for searching, not reporting.
        if cleaned.lower().endswith(" plc"):
            cleaned = cleaned[:-4].strip()
        alnum_length = sum(character.isalnum() for character in cleaned)
        # Ticker-only aliases are too ambiguous for headline collection and can
        # be rejected as short phrases, so they are kept out of GDELT queries.
        is_ticker = cleaned.lower().replace(".", "") in {ticker_base, company.ticker.lower().replace(".", "")}
        if not cleaned or is_ticker or alnum_length < 4:
            continue
        if cleaned not in aliases:
            aliases.append(cleaned)

    return aliases or [company.company_name]


def fetch_json_with_retries(url: str, retries: int = 5, base_delay_seconds: float = 30.0) -> dict:
    """
    Args:
        url                : prepared GDELT HTTP URL.
        retries            : maximum attempts before surfacing the API error.
        base_delay_seconds : delay multiplier used after throttling or network errors.

    Returns:
        Parsed JSON response from GDELT.
    """

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 429:
                can_retry = attempt < retries - 1
                if not can_retry:
                    response.raise_for_status()
                # GDELT asks public users to stay at roughly one request per 5 seconds;
                # 30 seconds is deliberately conservative for reproducible collection.
                time.sleep(base_delay_seconds * (attempt + 1))
                continue
            response.raise_for_status()
            try:
                return response.json()
            except JSONDecodeError as exc:
                snippet = response.text[:300].replace("\n", " ").strip()
                raise ValueError(f"GDELT returned a non-JSON response: {snippet}") from exc
        except requests.RequestException:
            can_retry = attempt < retries - 1
            if not can_retry:
                raise
            time.sleep(base_delay_seconds * (attempt + 1))

    return {}


def article_to_record(article: dict, company: Company) -> HeadlineRecord | None:
    """
    Args:
        article : raw GDELT article dictionary.
        company : company matched by the current GDELT query.

    Returns:
        Normalized headline record, or None if GDELT did not provide a title.
    """

    title = str(article.get("title", "")).strip()
    if not title:
        return None

    # Both UTC and London timestamps are kept so later checks can avoid
    # look-ahead bias when applying UK-market trading rules.
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
    """
    Args:
        value : GDELT seendate value in one of the formats observed from the API.

    Returns:
        Timezone-aware UTC datetime for downstream alignment.
    """

    cleaned = value.strip()
    formats = ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S")
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def stable_headline_id(ticker: str, published_at_utc: str, title: str) -> str:
    """
    Args:
        ticker           : mapped company ticker.
        published_at_utc : normalized publication timestamp.
        title            : headline text.

    Returns:
        Short deterministic hash used to identify duplicate headline rows.
    """

    raw = f"{ticker}|{published_at_utc}|{title}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def gdelt_datetime(value: date, start: bool) -> str:
    """
    Args:
        value : date to convert to GDELT's compact datetime format.
        start : true for start-of-day, false for end-of-day.

    Returns:
        GDELT datetime string in YYYYMMDDHHMMSS format.
    """

    suffix = "000000" if start else "235959"
    return value.strftime("%Y%m%d") + suffix


def collection_windows(start_date: date, end_date: date, months_per_window: int) -> Iterable[tuple[date, date]]:
    """
    Args:
        start_date        : first date in the full collection window.
        end_date          : final date in the full collection window.
        months_per_window : number of calendar months included in each request.

    Returns:
        Date windows to keep each GDELT request manageable without over-querying.
    """

    if months_per_window < 1:
        raise ValueError("months_per_window must be at least 1.")

    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        next_window = add_months(current, months_per_window)
        window_start = max(current, start_date)
        window_end = min(next_window - timedelta(days=1), end_date)
        yield window_start, window_end
        current = next_window


def add_months(value: date, months: int) -> date:
    """
    Args:
        value  : starting date.
        months : number of calendar months to add.

    Returns:
        Date moved forward by the requested number of months.
    """

    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def write_headlines(records: Iterable[HeadlineRecord], output_path: str | Path) -> None:
    """
    Args:
        records     : normalized headline records to persist.
        output_path : destination CSV path for the raw headline dataset.
    """

    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADLINE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
