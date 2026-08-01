from __future__ import annotations

from datetime import date
from pathlib import Path

from uk_dkcot.config import load_companies
from uk_dkcot.gdelt import collect_gdelt_headlines


OUTPUT_PATH = Path("data/raw/gdelt_azn_q1_smoke.csv")


def main() -> None:
    """
    Runs a deliberately small GDELT smoke test before the full data collection.

    This is not a unit test and it is not part of the final academic evaluation.
    It does not use a manually verified "gold" dataset as a source of truth.
    Instead, it checks that our code fetches the live GDELT DOC API results for
    the exact company/date/query we asked for.

    In other words, this smoke test proves the collection mechanism works; it
    does not make the first five headlines a source of truth for the full
    dataset. The full collection is not checked by matching it against this
    smoke-test CSV. Instead, the full collection later gets its own schema,
    date-range, duplicate and mapping-quality checks. Manual sentiment labels
    are created later as the academic source of truth for classification.

    It is a practical check that the live GDELT collector can:
    1. call the API successfully,
    2. write the expected headline CSV schema,
    3. parse GDELT timestamps correctly, and
    4. map returned headlines to the configured company.

    The test uses only AstraZeneca for Q1 2025 and asks for at most five
    headlines, so it does not replace the full 2025 ten-company collection.
    """

    companies = [company for company in load_companies("config/companies.csv") if company.ticker == "AZN.L"]
    records = collect_gdelt_headlines(
        companies=companies,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 3, 31),
        output_path=OUTPUT_PATH,
        max_records_per_window=5,
        pause_seconds=0,
        months_per_window=3,
    )
    print(f"Wrote {len(records)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
