from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .config import load_companies, load_experiment_config
from .gdelt import collect_gdelt_headlines
from .prices import collect_yfinance_prices


DEFAULT_CONFIG = "config/experiment.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="UK DK-CoT dissertation data collection CLI")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to experiment config JSON")

    subparsers = parser.add_subparsers(dest="command", required=True)

    headlines = subparsers.add_parser("collect-headlines", help="Collect GDELT headline data")
    add_date_args(headlines)
    headlines.add_argument("--output", help="Output CSV path for headlines")
    headlines.add_argument("--max-records-per-month", type=int, default=250)

    prices = subparsers.add_parser("collect-prices", help="Collect yfinance daily prices")
    add_date_args(prices)
    prices.add_argument("--output", help="Output CSV path for prices")

    collect_all = subparsers.add_parser("collect-all", help="Collect both headlines and prices")
    add_date_args(collect_all)
    collect_all.add_argument("--headlines-output", help="Output CSV path for headlines")
    collect_all.add_argument("--prices-output", help="Output CSV path for prices")
    collect_all.add_argument("--max-records-per-month", type=int, default=250)

    args = parser.parse_args()
    config = load_experiment_config(args.config)
    companies = load_companies(config["companies_path"])
    start_date, end_date = resolve_dates(args, config)

    if args.command == "collect-headlines":
        output = args.output or config["data_paths"]["raw_headlines"]
        records = collect_gdelt_headlines(companies, start_date, end_date, output, args.max_records_per_month)
        print(f"Wrote {len(records)} headline rows to {output}")
    elif args.command == "collect-prices":
        output = args.output or config["data_paths"]["raw_prices"]
        prices = collect_yfinance_prices(companies, start_date, end_date, output)
        print(f"Wrote {len(prices)} price rows to {output}")
    elif args.command == "collect-all":
        headlines_output = args.headlines_output or config["data_paths"]["raw_headlines"]
        prices_output = args.prices_output or config["data_paths"]["raw_prices"]
        records = collect_gdelt_headlines(
            companies,
            start_date,
            end_date,
            headlines_output,
            args.max_records_per_month,
        )
        prices = collect_yfinance_prices(companies, start_date, end_date, prices_output)
        print(f"Wrote {len(records)} headline rows to {headlines_output}")
        print(f"Wrote {len(prices)} price rows to {prices_output}")


def add_date_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format")


def resolve_dates(args: argparse.Namespace, config: dict) -> tuple[date, date]:
    config_window = config.get("date_window", {})
    start_value = args.start_date or config_window.get("start_date")
    end_value = args.end_date or config_window.get("end_date")
    if not start_value or not end_value:
        raise SystemExit("Provide --start-date and --end-date, or set date_window in config/experiment.json.")

    start_date = date.fromisoformat(start_value)
    end_date = date.fromisoformat(end_value)
    if end_date < start_date:
        raise SystemExit("--end-date must be on or after --start-date.")
    return start_date, end_date


if __name__ == "__main__":
    main()
