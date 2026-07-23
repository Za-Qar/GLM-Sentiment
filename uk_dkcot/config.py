from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Company:
    """
    Args:
        ticker       : yfinance ticker used to collect daily UK equity prices.
        company_name : official company name used in outputs and reports.
        aliases      : names/ticker variants used to map headlines to the company.
        sector       : sector-level knowledge used in the DK-CoT sector treatment.
        products     : firm-specific product knowledge used in the DK-CoT firm treatment.
        risks        : firm-specific risk knowledge used in the DK-CoT firm treatment.
    """

    ticker: str
    company_name: str
    aliases: tuple[str, ...]
    sector: str
    products: str
    risks: str


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """
    Args:
        path : path to the JSON file that fixes the experiment setup.

    Returns:
        Dictionary containing the selected companies file, date window, schemas,
        model settings and evaluation choices from the submitted project scope.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_companies(path: str | Path) -> list[Company]:
    """
    Args:
        path : path to the CSV file containing the ten-company universe.

    Returns:
        List of Company records used by headline collection, price collection and
        later knowledge-injection experiments.
    """

    companies_path = Path(path)
    with companies_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        # These columns are required because they directly map to the report's
        # fixed universe, alias mapping and firm-knowledge treatments.
        required = {"ticker", "company_name", "aliases", "sector", "products", "risks"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required company columns: {missing_cols}")

        companies: list[Company] = []
        for row in reader:
            # Aliases use "|" rather than "," so the CSV remains simple and the
            # headline mapping can keep several names for the same company.
            aliases = tuple(alias.strip() for alias in row["aliases"].split("|") if alias.strip())
            companies.append(
                Company(
                    ticker=row["ticker"].strip(),
                    company_name=row["company_name"].strip(),
                    aliases=aliases,
                    sector=row["sector"].strip(),
                    products=row["products"].strip(),
                    risks=row["risks"].strip(),
                )
            )
        return companies


def ensure_parent_dir(path: str | Path) -> None:
    """
    Args:
        path : file path whose parent directory should exist before writing.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
