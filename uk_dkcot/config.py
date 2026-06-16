from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Company:
    ticker: str
    company_name: str
    aliases: tuple[str, ...]
    sector: str
    products: str
    risks: str


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_companies(path: str | Path) -> list[Company]:
    companies_path = Path(path)
    with companies_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "company_name", "aliases", "sector", "products", "risks"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required company columns: {missing_cols}")

        companies: list[Company] = []
        for row in reader:
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
