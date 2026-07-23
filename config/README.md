# Configuration Notes

`experiment.json` is the single place that fixes the dissertation experiment setup.
It stays as strict JSON because Python can load it directly without custom parsing.

`companies.csv` stores the fixed ten-company universe and the knowledge fields used
later by the DK-CoT treatments.

## `experiment.json`

- `project_title`: names the UK equity adaptation of Chen et al.'s DK-CoT design.
- `companies_path`: points to the selected-company universe in `companies.csv`.
- `date_window`: fixes the 12-month experiment window so collection and evaluation are reproducible.
- `data_paths`: keeps raw GDELT headlines and yfinance prices in predictable CSV locations.
- `schemas`: documents the required columns for each dataset before later validation steps.
- `sentiment_labels`: limits manual labels and model outputs to positive, negative and neutral.
- `alignment_rule`: uses next-trading-day open to avoid look-ahead bias.
- `knowledge_levels`: defines the planned ablation settings: no knowledge, sector knowledge and firm knowledge.
- `trading_strategies`: limits the backtest scope to long-only and long-short rules from the report.
- `evaluation_metrics`: limits reporting to classification metrics and trading metrics named in the report.

## `companies.csv`

- `ticker`: yfinance ticker used to collect daily UK equity prices.
- `company_name`: official name shown in outputs and dissertation tables.
- `aliases`: pipe-separated names used for headline matching. The pipe avoids conflicts with CSV commas.
- `sector`: sector-level knowledge for the DK-CoT sector treatment.
- `products`: firm-specific knowledge for the DK-CoT firm treatment.
- `risks`: firm-specific risk context for the DK-CoT firm treatment.
