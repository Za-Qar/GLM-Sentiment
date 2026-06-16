# GLM-Sentiment
Using Big Language Models for Financial News Sentiment Classification

## UK DK-CoT dissertation prototype

This fork is being adapted only for the submitted specification-and-design scope:
UK equity headlines from GDELT, daily price data from yfinance, FinBERT vs DK-CoT
sentiment, and long-only/long-short trading evaluation.

The first implemented stage is data collection for the ten companies in the
submitted report.

The core experiment uses a fixed 12-month historical window:
`2025-01-01` to `2025-12-31`.

```powershell
python -m uk_dkcot.cli collect-headlines
python -m uk_dkcot.cli collect-prices
python -m uk_dkcot.cli collect-all
```

You can still override the dates for small test runs:

```powershell
python -m uk_dkcot.cli collect-all --start-date 2025-01-01 --end-date 2025-01-31
```
