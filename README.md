# Option QuantLib

_This documentation and code comments were generated with AI assistance to enhance technical clarity._

A practical toolkit for European option research using QuantLib, providing convenient wrappers for calendar setup, stock/option modeling, pricing calculations, Greeks, and hedging strategies.

## Project Overview

This project encapsulates QuantLib functionality into a more accessible Python interface, focusing on:
- European option pricing and analysis
- Calendar and market conventions
- Stock modeling (including Black-Scholes and Heston and other models)
- Option Greeks calculation
- Delta hedging strategies
- Implied volatility surface analysis

# Data Sources

All options datasets were sourced from Kaggle user [Kyle Graupe](https://www.kaggle.com/kylegraupe)'s high-quality collections. 

The raw datasets can be downloaded via the links below and processed using our conversion script:

| Ticker | Dataset Period | Link |
|--------|----------------|------|
| AAPL   | 2016-2020 | [Dataset](https://www.kaggle.com/datasets/kylegraupe/aapl-options-data-2016-2020) |
| SPY    | 2020-2022 | [Dataset](https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022) |
| TSLA   | 2019-2022 | [Dataset](https://www.kaggle.com/datasets/kylegraupe/tsla-daily-eod-options-quotes-2019-2022) |
| NVDA   | 2020-2022 | [Dataset](https://www.kaggle.com/datasets/kylegraupe/nvda-daily-option-chains-q1-2020-to-q4-2022) |
| QQQ    | 2020-2022 | [Dataset](https://www.kaggle.com/datasets/kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022) |

**Data Processing Pipeline**:
1. Download raw CSV files from Kaggle links
2. Run conversion script:
   ```bash
   python data/real_data/to_feather.py
   ```
3. Processed files will be saved in optimized Feather format

**Data Characteristics:**
- End-of-Day (EOD) options chain data
- Cleaned and standardized columns
- Includes bid/ask prices, open interest, and volume
- Covers periods of significant market volatility
- Now in optimized Feather format for faster analysis

## Repository Structure

## Validation Methodology
1. **Data Preprocessing**:
   - Aligned raw Kaggle data with QuantLib's market conventions
   - Normalized strike prices and expiration dates
   - Filtered for European-style options only

2. **Benchmarking**:
   - Compared computed Greeks against broker-reported values
   - Verified volatility surfaces against historical realized volatility
   - Backtested hedging strategies using the SPY dataset

