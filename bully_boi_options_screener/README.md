# Bully Boi Options Play Screener v1

This is a starter Python screener for finding CALL/PUT watchlist setups using:

- 9 EMA trend
- RSI momentum
- relative volume
- daily price change
- SPY market context
- near-the-money options liquidity/spread check
- ATR-based entry, stop, and targets

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python screener.py
```

## Edit the watchlist

Open `tickers.txt` and add/remove tickers, one per line.

## Output

The script prints the ranked scan and saves:

```bash
options_screener_results.csv
```

## Signal guide

- Score 80-100 = stronger watchlist setup
- Score 60-79 = possible setup, needs chart confirmation
- PASS = not enough confirmation

## Risk rules to follow manually

- Avoid contracts with wide bid/ask spreads.
- Avoid trading during major news unless that is the strategy.
- Consider selling partial profit around +25% option premium.
- Consider cutting quickly near -20% option premium or when price loses the 9 EMA.

This is an educational screener, not financial advice or an auto-trading system.
