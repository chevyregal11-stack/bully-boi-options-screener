import argparse
from dataclasses import dataclass
from datetime import datetime
import math
import numpy as np
import pandas as pd
import yfinance as yf


def ema(series: pd.Series, length: int = 9) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


@dataclass
class Setup:
    ticker: str
    direction: str
    score: int
    price: float
    rsi: float
    ema9: float
    rel_volume: float
    day_change_pct: float
    atr: float
    entry: float
    stop: float
    target1: float
    target2: float
    option_contract: str
    option_bid: float
    option_ask: float
    option_spread_pct: float
    option_volume: int
    open_interest: int
    notes: str


def choose_option(ticker_obj: yf.Ticker, price: float, direction: str):
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None
        expiry = expirations[0]
        chain = ticker_obj.option_chain(expiry)
        options = chain.calls if direction == "CALL" else chain.puts
        if options.empty:
            return None

        # Pick a near-the-money contract with decent liquidity and not-crazy spread.
        options = options.copy()
        options["distance"] = (options["strike"] - price).abs()
        options["mid"] = (options["bid"].fillna(0) + options["ask"].fillna(0)) / 2
        options["spread_pct"] = np.where(options["mid"] > 0, (options["ask"] - options["bid"]) / options["mid"] * 100, 999)
        options = options.sort_values(["distance", "spread_pct", "volume"], ascending=[True, True, False])
        liquid = options[(options["bid"] > 0) & (options["ask"] > 0) & (options["spread_pct"] <= 20)]
        row = liquid.iloc[0] if not liquid.empty else options.iloc[0]
        return {
            "contract": f"{expiry} {direction} ${row['strike']}",
            "bid": safe_float(row.get("bid")),
            "ask": safe_float(row.get("ask")),
            "spread_pct": safe_float(row.get("spread_pct"), 999),
            "volume": int(safe_float(row.get("volume"), 0)),
            "open_interest": int(safe_float(row.get("openInterest"), 0)),
        }
    except Exception:
        return None


def scan_ticker(symbol: str, market_bullish: bool, market_bearish: bool) -> Setup | None:
    symbol = symbol.strip().upper()
    if not symbol:
        return None
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period="3mo", interval="1d", auto_adjust=False)
        if df is None or len(df) < 30:
            return None

        df["EMA9"] = ema(df["Close"], 9)
        df["RSI14"] = rsi(df["Close"], 14)
        df["ATR14"] = atr(df, 14)
        df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = safe_float(last["Close"])
        ema9 = safe_float(last["EMA9"])
        rsi14 = safe_float(last["RSI14"])
        rel_vol = safe_float(last["Volume"] / last["AVG_VOL20"] if last["AVG_VOL20"] else 0)
        change_pct = safe_float((last["Close"] - prev["Close"]) / prev["Close"] * 100)
        atr14 = safe_float(last["ATR14"])

        bull_score = 0
        bear_score = 0
        bull_notes = []
        bear_notes = []

        if price > ema9:
            bull_score += 20; bull_notes.append("above 9 EMA")
        else:
            bear_score += 20; bear_notes.append("below 9 EMA")

        if 55 <= rsi14 <= 75:
            bull_score += 20; bull_notes.append("bullish RSI")
        if rsi14 < 45:
            bear_score += 20; bear_notes.append("bearish RSI")

        if rel_vol >= 1.5:
            bull_score += 20; bear_score += 20
            bull_notes.append("relative volume hot"); bear_notes.append("relative volume hot")

        if change_pct >= 1:
            bull_score += 20; bull_notes.append("green momentum day")
        if change_pct <= -1:
            bear_score += 20; bear_notes.append("red momentum day")

        if market_bullish:
            bull_score += 20; bull_notes.append("SPY bullish")
        if market_bearish:
            bear_score += 20; bear_notes.append("SPY bearish")

        direction = "CALL" if bull_score >= bear_score else "PUT"
        score = max(bull_score, bear_score)
        notes = ", ".join(bull_notes if direction == "CALL" else bear_notes)

        if score < 60:
            direction = "PASS"

        option = choose_option(ticker_obj, price, direction) if direction in ["CALL", "PUT"] else None
        if option is None:
            option = {"contract": "N/A", "bid": 0, "ask": 0, "spread_pct": 999, "volume": 0, "open_interest": 0}

        if direction == "CALL":
            entry = round(price, 2)
            stop = round(price - atr14, 2)
            target1 = round(price + atr14, 2)
            target2 = round(price + (atr14 * 2), 2)
        elif direction == "PUT":
            entry = round(price, 2)
            stop = round(price + atr14, 2)
            target1 = round(price - atr14, 2)
            target2 = round(price - (atr14 * 2), 2)
        else:
            entry = stop = target1 = target2 = round(price, 2)

        return Setup(symbol, direction, score, round(price,2), round(rsi14,1), round(ema9,2), round(rel_vol,2), round(change_pct,2), round(atr14,2), entry, stop, target1, target2, option["contract"], option["bid"], option["ask"], round(option["spread_pct"],1), option["volume"], option["open_interest"], notes)
    except Exception as exc:
        print(f"Skipping {symbol}: {exc}")
        return None


def market_context():
    spy = yf.Ticker("SPY").history(period="1mo", interval="1d", auto_adjust=False)
    spy["EMA9"] = ema(spy["Close"], 9)
    last = spy.iloc[-1]
    bullish = last["Close"] > last["EMA9"]
    bearish = last["Close"] < last["EMA9"]
    return bullish, bearish

def get_option_contract(symbol, direction, price):
    try:
        ticker_obj = yf.Ticker(symbol)
        expirations = ticker_obj.options

        if not expirations:
            return {}

        expiration = expirations[0]
        chain = ticker_obj.option_chain(expiration)

        options = chain.calls if direction == "CALL" else chain.puts

        if options.empty:
            return {}

        options["distance"] = (options["strike"] - price).abs()
        contract = options.sort_values("distance").iloc[0]

        bid = safe_float(contract.get("bid", 0))
        ask = safe_float(contract.get("ask", 0))
        spread_pct = ((ask - bid) / ask * 100) if ask else 999

        return {
            "option_expiration": expiration,
            "option_type": direction,
            "option_strike": safe_float(contract.get("strike", 0)),
            "option_bid": bid,
            "option_ask": ask,
            "option_spread_pct": round(spread_pct, 2),
            "option_volume": safe_float(contract.get("volume", 0)),
            "option_open_interest": safe_float(contract.get("openInterest", 0)),
            "option_symbol": contract.get("contractSymbol", ""),
        }

    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Bully Boi Options Play Screener")
    parser.add_argument("--tickers", default="tickers.txt", help="Path to ticker list, one symbol per line")
    parser.add_argument("--out", default="options_screener_results.csv", help="CSV output file")
    args = parser.parse_args()

    with open(args.tickers, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    market_bullish, market_bearish = market_context()
    results = []
    for symbol in tickers:
        setup = scan_ticker(symbol, market_bullish, market_bearish)
        if setup:
            results.append(setup.__dict__)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["score", "rel_volume"], ascending=[False, False])
        df.to_csv(args.out, index=False)
        print("\nBULLY BOI OPTIONS SCREENER")
        print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Market context: {'BULLISH' if market_bullish else 'BEARISH'}")
        print(df[["ticker", "direction", "score", "price", "rsi", "ema9", "rel_volume", "day_change_pct", "option_contract", "option_bid", "option_ask", "option_spread_pct", "notes"]].to_string(index=False))
        print(f"\nSaved: {args.out}")
    else:
        print("No results. Check ticker list or data connection.")


if __name__ == "__main__":
    main()
