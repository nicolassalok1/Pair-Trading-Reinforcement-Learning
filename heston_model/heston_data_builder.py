"""
Utilities to build a Heston-friendly implied volatility surface using existing fetchers.

The entry point is `get_heston_dataset(symbol, source)` which:
  - fetches the latest spot/close price,
  - pulls option chains,
  - selects target maturities (30/60/90/180/365 days) and ATM/OTM/ITM strikes,
  - returns a tidy DataFrame and saves it to ./data/heston_iv_surface.csv.

Sources supported:
  - "yahoo": uses DATA.API.YahooFinance (optional dependency: yfinance)
  - "local": uses CSV in STATICS/PRICE for spot; options must be pre-fetched (not automated)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from DATA.API import YahooFinance

logger = logging.getLogger(__name__)

MATURITY_TARGET_DAYS: List[int] = [30, 60, 90, 180, 365]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "heston_iv_surface.csv"
PRICE_DIR = Path(__file__).resolve().parents[1] / "STATICS" / "PRICE"


def _round_price(value: float) -> float:
    return float(np.round(value, 4))


def _load_cached(symbol: str, as_of_date: datetime) -> Optional[pd.DataFrame]:
    if not OUTPUT_PATH.exists():
        return None
    try:
        df = pd.read_csv(OUTPUT_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read cache %s: %s", OUTPUT_PATH, exc)
        return None
    if df.empty or "symbol" not in df.columns or "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    today_rows = df[(df["symbol"] == symbol) & (df["date"] >= as_of_date.date())]
    if today_rows.empty:
        return None
    if not set(MATURITY_TARGET_DAYS).issubset(set(today_rows["maturity"].unique())):
        return None
    logger.info("Using cached Heston dataset from %s for %s", OUTPUT_PATH, symbol)
    return today_rows.reset_index(drop=True)


def _save_dataset(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved Heston IV surface to %s", OUTPUT_PATH)


def _get_spot_price(symbol: str, source: str) -> float:
    source = source.lower()
    if source == "yahoo":
        yf_fetcher = YahooFinance(symbol)
        price = yf_fetcher.get_last_close()
        logger.info("Fetched spot from Yahoo Finance: %s -> %.4f", symbol, price)
        return price

    # Fallback: local CSV in STATICS/PRICE
    local_path = PRICE_DIR / f"{symbol}.csv"
    if local_path.exists():
        df = pd.read_csv(local_path)
        if "close" in df.columns and not df["close"].dropna().empty:
            price = float(df["close"].dropna().iloc[-1])
            logger.info("Using spot from local file %s -> %.4f", local_path, price)
            return price
    raise ValueError(f"Unable to determine spot price for {symbol} using source '{source}'.")


def _target_strikes(spot: float) -> Dict[str, List[float]]:
    return {
        "call": [_round_price(spot), _round_price(spot * 1.10), _round_price(spot * 0.90)],
        "put": [_round_price(spot), _round_price(spot * 0.90), _round_price(spot * 1.10)],
    }


def _select_expirations(
    available: Iterable[str], as_of: datetime, targets: List[int]
) -> Dict[int, datetime]:
    exp_dates = [pd.to_datetime(e) for e in available if e]
    if not exp_dates:
        return {}
    selected: Dict[int, datetime] = {}
    for ttm in targets:
        desired = as_of + timedelta(days=int(ttm))
        future_exps = [e for e in exp_dates if e > as_of]
        if not future_exps:
            continue
        best = min(future_exps, key=lambda d: abs((d - desired).days))
        selected[ttm] = best
    return selected


def _fetch_option_chains(
    symbol: str, source: str, expirations: Dict[int, datetime]
) -> pd.DataFrame:
    if not expirations:
        raise ValueError("No expirations selected; cannot build option surface.")

    source = source.lower()
    all_rows: List[pd.DataFrame] = []

    if source == "yahoo":
        fetcher = YahooFinance(symbol)
        for ttm_days, exp_dt in expirations.items():
            exp_str = exp_dt.strftime("%Y-%m-%d")
            calls, puts = fetcher.get_option_chain(exp_str)
            calls = calls.assign(option_type="call", maturity=ttm_days, expiration=exp_dt)
            puts = puts.assign(option_type="put", maturity=ttm_days, expiration=exp_dt)
            all_rows.extend([calls, puts])
    else:
        raise ValueError(f"Unsupported options source '{source}'.")

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def _nearest_row(df: pd.DataFrame, target_strike: float) -> Optional[pd.Series]:
    if df.empty or "strike" not in df.columns:
        return None
    df = df.copy()
    df["diff"] = (df["strike"] - target_strike).abs()
    row = df.sort_values("diff").head(1)
    if row.empty:
        return None
    return row.iloc[0]


def _build_surface(
    symbol: str,
    spot: float,
    chains: pd.DataFrame,
    strikes: Dict[str, List[float]],
    as_of: datetime,
) -> pd.DataFrame:
    required_cols = {"maturity", "strike", "option_type", "impliedVolatility"}
    if chains.empty or not required_cols.issubset(set(chains.columns)):
        raise ValueError("Option chain is missing required columns for surface construction.")

    records: List[Dict[str, float]] = []
    for opt_type, strike_list in strikes.items():
        for target_strike in strike_list:
            subset = chains[chains["option_type"] == opt_type]
            for maturity in MATURITY_TARGET_DAYS:
                exp_rows = subset[subset["maturity"] == maturity]
                selected = _nearest_row(exp_rows, target_strike)
                if selected is None:
                    logger.warning(
                        "Missing option for %s @ strike %.2f, maturity %s days", opt_type, target_strike, maturity
                    )
                    continue
                iv = selected.get("impliedVolatility", np.nan)
                records.append(
                    {
                        "symbol": symbol,
                        "date": as_of.date(),
                        "underlying_price": spot,
                        "maturity": int(maturity),
                        "strike": float(selected["strike"]),
                        "option_type": opt_type,
                        "implied_volatility": float(iv) if pd.notnull(iv) else np.nan,
                    }
                )

    if not records:
        raise ValueError("No option records selected for the Heston surface.")
    return pd.DataFrame(records)


def get_heston_dataset(symbol: str, source: str = "yahoo") -> pd.DataFrame:
    """
    Fetch option data and build a Heston IV surface for the given symbol.

    Parameters
    ----------
    symbol : str
        Underlying ticker, e.g. 'AAPL' or 'SPY'.
    source : str
        Data source for spot and options. Currently 'yahoo' is supported.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, date, underlying_price, maturity, strike, option_type, implied_volatility
    """
    as_of = pd.Timestamp.utcnow().normalize()

    cached = _load_cached(symbol, as_of)
    if cached is not None:
        return cached

    spot = _get_spot_price(symbol, source=source)
    strikes = _target_strikes(spot)

    if source.lower() == "yahoo":
        yf_fetcher = YahooFinance(symbol)
        expirations = _select_expirations(yf_fetcher.list_expirations(), as_of, MATURITY_TARGET_DAYS)
    else:
        expirations = {}

    chains = _fetch_option_chains(symbol, source, expirations)
    surface = _build_surface(symbol, spot, chains, strikes, as_of)
    _save_dataset(surface)
    return surface


__all__ = ["get_heston_dataset"]
