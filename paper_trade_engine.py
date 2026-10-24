"""
Scheduled 3:00 PM IST Weekday Paper Trading Engine & 10-Year Performance Ledger
===============================================================================
Executes and tracks paper trades for the HNI & Institutional Action Desk:
- Schedule: Every Weekday (Monday–Friday) at 3:00 PM IST (15:00 Asia/Kolkata).
- Selection: Top 3 High-Conviction BUY tickers & Top 3 High-Conviction SELL tickers
  (strictly requiring Factual Success Probability >= 50%).
- Guardrails:
  1. Weekdays Only (Monday–Friday; weekend filings roll to next Monday 3:00 PM IST).
  2. Value-Change Gate: Executes ONLY if live price / signal value has changed since
     the previous 3:00 PM IST snapshot.
  3. Zero Duplicates: Never buys/sells a ticker that is already open as an ACTIVE trade.
- Performance Analytics:
  - Total Invested (Active + Closed), Active Invested, Closed Invested
  - Count of Active & Closed Trades (broken down by BUY and SELL)
  - Explicit BUY / SELL Tickers against each trade (Active & Closed)
  - Closed Trades Success Rate (Win %) & Total (Active + Closed) Success Rate
  - Performance across:
    * Total (Active + Closed)
    * Closed Trades Only
    * 1 Week (7D)
    * 2 Weeks (14D)
    * 1 Month (30D)
    * 3 Months (91D)
    * 1 Year (365D)
    * 2 Years (2Y)
    * 3 Years (3Y)
    * 5 Years (5Y)
    * 10 Years (10Y)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from live_data_engine import CACHE_DIR, DATA_DIR

logger = logging.getLogger(__name__)

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
IST_TZ = dt.timezone(IST_OFFSET, name="IST")

CAPITAL_PER_TRADE_INR: float = 100_000.0  # ₹1,00,000 standard allocation per paper trade
DEFAULT_MAX_HOLDING_DAYS: int = 20        # 20 trading days (~1 month) max holding period
LEDGER_JSON_PATH: Path = DATA_DIR / "paper_trades_ledger.json"
LEDGER_CSV_PATH: Path = DATA_DIR / "paper_trades_ledger.csv"
SCHEDULE_STATE_PATH: Path = DATA_DIR / "paper_trade_schedule_state.json"

_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER_STARTED = False


def now_ist() -> dt.datetime:
    """Return current timezone-aware datetime in IST (UTC+05:30)."""
    return dt.datetime.now(dt.timezone.utc).astimezone(IST_TZ)


def next_weekday_3pm_ist(ref_dt: Optional[dt.datetime] = None) -> dt.datetime:
    """Compute the next scheduled 3:00 PM IST weekday (Mon–Fri) execution timestamp."""
    cur = ref_dt.astimezone(IST_TZ) if ref_dt is not None else now_ist()
    candidate = cur.replace(hour=15, minute=0, second=0, microsecond=0)
    if cur >= candidate or candidate.weekday() >= 5:
        candidate += dt.timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += dt.timedelta(days=1)
    return candidate


def roll_to_weekday(d: pd.Timestamp) -> pd.Timestamp:
    """If a filing date falls on Saturday/Sunday, roll forward to Monday (next weekday)."""
    ts = pd.Timestamp(d).tz_localize(None).normalize()
    while ts.weekday() >= 5:
        ts += pd.Timedelta(days=1)
    return ts


def _load_cached_price_map() -> Dict[str, pd.DataFrame]:
    """Load cached daily OHLCV DataFrames from data/cache/yf_ohlcv/*.pkl."""
    price_map: Dict[str, pd.DataFrame] = {}
    pkl_dir = CACHE_DIR / "yf_ohlcv"
    if not pkl_dir.exists():
        return price_map
    for pkl in pkl_dir.glob("*.pkl"):
        sym = pkl.stem.replace("IDX_", "^")
        try:
            df = pd.read_pickle(pkl)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                df = df.copy()
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df = df[~df.index.duplicated(keep="last")].sort_index()
                price_map[sym] = df
        except Exception:
            pass
    return price_map


def _compute_factual_prob_for_row(row: pd.Series) -> Tuple[str, float, float, float]:
    """
    Compute factual Buy vs Sell/Exit probability for a historical or live event row.
    Returns (side, action_prob_pct, buy_prob_pct, sell_prob_pct).
    Strictly uses empirical stock conditional matched-state rates from actual OHLCV history
    (zero synthetic RSI offsets or arbitrary clipping).
    """
    direction = str(row.get("direction", "BUY")).upper()
    stock_up = row.get("stock_cond_up_20d_pct", np.nan)
    stock_down = row.get("stock_cond_down_20d_pct", np.nan)
    mfe3 = row.get("stock_cond_mfe3_20d_pct", np.nan)
    mae3 = row.get("stock_cond_mae3_20d_pct", np.nan)

    # If no matched historical OHLCV state is available, return < 50% so it is strictly ignored
    if pd.isna(stock_up) and pd.isna(stock_down):
        return direction, 0.0, 0.0, 0.0

    buy_p = float(stock_up) if pd.notna(stock_up) else (100.0 - float(stock_down))
    sell_p = float(stock_down) if pd.notna(stock_down) else (100.0 - float(stock_up))

    if direction == "BUY":
        # Require raw conditional upward win rate >= 50%
        if buy_p < 50.0:
            return "BUY", round(buy_p, 1), round(buy_p, 1), round(sell_p, 1)
        if pd.notna(mfe3):
            buy_p = 0.5 * (buy_p + float(mfe3))
        return "BUY", round(buy_p, 1), round(buy_p, 1), round(sell_p, 1)
    else:
        # Require raw conditional downward rate >= 50% for SELL
        if sell_p < 50.0:
            return "SELL", round(sell_p, 1), round(buy_p, 1), round(sell_p, 1)
        if pd.notna(mae3):
            sell_p = 0.5 * (sell_p + float(mae3))
        return "SELL", round(sell_p, 1), round(buy_p, 1), round(sell_p, 1)


def _simulate_trade_exit_on_ohlcv(
    side: str,
    entry_date: pd.Timestamp,
    entry_price: float,
    ohlcv: Optional[pd.DataFrame],
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    cutoff_date: Optional[pd.Timestamp] = None,
    fallback_ret_20d_pct: Optional[float] = None,
    fallback_live_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate a paper trade opened at 3:00 PM IST on `entry_date` using actual weekday OHLCV prices.
    - Checks subsequent weekday bars up to `max_holding_days` trading days.
    - Closes when:
      * Profit target (+4.5% gain in trade direction) is reached on the session bar, OR
      * Closing Stop-Loss (-6.0% adverse move at 3:00 PM IST close) is breached, OR
      * The `max_holding_days`-th trading day is completed at 3:00 PM IST.
    - If fewer than `max_holding_days` trading days have elapsed as of `cutoff_date` (today)
      and neither target nor stop-loss has triggered, the trade remains `ACTIVE` marked to
      the latest live price!
    """
    if cutoff_date is None:
        cutoff_date = pd.Timestamp(now_ist().date())

    target_tp_pct = 4.5
    stop_loss_pct = -6.0

    if ohlcv is not None and not ohlcv.empty and "Close" in ohlcv.columns:
        future_bars = ohlcv[ohlcv.index > entry_date]
        # Keep only weekdays (Mon–Fri)
        future_bars = future_bars[future_bars.index.weekday < 5]
        if not future_bars.empty:
            bars_up_to_now = future_bars[future_bars.index <= cutoff_date]
            n_avail = len(bars_up_to_now)

            exit_bar_date: Optional[pd.Timestamp] = None
            exit_price: Optional[float] = None
            exit_reason = ""
            days_held = 0
            realized_ret_pct: Optional[float] = None

            scan_bars = bars_up_to_now.iloc[:max_holding_days]
            for idx_i, (b_date, b_row) in enumerate(scan_bars.iterrows(), start=1):
                c_px = float(b_row["Close"])
                if c_px <= 0 or np.isnan(c_px):
                    continue
                h_px = float(b_row["High"]) if "High" in b_row and pd.notna(b_row["High"]) else c_px
                l_px = float(b_row["Low"]) if "Low" in b_row and pd.notna(b_row["Low"]) else c_px

                c_ret = ((c_px - entry_price) / entry_price) * 100.0
                h_ret = ((h_px - entry_price) / entry_price) * 100.0
                l_ret = ((l_px - entry_price) / entry_price) * 100.0

                best_ret = max(c_ret, h_ret) if side == "BUY" else max(-c_ret, -l_ret)
                close_ret = c_ret if side == "BUY" else -c_ret

                if best_ret >= target_tp_pct:
                    exit_bar_date = b_date
                    realized_ret_pct = max(target_tp_pct, close_ret)
                    exit_price = (
                        entry_price * (1.0 + realized_ret_pct / 100.0)
                        if side == "BUY"
                        else entry_price * (1.0 - realized_ret_pct / 100.0)
                    )
                    exit_reason = f"Target Hit (+{realized_ret_pct:.1f}% at 3PM IST)"
                    days_held = idx_i
                    break
                if close_ret <= stop_loss_pct:
                    exit_bar_date = b_date
                    realized_ret_pct = close_ret
                    exit_price = c_px
                    exit_reason = f"Stop-Loss Guard ({close_ret:.1f}% at 3PM IST)"
                    days_held = idx_i
                    break

            if exit_bar_date is None and n_avail >= max_holding_days:
                last_bar = scan_bars.iloc[-1]
                exit_bar_date = scan_bars.index[-1]
                exit_price = float(last_bar["Close"])
                raw_ret = ((exit_price - entry_price) / entry_price) * 100.0
                realized_ret_pct = raw_ret if side == "BUY" else -raw_ret
                exit_reason = f"+{max_holding_days}D Horizon Completed (3PM IST)"
                days_held = max_holding_days

            if exit_bar_date is not None and exit_price is not None and realized_ret_pct is not None:
                return {
                    "status": "CLOSED",
                    "exit_date": exit_bar_date.strftime("%Y-%m-%d"),
                    "exit_timestamp_ist": f"{exit_bar_date.strftime('%Y-%m-%d')} 15:00:00 IST",
                    "exit_price": round(exit_price, 2),
                    "current_price": round(exit_price, 2),
                    "holding_trading_days": int(days_held),
                    "return_pct": round(realized_ret_pct, 2),
                    "exit_reason": exit_reason,
                }

            # Otherwise the trade is still ACTIVE (fewer than max_holding_days elapsed)
            latest_px = (
                float(bars_up_to_now["Close"].iloc[-1])
                if not bars_up_to_now.empty
                else (float(fallback_live_price) if fallback_live_price and fallback_live_price > 0 else entry_price)
            )
            latest_dt = (
                bars_up_to_now.index[-1].strftime("%Y-%m-%d")
                if not bars_up_to_now.empty
                else entry_date.strftime("%Y-%m-%d")
            )
            raw_ret = ((latest_px - entry_price) / entry_price) * 100.0
            trade_ret = raw_ret if side == "BUY" else -raw_ret
            return {
                "status": "ACTIVE",
                "exit_date": None,
                "exit_timestamp_ist": f"Active (M2M {latest_dt} 15:00 IST)",
                "exit_price": None,
                "current_price": round(latest_px, 2),
                "holding_trading_days": int(n_avail),
                "return_pct": round(trade_ret, 2),
                "exit_reason": f"Open ({n_avail}/{max_holding_days}d elapsed)",
            }

    # Fallback if ticker OHLCV slice is not in memory but study_df has factual 20d return
    calendar_age_days = max(0, int((cutoff_date - entry_date).days))
    if calendar_age_days >= 28 and fallback_ret_20d_pct is not None and not pd.isna(fallback_ret_20d_pct):
        raw_ret = float(fallback_ret_20d_pct)
        exit_px = entry_price * (1.0 + raw_ret / 100.0)
        trade_ret = raw_ret if side == "BUY" else -raw_ret
        est_exit_dt = roll_to_weekday(entry_date + pd.Timedelta(days=28))
        return {
            "status": "CLOSED",
            "exit_date": est_exit_dt.strftime("%Y-%m-%d"),
            "exit_timestamp_ist": f"{est_exit_dt.strftime('%Y-%m-%d')} 15:00:00 IST",
            "exit_price": round(exit_px, 2),
            "current_price": round(exit_px, 2),
            "holding_trading_days": max_holding_days,
            "return_pct": round(trade_ret, 2),
            "exit_reason": f"+{max_holding_days}D Horizon Completed (3PM IST)",
        }

    live_px = float(fallback_live_price) if fallback_live_price and not pd.isna(fallback_live_price) and fallback_live_price > 0 else entry_price
    raw_ret = ((live_px - entry_price) / entry_price) * 100.0
    trade_ret = raw_ret if side == "BUY" else -raw_ret
    return {
        "status": "ACTIVE",
        "exit_date": None,
        "exit_timestamp_ist": "Active (Live 3PM IST M2M)",
        "exit_price": None,
        "current_price": round(live_px, 2),
        "holding_trading_days": max(1, int(calendar_age_days * 5 / 7)),
        "return_pct": round(trade_ret, 2),
        "exit_reason": f"Open (<{max_holding_days}d)",
    }


def build_or_update_paper_trade_ledger(
    study_df: pd.DataFrame,
    live_signals_scored: Optional[pd.DataFrame] = None,
    top_buys: Optional[List[Dict[str, Any]]] = None,
    top_sells: Optional[List[Dict[str, Any]]] = None,
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build or incrementally update the 3:00 PM IST Weekday Paper Trade Ledger.
    1. Replays all historical weekday 3:00 PM IST sessions across the 10+ year factual event study
       (`study_df`, 2015–present), enforcing:
       - Weekdays Only (Mon–Fri at 15:00 IST)
       - Top 3 BUY (Prob >= 50%) and Top 3 SELL (Prob >= 50%) per session
       - Value-Change Gate (only trades when ticker price/value changed from prior snapshot)
       - No Duplicate Active Tickers (never buys/sells a ticker already open in Active trades)
    2. Evaluates today's live 3:00 PM IST weekday session (`top_buys` and `top_sells`),
       automatically executing today's paper trades if today is a weekday (Mon–Fri),
       current IST time is >= 15:00 IST (or manual trigger), value changed, and no duplicate active ticker.
    """
    price_map = _load_cached_price_map()
    cur_ist = now_ist()
    cutoff_ts = pd.Timestamp(cur_ist.date())

    # Fast path: if ledger already exists on disk and wasn't forced to rebuild, check if we can load & refresh M2M
    if not force_rebuild and LEDGER_JSON_PATH.exists():
        try:
            saved_payload = json.loads(LEDGER_JSON_PATH.read_text(encoding="utf-8"))
            saved_trades = saved_payload.get("trades", [])
            if len(saved_trades) >= 50:
                return _refresh_and_append_live_3pm_session(
                    saved_payload=saved_payload,
                    price_map=price_map,
                    top_buys=top_buys or [],
                    top_sells=top_sells or [],
                    max_holding_days=max_holding_days,
                )
        except Exception as exc:
            logger.debug("Rebuilding ledger due to load exception: %s", exc)

    if study_df is None or study_df.empty:
        return {"trades": [], "schedule_state": {}}

    df = study_df.copy()
    df["pub_dt_clean"] = pd.to_datetime(df["publication_dt"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["pub_dt_clean", "ticker"]).copy()

    # Map every filing to its weekday 3:00 PM IST execution date
    # If entry_trading_date is present and valid, use it; otherwise roll publication date to weekday
    if "entry_trading_date" in df.columns:
        entry_dt_s = pd.to_datetime(df["entry_trading_date"], errors="coerce").dt.tz_localize(None)
        df["exec_weekday_dt"] = entry_dt_s.fillna(df["pub_dt_clean"]).apply(roll_to_weekday)
    else:
        df["exec_weekday_dt"] = df["pub_dt_clean"].apply(roll_to_weekday)

    # Compute factual probabilities for each historical row
    sides, act_probs, buy_probs, sell_probs = [], [], [], []
    for _, r in df.iterrows():
        s_side, a_p, b_p, s_p = _compute_factual_prob_for_row(r)
        sides.append(s_side)
        act_probs.append(a_p)
        buy_probs.append(b_p)
        sell_probs.append(s_p)

    df["pt_side"] = sides
    df["pt_prob_pct"] = act_probs
    df["pt_buy_prob_pct"] = buy_probs
    df["pt_sell_prob_pct"] = sell_probs

    # Filter strictly to signals with >= 50.0% factual probability of success
    df = df[df["pt_prob_pct"] >= 50.0].copy()

    # Walk chronologically across every weekday session (Mon–Fri at 15:00 IST)
    trades: List[Dict[str, Any]] = []
    last_ticker_price: Dict[str, float] = {}
    last_batch_signature: str = ""
    skipped_duplicates_count = 0
    skipped_no_value_change_count = 0

    for session_dt, day_group in df.groupby("exec_weekday_dt", sort=True):
        if session_dt > cutoff_ts:
            continue
        # Ensure strict weekday (Mon–Fri)
        if session_dt.weekday() >= 5:
            continue

        # Which trades are currently ACTIVE at 3:00 PM IST on `session_dt`?
        active_buy_tickers = set()
        active_sell_tickers = set()
        for t in trades:
            t_entry = pd.Timestamp(t["entry_date"])
            t_exit = pd.Timestamp(t["exit_date"]) if t.get("exit_date") else pd.Timestamp("2099-12-31")
            if t_entry <= session_dt < t_exit:
                if t["side"] == "BUY":
                    active_buy_tickers.add(t["ticker"])
                else:
                    active_sell_tickers.add(t["ticker"])

        # Pick Top 3 BUY and Top 3 SELL for this weekday session
        day_buys = (
            day_group[day_group["pt_side"] == "BUY"]
            .sort_values(["pt_prob_pct", "traded_pct"], ascending=[False, False])
            .drop_duplicates(subset=["ticker"], keep="first")
            .head(3)
        )
        day_sells = (
            day_group[day_group["pt_side"] == "SELL"]
            .sort_values(["pt_prob_pct", "traded_pct"], ascending=[False, False])
            .drop_duplicates(subset=["ticker"], keep="first")
            .head(3)
        )

        candidates = pd.concat([day_buys, day_sells], ignore_index=True)
        if candidates.empty:
            continue

        # Check batch value signature
        sig_parts = [
            f"{r['ticker']}:{r['pt_side']}:{float(r.get('entry_price') or r.get('t0_close') or r.get('deal_price') or 0.0):.2f}"
            for _, r in candidates.iterrows()
        ]
        batch_sig = "|".join(sig_parts)
        if batch_sig == last_batch_signature:
            skipped_no_value_change_count += len(candidates)
            continue
        last_batch_signature = batch_sig

        for rank_idx, (_, crow) in enumerate(candidates.iterrows(), start=1):
            ticker = str(crow["ticker"]).strip().upper()
            yf_sym = str(crow.get("yf_ticker") or f"{ticker}.NS").strip()
            side = str(crow["pt_side"]).upper()

            # Determine factual 3:00 PM IST entry price
            ohlcv = price_map.get(yf_sym)
            entry_px = np.nan
            if ohlcv is not None and not ohlcv.empty:
                bars_on_or_before = ohlcv[ohlcv.index <= session_dt]
                if not bars_on_or_before.empty:
                    entry_px = float(bars_on_or_before["Close"].iloc[-1])
            if pd.isna(entry_px) or entry_px <= 0:
                for col_px in ("entry_price", "t0_close", "deal_price", "current_price"):
                    val_px = crow.get(col_px, np.nan)
                    if pd.notna(val_px) and float(val_px) > 0:
                        entry_px = float(val_px)
                        break
            if pd.isna(entry_px) or entry_px <= 0:
                continue

            # Rule: Only if change in value from previous snapshot for this ticker
            prev_px = last_ticker_price.get(ticker)
            if prev_px is not None and abs(entry_px - prev_px) < 1e-6:
                skipped_no_value_change_count += 1
                continue

            # Rule: Don't buy/sell duplicates if already active in same direction
            if side == "BUY" and ticker in active_buy_tickers:
                skipped_duplicates_count += 1
                continue
            if side == "SELL" and ticker in active_sell_tickers:
                skipped_duplicates_count += 1
                continue

            # Simulate factual exit / live M2M on weekday 3:00 PM IST bars
            sim = _simulate_trade_exit_on_ohlcv(
                side=side,
                entry_date=session_dt,
                entry_price=entry_px,
                ohlcv=ohlcv,
                max_holding_days=max_holding_days,
                cutoff_date=cutoff_ts,
                fallback_ret_20d_pct=crow.get("ret_20d_pct", np.nan),
                fallback_live_price=crow.get("current_price", entry_px),
            )

            qty = round(CAPITAL_PER_TRADE_INR / entry_px, 2)
            invested_inr = round(CAPITAL_PER_TRADE_INR, 2)
            ret_pct = float(sim["return_pct"])
            pnl_inr = round(invested_inr * (ret_pct / 100.0), 2)
            current_val_inr = round(invested_inr + pnl_inr, 2)

            trade_id = f"PT-{session_dt.strftime('%Y%m%d')}-{side[0]}-{ticker}"
            trade_record: Dict[str, Any] = {
                "trade_id": trade_id,
                "status": sim["status"],  # 'ACTIVE' or 'CLOSED'
                "side": side,             # 'BUY' or 'SELL'
                "side_badge": "🟢 BUY" if side == "BUY" else "🔴 SELL",
                "ticker": ticker,
                "yf_ticker": yf_sym,
                "stock_name": str(crow.get("stock_name") or ticker),
                "action_ticker_label": f"{'🟢 BUY' if side == 'BUY' else '🔴 SELL'} • {ticker}",
                "superstar_name": str(crow.get("superstar_name") or crow.get("client_name") or "Superstar Filing"),
                "category": str(crow.get("category") or "HNI / Institutional"),
                "event_source": str(crow.get("event_source") or "Exchange Filing"),
                "publication_date": str(crow.get("publication_date") or session_dt.strftime("%d %b %Y")),
                "entry_date": session_dt.strftime("%Y-%m-%d"),
                "entry_weekday": session_dt.strftime("%a"),
                "scheduled_exec_ist": f"{session_dt.strftime('%Y-%m-%d')} 15:00:00 IST",
                "entry_price": round(entry_px, 2),
                "exit_date": sim["exit_date"],
                "exit_timestamp_ist": sim["exit_timestamp_ist"],
                "exit_price": sim["exit_price"],
                "current_price": sim["current_price"],
                "quantity": qty,
                "invested_inr": invested_inr,
                "current_value_inr": current_val_inr,
                "pnl_inr": pnl_inr,
                "return_pct": ret_pct,
                "is_win": bool(ret_pct > 0.0),
                "prob_at_entry_pct": float(crow["pt_prob_pct"]),
                "holding_trading_days": int(sim["holding_trading_days"]),
                "exit_reason": sim["exit_reason"],
            }
            trades.append(trade_record)
            last_ticker_price[ticker] = round(entry_px, 2)
            if side == "BUY":
                active_buy_tickers.add(ticker)
            else:
                active_sell_tickers.add(ticker)

    payload = {
        "schema_version": "v1_3pm_ist_paper_trader",
        "generated_at_ist": cur_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "capital_per_trade_inr": CAPITAL_PER_TRADE_INR,
        "max_holding_days": max_holding_days,
        "skipped_duplicates_count": skipped_duplicates_count,
        "skipped_no_value_change_count": skipped_no_value_change_count,
        "last_ticker_price": last_ticker_price,
        "last_batch_signature": last_batch_signature,
        "trades": trades,
    }

    return _refresh_and_append_live_3pm_session(
        saved_payload=payload,
        price_map=price_map,
        top_buys=top_buys or [],
        top_sells=top_sells or [],
        max_holding_days=max_holding_days,
    )


def _refresh_and_append_live_3pm_session(
    saved_payload: Dict[str, Any],
    price_map: Dict[str, pd.DataFrame],
    top_buys: List[Dict[str, Any]],
    top_sells: List[Dict[str, Any]],
    max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
    force_manual_weekday_exec: bool = False,
) -> Dict[str, Any]:
    """
    1. Marks all ACTIVE trades to latest live market prices and closes any trade that has
       completed its holding period or hit target/stop-loss on a weekday at 3:00 PM IST.
    2. Checks if a scheduled 3:00 PM IST weekday run is due today (Mon–Fri, IST >= 15:00)
       and appends Top 3 BUY & Top 3 SELL trades if:
       - Today is a weekday (Mon–Fri)
       - Value has changed compared to the last 3:00 PM IST snapshot
       - Ticker is not already open as an ACTIVE duplicate trade
    """
    cur_ist = now_ist()
    today_str = cur_ist.strftime("%Y-%m-%d")
    is_weekday = cur_ist.weekday() < 5
    is_after_3pm = (cur_ist.hour > 15) or (cur_ist.hour == 15 and cur_ist.minute >= 0)

    trades: List[Dict[str, Any]] = list(saved_payload.get("trades", []))
    last_ticker_price: Dict[str, float] = dict(saved_payload.get("last_ticker_price", {}))
    last_batch_sig: str = str(saved_payload.get("last_batch_signature", ""))
    skipped_dups = int(saved_payload.get("skipped_duplicates_count", 0))
    skipped_no_chg = int(saved_payload.get("skipped_no_value_change_count", 0))

    # Build a fast lookup of current live prices from top_buys / top_sells + price_map
    live_px_lookup: Dict[str, float] = {}
    for item in (top_buys or []) + (top_sells or []):
        t_sym = str(item.get("ticker", "")).strip().upper()
        cmp_v = item.get("cmp", np.nan)
        if t_sym and pd.notna(cmp_v) and float(cmp_v) > 0:
            live_px_lookup[t_sym] = float(cmp_v)

    # 1. Refresh ACTIVE trades
    cutoff_ts = pd.Timestamp(cur_ist.date())
    for t in trades:
        if t.get("status") == "ACTIVE":
            yf_sym = t.get("yf_ticker") or f"{t['ticker']}.NS"
            ohlcv = price_map.get(yf_sym)
            fb_px = live_px_lookup.get(t["ticker"], t.get("current_price", t["entry_price"]))
            sim = _simulate_trade_exit_on_ohlcv(
                side=t["side"],
                entry_date=pd.Timestamp(t["entry_date"]),
                entry_price=float(t["entry_price"]),
                ohlcv=ohlcv,
                max_holding_days=max_holding_days,
                cutoff_date=cutoff_ts,
                fallback_live_price=fb_px,
            )
            t["status"] = sim["status"]
            t["exit_date"] = sim["exit_date"]
            t["exit_timestamp_ist"] = sim["exit_timestamp_ist"]
            t["exit_price"] = sim["exit_price"]
            t["current_price"] = sim["current_price"]
            t["holding_trading_days"] = sim["holding_trading_days"]
            t["return_pct"] = sim["return_pct"]
            t["pnl_inr"] = round(float(t["invested_inr"]) * (float(sim["return_pct"]) / 100.0), 2)
            t["current_value_inr"] = round(float(t["invested_inr"]) + float(t["pnl_inr"]), 2)
            t["is_win"] = bool(float(t["return_pct"]) > 0.0)
            t["exit_reason"] = sim["exit_reason"]

    # Determine currently ACTIVE tickers so we never buy/sell duplicates
    active_buy_tickers = {t["ticker"] for t in trades if t.get("status") == "ACTIVE" and t.get("side") == "BUY"}
    active_sell_tickers = {t["ticker"] for t in trades if t.get("status") == "ACTIVE" and t.get("side") == "SELL"}

    # Ensure that if the current Top 3 Buy / Top 3 Sell have not yet been entered as ACTIVE trades,
    # we evaluate them on the latest valid weekday 3:00 PM IST slot (today if Mon-Fri >= 15:00 IST,
    # or the most recent completed weekday 3:00 PM IST session!).
    latest_completed_3pm_dt = cur_ist.date()
    if not is_weekday or not is_after_3pm:
        # Roll back to most recent completed weekday 3:00 PM IST
        d_cursor = cur_ist.date()
        if is_weekday and not is_after_3pm:
            d_cursor -= dt.timedelta(days=1)
        while d_cursor.weekday() >= 5:
            d_cursor -= dt.timedelta(days=1)
        latest_completed_3pm_dt = d_cursor

    exec_date_str = latest_completed_3pm_dt.strftime("%Y-%m-%d")
    exec_weekday_str = latest_completed_3pm_dt.strftime("%a")

    candidates_live: List[Tuple[str, Dict[str, Any]]] = []
    for b_item in (top_buys or [])[:3]:
        if float(b_item.get("prob_pct", 0.0)) >= 50.0:
            candidates_live.append(("BUY", b_item))
    for s_item in (top_sells or [])[:3]:
        if float(s_item.get("prob_pct", 0.0)) >= 50.0:
            candidates_live.append(("SELL", s_item))

    new_trades_added = 0
    if candidates_live:
        sig_parts = [
            f"{item.get('ticker')}:{side}:{float(item.get('cmp') or item.get('deal_price') or 0.0):.2f}"
            for side, item in candidates_live
        ]
        live_sig = hashlib.sha256("|".join(sig_parts).encode("utf-8")).hexdigest()[:16]

        if live_sig != last_batch_sig or force_manual_weekday_exec:
            for side, item in candidates_live:
                ticker = str(item.get("ticker", "")).strip().upper()
                if not ticker:
                    continue
                entry_px = item.get("cmp", np.nan)
                if pd.isna(entry_px) or float(entry_px) <= 0:
                    entry_px = item.get("deal_price", np.nan)
                if pd.isna(entry_px) or float(entry_px) <= 0:
                    continue
                entry_px = float(entry_px)

                # Rule 1: Do not duplicate an already ACTIVE trade on the same side
                if side == "BUY" and ticker in active_buy_tickers:
                    skipped_dups += 1
                    continue
                if side == "SELL" and ticker in active_sell_tickers:
                    skipped_dups += 1
                    continue

                # Rule 2: If opposite side is currently ACTIVE, close it at 3:00 PM IST price first
                if side == "BUY" and ticker in active_sell_tickers:
                    for t in trades:
                        if t.get("status") == "ACTIVE" and t.get("side") == "SELL" and t.get("ticker") == ticker:
                            t["status"] = "CLOSED"
                            t["exit_date"] = exec_date_str
                            t["exit_timestamp_ist"] = f"{exec_date_str} 15:00:00 IST"
                            t["exit_price"] = round(entry_px, 2)
                            t["current_price"] = round(entry_px, 2)
                            raw_r = ((float(t["entry_price"]) - entry_px) / float(t["entry_price"])) * 100.0
                            t["return_pct"] = round(raw_r, 2)
                            t["pnl_inr"] = round(float(t["invested_inr"]) * (raw_r / 100.0), 2)
                            t["current_value_inr"] = round(float(t["invested_inr"]) + float(t["pnl_inr"]), 2)
                            t["is_win"] = bool(raw_r > 0.0)
                            t["exit_reason"] = "Reversed to Top-3 BUY at 3PM IST"
                    active_sell_tickers.discard(ticker)

                elif side == "SELL" and ticker in active_buy_tickers:
                    for t in trades:
                        if t.get("status") == "ACTIVE" and t.get("side") == "BUY" and t.get("ticker") == ticker:
                            t["status"] = "CLOSED"
                            t["exit_date"] = exec_date_str
                            t["exit_timestamp_ist"] = f"{exec_date_str} 15:00:00 IST"
                            t["exit_price"] = round(entry_px, 2)
                            t["current_price"] = round(entry_px, 2)
                            raw_r = ((entry_px - float(t["entry_price"])) / float(t["entry_price"])) * 100.0
                            t["return_pct"] = round(raw_r, 2)
                            t["pnl_inr"] = round(float(t["invested_inr"]) * (raw_r / 100.0), 2)
                            t["current_value_inr"] = round(float(t["invested_inr"]) + float(t["pnl_inr"]), 2)
                            t["is_win"] = bool(raw_r > 0.0)
                            t["exit_reason"] = "Reversed to Top-3 SELL at 3PM IST"
                    active_buy_tickers.discard(ticker)

                # Rule 3: Value change check vs last recorded price for this ticker
                prev_px = last_ticker_price.get(ticker)
                if prev_px is not None and abs(entry_px - float(prev_px)) < 1e-6 and not force_manual_weekday_exec:
                    skipped_no_chg += 1
                    continue

                qty = round(CAPITAL_PER_TRADE_INR / entry_px, 2)
                invested_inr = round(CAPITAL_PER_TRADE_INR, 2)
                trade_id = f"PT-{exec_date_str.replace('-', '')}-{side[0]}-{ticker}"

                # Avoid duplicate trade_id
                if any(t.get("trade_id") == trade_id for t in trades):
                    skipped_dups += 1
                    continue

                new_trade: Dict[str, Any] = {
                    "trade_id": trade_id,
                    "status": "ACTIVE",
                    "side": side,
                    "side_badge": "🟢 BUY" if side == "BUY" else "🔴 SELL",
                    "ticker": ticker,
                    "yf_ticker": f"{ticker}.NS",
                    "stock_name": str(item.get("stock") or ticker),
                    "action_ticker_label": f"{'🟢 BUY' if side == 'BUY' else '🔴 SELL'} • {ticker}",
                    "superstar_name": str(item.get("superstar") or "Superstar Filing"),
                    "category": str(item.get("category") or "HNI / Institutional"),
                    "event_source": str(item.get("channel") or "Exchange Filing"),
                    "publication_date": str(item.get("pub_date") or exec_date_str),
                    "entry_date": exec_date_str,
                    "entry_weekday": exec_weekday_str,
                    "scheduled_exec_ist": f"{exec_date_str} 15:00:00 IST",
                    "entry_price": round(entry_px, 2),
                    "exit_date": None,
                    "exit_timestamp_ist": f"Active (Opened {exec_date_str} 15:00 IST)",
                    "exit_price": None,
                    "current_price": round(entry_px, 2),
                    "quantity": qty,
                    "invested_inr": invested_inr,
                    "current_value_inr": invested_inr,
                    "pnl_inr": 0.0,
                    "return_pct": 0.0,
                    "is_win": False,
                    "prob_at_entry_pct": round(float(item.get("prob_pct", 55.0)), 1),
                    "holding_trading_days": 1,
                    "exit_reason": f"Open (Scheduled 3PM IST Top-3 {side})",
                }
                trades.append(new_trade)
                last_ticker_price[ticker] = round(entry_px, 2)
                new_trades_added += 1
                if side == "BUY":
                    active_buy_tickers.add(ticker)
                else:
                    active_sell_tickers.add(ticker)

            last_batch_sig = live_sig
        else:
            skipped_no_chg += len(candidates_live)

    # Sort trades most recent first (ACTIVE on top, then by entry_date descending)
    trades.sort(
        key=lambda x: (0 if x.get("status") == "ACTIVE" else 1, -int(str(x.get("entry_date", "1970-01-01")).replace("-", ""))),
    )

    next_run_dt = next_weekday_3pm_ist(cur_ist)
    schedule_state = {
        "schedule_rule": "Every Weekday (Mon–Fri) at 15:00 IST (3:00 PM IST)",
        "value_change_gate": "Active (Executes ONLY if live price/signal value changed)",
        "duplicate_guard": "Active (Never duplicates an already ACTIVE ticker)",
        "current_ist_time": cur_ist.strftime("%Y-%m-%d %H:%M:%S IST (%a)"),
        "is_weekday_today": is_weekday,
        "last_completed_session_ist": f"{exec_date_str} 15:00:00 IST ({exec_weekday_str})",
        "next_scheduled_run_ist": next_run_dt.strftime("%Y-%m-%d %H:%M:%S IST (%a)"),
        "new_trades_triggered_last_check": new_trades_added,
        "skipped_duplicates_total": skipped_dups,
        "skipped_no_value_change_total": skipped_no_chg,
    }

    updated_payload = {
        "schema_version": "v1_3pm_ist_paper_trader",
        "generated_at_ist": cur_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "capital_per_trade_inr": CAPITAL_PER_TRADE_INR,
        "max_holding_days": max_holding_days,
        "skipped_duplicates_count": skipped_dups,
        "skipped_no_value_change_count": skipped_no_chg,
        "last_ticker_price": last_ticker_price,
        "last_batch_signature": last_batch_sig,
        "schedule_state": schedule_state,
        "trades": trades,
    }

    try:
        LEDGER_JSON_PATH.write_text(json.dumps(updated_payload, indent=2), encoding="utf-8")
        SCHEDULE_STATE_PATH.write_text(json.dumps(schedule_state, indent=2), encoding="utf-8")
        if trades:
            pd.DataFrame(trades).to_csv(LEDGER_CSV_PATH, index=False)
    except Exception as exc:
        logger.debug("Could not persist paper trade ledger: %s", exc)

    return updated_payload


def compute_paper_trade_performance_summary(ledger_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive KPIs and the multi-horizon performance indicator table:
    - Total Invested (Active + Closed), Active Invested, Closed Invested
    - Count of Active & Closed Trades (Total, Active, Closed, Active BUY/SELL, Closed BUY/SELL)
    - Buy & Sell Tickers against Active and Closed trades
    - Performance of Total (Active + Closed)
    - Performance of Closed Trades (Success Rate %, Return %, Net PnL ₹)
    - 9 Time Horizon Performance Buckets:
      1 Week (7D), 2 Weeks (14D), 1 Month (30D), 3 Months (91D),
      1 Year (365D), 2 Years (730D), 3 Years (1095D), 5 Years (1825D), 10 Years (3650D).
    """
    trades = ledger_payload.get("trades", [])
    if not trades:
        return {
            "kpis": {},
            "horizon_df": pd.DataFrame(),
            "ledger_df": pd.DataFrame(),
            "active_df": pd.DataFrame(),
            "closed_df": pd.DataFrame(),
        }

    df = pd.DataFrame(trades)
    df["entry_dt"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_date"], errors="coerce")

    active_df = df[df["status"] == "ACTIVE"].copy()
    closed_df = df[df["status"] == "CLOSED"].copy()

    def _summarize_slice(sub: pd.DataFrame, label: str, window_desc: str) -> Dict[str, Any]:
        if sub.empty:
            return {
                "Performance Horizon": label,
                "Window": window_desc,
                "Total Trades": 0,
                "Active Trades": 0,
                "Closed Trades": 0,
                "Closed Success Rate (Win %)": np.nan,
                "Total Positive Rate (%)": np.nan,
                "Total Invested (₹ Lakhs)": 0.0,
                "Current / Exit Value (₹ Lakhs)": 0.0,
                "Net PnL (₹ Lakhs)": 0.0,
                "Portfolio Return (%)": 0.0,
                "Closed Return (%)": np.nan,
                "Avg Return / Trade (%)": np.nan,
                "BUY Tickers (Count & List)": "-",
                "SELL Tickers (Count & List)": "-",
            }

        sub_act = sub[sub["status"] == "ACTIVE"]
        sub_cls = sub[sub["status"] == "CLOSED"]

        n_tot = len(sub)
        n_act = len(sub_act)
        n_cls = len(sub_cls)

        tot_inv = float(sub["invested_inr"].sum())
        tot_val = float(sub["current_value_inr"].sum())
        tot_pnl = float(sub["pnl_inr"].sum())
        port_ret_pct = ((tot_pnl / tot_inv) * 100.0) if tot_inv > 0 else 0.0
        avg_ret_pct = float(sub["return_pct"].mean())

        cls_win_rate = float((sub_cls["return_pct"] > 0).mean() * 100.0) if n_cls > 0 else np.nan
        cls_inv = float(sub_cls["invested_inr"].sum()) if n_cls > 0 else 0.0
        cls_pnl = float(sub_cls["pnl_inr"].sum()) if n_cls > 0 else 0.0
        cls_ret_pct = ((cls_pnl / cls_inv) * 100.0) if cls_inv > 0 else np.nan

        # If a very recent window (like 1W) has mostly Active trades and 0 Closed yet,
        # also report the live Mark-to-Market Win % alongside Closed Win %
        tot_pos_rate = float((sub["return_pct"] > 0).mean() * 100.0) if n_tot > 0 else np.nan
        effective_win_rate = cls_win_rate if pd.notna(cls_win_rate) else tot_pos_rate

        buy_sub = sub[sub["side"] == "BUY"]
        sell_sub = sub[sub["side"] == "SELL"]
        buy_syms = list(dict.fromkeys(buy_sub["ticker"].tolist()))
        sell_syms = list(dict.fromkeys(sell_sub["ticker"].tolist()))

        buy_preview = ", ".join(buy_syms[:8]) + (f" (+{len(buy_syms)-8} more)" if len(buy_syms) > 8 else "")
        sell_preview = ", ".join(sell_syms[:8]) + (f" (+{len(sell_syms)-8} more)" if len(sell_syms) > 8 else "")

        return {
            "Performance Horizon": label,
            "Window": window_desc,
            "Total Trades": int(n_tot),
            "Active Trades": int(n_act),
            "Closed Trades": int(n_cls),
            "Closed Success Rate (Win %)": round(effective_win_rate, 1) if pd.notna(effective_win_rate) else np.nan,
            "Total Positive Rate (%)": round(tot_pos_rate, 1) if pd.notna(tot_pos_rate) else np.nan,
            "Total Invested (₹ Lakhs)": round(tot_inv / 1e5, 2),
            "Current / Exit Value (₹ Lakhs)": round(tot_val / 1e5, 2),
            "Net PnL (₹ Lakhs)": round(tot_pnl / 1e5, 2),
            "Portfolio Return (%)": round(port_ret_pct, 2),
            "Closed Return (%)": round(cls_ret_pct, 2) if pd.notna(cls_ret_pct) else round(port_ret_pct, 2),
            "Avg Return / Trade (%)": round(avg_ret_pct, 2),
            "BUY Tickers (Count & List)": f"{len(buy_sub)} trades [{buy_preview}]" if buy_syms else "0",
            "SELL Tickers (Count & List)": f"{len(sell_sub)} trades [{sell_preview}]" if sell_syms else "0",
        }

    ref_date = pd.Timestamp(now_ist().date())
    horizon_rows: List[Dict[str, Any]] = [
        _summarize_slice(df, "📌 Total (Active + Closed)", "All Scheduled Trades (Full Book)"),
        _summarize_slice(closed_df, "✅ Closed Trades Only", "Completed & Realized Trades"),
        _summarize_slice(active_df, "🟢 Active Trades Only", "Currently Open Positions (Live M2M)"),
    ]

    time_windows = [
        ("1 Week Performance", 7, "Last 7 Calendar Days (5 Trading Days)"),
        ("2 Week Performance", 14, "Last 14 Calendar Days (10 Trading Days)"),
        ("1 Month Performance", 30, "Last 30 Calendar Days (~21 Trading Days)"),
        ("3 Month Performance", 91, "Last 91 Calendar Days (~63 Trading Days)"),
        ("1 Year Performance", 365, "Last 1 Year (365 Days)"),
        ("2 Year Performance", 730, "Last 2 Years (730 Days)"),
        ("3 Year Performance", 1095, "Last 3 Years (1,095 Days)"),
        ("5 Year Performance", 1825, "Last 5 Years (1,825 Days)"),
        ("10 Year Performance", 3650, "Last 10 Years (3,650 Days)"),
    ]

    for h_label, days_back, w_desc in time_windows:
        cutoff = ref_date - pd.Timedelta(days=days_back)
        # Include trades entered on or after cutoff OR closed on or after cutoff
        sub_w = df[(df["entry_dt"] >= cutoff) | (df["exit_dt"].notna() & (df["exit_dt"] >= cutoff))].copy()
        horizon_rows.append(_summarize_slice(sub_w, h_label, w_desc))

    horizon_df = pd.DataFrame(horizon_rows)

    # Headline KPIs
    tot_invested_inr = float(df["invested_inr"].sum())
    act_invested_inr = float(active_df["invested_inr"].sum()) if not active_df.empty else 0.0
    cls_invested_inr = float(closed_df["invested_inr"].sum()) if not closed_df.empty else 0.0

    tot_pnl_inr = float(df["pnl_inr"].sum())
    act_pnl_inr = float(active_df["pnl_inr"].sum()) if not active_df.empty else 0.0
    cls_pnl_inr = float(closed_df["pnl_inr"].sum()) if not closed_df.empty else 0.0

    tot_ret_pct = (tot_pnl_inr / tot_invested_inr * 100.0) if tot_invested_inr > 0 else 0.0
    cls_ret_pct = (cls_pnl_inr / cls_invested_inr * 100.0) if cls_invested_inr > 0 else 0.0
    act_ret_pct = (act_pnl_inr / act_invested_inr * 100.0) if act_invested_inr > 0 else 0.0

    closed_win_rate_pct = float((closed_df["return_pct"] > 0).mean() * 100.0) if not closed_df.empty else 0.0
    closed_wins_count = int((closed_df["return_pct"] > 0).sum()) if not closed_df.empty else 0
    closed_losses_count = int((closed_df["return_pct"] <= 0).sum()) if not closed_df.empty else 0

    cls_buy_df = closed_df[closed_df["side"] == "BUY"]
    cls_sell_df = closed_df[closed_df["side"] == "SELL"]
    cls_buy_win_pct = float((cls_buy_df["return_pct"] > 0).mean() * 100.0) if not cls_buy_df.empty else 0.0
    cls_sell_win_pct = float((cls_sell_df["return_pct"] > 0).mean() * 100.0) if not cls_sell_df.empty else 0.0

    act_buy_tickers = active_df[active_df["side"] == "BUY"]["ticker"].tolist() if not active_df.empty else []
    act_sell_tickers = active_df[active_df["side"] == "SELL"]["ticker"].tolist() if not active_df.empty else []

    kpis = {
        "total_invested_inr": tot_invested_inr,
        "total_invested_cr": round(tot_invested_inr / 1e7, 2),
        "total_invested_lakhs": round(tot_invested_inr / 1e5, 2),
        "active_invested_lakhs": round(act_invested_inr / 1e5, 2),
        "closed_invested_lakhs": round(cls_invested_inr / 1e5, 2),
        "total_trades_count": len(df),
        "active_trades_count": len(active_df),
        "closed_trades_count": len(closed_df),
        "active_buy_count": len(act_buy_tickers),
        "active_sell_count": len(act_sell_tickers),
        "closed_buy_count": len(cls_buy_df),
        "closed_sell_count": len(cls_sell_df),
        "active_buy_tickers": act_buy_tickers,
        "active_sell_tickers": act_sell_tickers,
        "closed_win_rate_pct": round(closed_win_rate_pct, 1),
        "closed_wins_count": closed_wins_count,
        "closed_losses_count": closed_losses_count,
        "closed_buy_win_pct": round(cls_buy_win_pct, 1),
        "closed_sell_win_pct": round(cls_sell_win_pct, 1),
        "total_return_pct": round(tot_ret_pct, 2),
        "total_pnl_lakhs": round(tot_pnl_inr / 1e5, 2),
        "closed_return_pct": round(cls_ret_pct, 2),
        "closed_pnl_lakhs": round(cls_pnl_inr / 1e5, 2),
        "active_return_pct": round(act_ret_pct, 2),
        "active_pnl_lakhs": round(act_pnl_inr / 1e5, 2),
    }

    # Format clean display ledger DataFrame showing Buy/Sell tickers against each trade (Active or Closed)
    ledger_display = pd.DataFrame({
        "Trade Status": df["status"].map(lambda s: "🟢 ACTIVE" if s == "ACTIVE" else "✅ CLOSED"),
        "Buy / Sell & Ticker": df["action_ticker_label"],
        "Side": df["side_badge"],
        "Ticker": df["ticker"],
        "Stock Name": df["stock_name"],
        "Outcome": [
            "⏳ Open (Live M2M)" if st_val == "ACTIVE" else ("🏆 WIN" if w_val else "❌ LOSS")
            for st_val, w_val in zip(df["status"], df["is_win"])
        ],
        "Return (%)": df["return_pct"],
        "Net PnL (₹)": df["pnl_inr"],
        "Invested (₹)": df["invested_inr"],
        "Current / Exit Value (₹)": df["current_value_inr"],
        "Entry Price (₹)": df["entry_price"],
        "Exit / Live Price (₹)": df["current_price"],
        "Entry Prob (%)": df["prob_at_entry_pct"],
        "Scheduled Entry (3PM IST)": df["scheduled_exec_ist"],
        "Exit / M2M Timestamp": df["exit_timestamp_ist"],
        "Days Held": df["holding_trading_days"],
        "Superstar / Trigger": df["superstar_name"],
        "Category": df["category"],
        "Exit / Status Note": df["exit_reason"],
    })

    return {
        "kpis": kpis,
        "horizon_df": horizon_df,
        "ledger_df": ledger_display,
        "active_df": ledger_display[ledger_display["Trade Status"] == "🟢 ACTIVE"].reset_index(drop=True),
        "closed_df": ledger_display[ledger_display["Trade Status"] == "✅ CLOSED"].reset_index(drop=True),
    }


def sync_daily_live_candidates_and_tables(force_network: bool = False) -> Dict[str, Any]:
    """
    Synchronizes all live data sources, discovers any new Superstar/Exchange candidate filings,
    and refreshes live CMPs across precomputed study tables and OHLCV caches.
    Runs automatically every day (even if the app was not open, via background daemon or
    on startup catch-up) and on-demand when any tab's 'Refresh Live CMP' button is clicked.
    """
    from live_data_engine import (
        fetch_live_bulk_block_deals,
        fetch_live_cmp_batch,
        fetch_live_sast_insider_deals,
        get_indian_market_status_ist,
    )

    sync_state_path = DATA_DIR / "daily_candidate_sync_state.json"
    cur_ist = now_ist()
    today_str = cur_ist.strftime("%Y-%m-%d")
    mkt_status = get_indian_market_status_ist()

    existing_state: Dict[str, Any] = {}
    if sync_state_path.exists():
        try:
            existing_state = json.loads(sync_state_path.read_text(encoding="utf-8"))
        except Exception:
            existing_state = {}

    # Skip network hit if already synced today within the last 15 minutes and not forced
    if not force_network and existing_state.get("last_sync_date_ist") == today_str:
        last_ep = float(existing_state.get("last_sync_epoch", 0.0))
        if (cur_ist.timestamp() - last_ep) < 900.0:
            existing_state["market_status"] = mkt_status
            return existing_state

    new_candidates_added = 0
    cmp_updated_count = 0

    try:
        study_path = DATA_DIR / "historical_event_study_t1_open.csv"
        events_path = DATA_DIR / "historical_superstar_events.csv"
        study_df = pd.read_csv(study_path) if study_path.exists() else pd.DataFrame()

        # 1. Check live NSE/BSE Bulk/Block and SAST/Insider feeds for any brand-new candidate filings
        bulk_live = fetch_live_bulk_block_deals(use_cache=not force_network)
        sast_live = fetch_live_sast_insider_deals(use_cache=not force_network)

        existing_keys = set()
        if not study_df.empty and {"ticker", "publication_date", "direction"}.issubset(study_df.columns):
            for _, r in study_df[["ticker", "publication_date", "direction"]].dropna().iterrows():
                existing_keys.add(
                    f"{str(r['ticker']).strip().upper()}|{str(r['publication_date']).strip()}|{str(r['direction']).strip().upper()}"
                )

        new_rows: List[Dict[str, Any]] = []
        for src_df, src_label in [(bulk_live, "Bulk / Block Deal"), (sast_live, "SAST / Insider Filing")]:
            if src_df is None or src_df.empty:
                continue
            for _, row in src_df.head(60).iterrows():
                t_sym = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
                pub_d = str(row.get("publication_date") or row.get("date") or today_str).strip()
                raw_dir = str(row.get("direction") or row.get("action") or "BUY").strip().upper()
                dir_norm = "BUY" if ("BUY" in raw_dir or "ACQ" in raw_dir) else "SELL"
                if not t_sym or len(t_sym) > 15:
                    continue
                k = f"{t_sym}|{pub_d}|{dir_norm}"
                if k in existing_keys:
                    continue
                existing_keys.add(k)

                # If this ticker already has historical state statistics in study_df, inherit its stock state
                template_row: Dict[str, Any] = {}
                if not study_df.empty and "ticker" in study_df.columns:
                    m_sub = study_df[study_df["ticker"].astype(str).str.upper() == t_sym]
                    if not m_sub.empty:
                        template_row = m_sub.iloc[0].to_dict()

                deal_px = float(row.get("deal_price") or row.get("price") or template_row.get("current_price") or 0.0)
                if deal_px <= 0:
                    continue

                nr = dict(template_row) if template_row else {}
                nr.update({
                    "ticker": t_sym,
                    "yf_ticker": f"{t_sym}.NS",
                    "stock_name": str(row.get("stock_name") or row.get("company") or template_row.get("stock_name") or t_sym),
                    "superstar_name": str(row.get("client_name") or row.get("superstar_name") or row.get("acquirer_name") or "Institutional / HNI Filing"),
                    "category": str(row.get("category") or template_row.get("category") or "Institutional"),
                    "event_source": src_label,
                    "direction": dir_norm,
                    "publication_date": pub_d,
                    "event_date": today_str,
                    "entry_date": today_str,
                    "deal_price": deal_px,
                    "t0_close": deal_px,
                    "entry_price": deal_px,
                    "current_price": deal_px,
                    "live_vs_deal_pct": 0.0,
                    "stake_pct": float(row.get("traded_pct") or row.get("stake_pct") or 0.5),
                    "traded_pct": float(row.get("traded_pct") or row.get("stake_pct") or 0.5),
                })
                new_rows.append(nr)

        if new_rows and not study_df.empty:
            new_df = pd.DataFrame(new_rows)
            for col in study_df.columns:
                if col not in new_df.columns:
                    new_df[col] = np.nan
            study_df = pd.concat([new_df[study_df.columns], study_df], ignore_index=True)
            new_candidates_added = len(new_rows)

        # 2. Batch-fetch live CMPs for top candidate stocks + active paper trade stocks
        target_yf_syms: List[str] = []
        if not study_df.empty and "yf_ticker" in study_df.columns:
            # Take most recent 60 unique tickers in study_df so live signals have fresh CMP
            recent_syms = (
                study_df.sort_values("entry_date", ascending=False)["yf_ticker"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()[:60]
            )
            target_yf_syms.extend(recent_syms)

        if LEDGER_JSON_PATH.exists():
            try:
                led = json.loads(LEDGER_JSON_PATH.read_text(encoding="utf-8"))
                for tr in led.get("trades", []):
                    if tr.get("status") == "ACTIVE" and tr.get("yf_ticker"):
                        target_yf_syms.append(str(tr["yf_ticker"]))
            except Exception:
                pass

        target_yf_syms = list(dict.fromkeys(target_yf_syms))
        live_quotes = fetch_live_cmp_batch(target_yf_syms, update_disk_cache=True)
        cmp_updated_count = len(live_quotes)

        # 3. Update current_price and live_vs_deal_pct in study_df and persist
        if not study_df.empty and live_quotes:
            for yf_s, qinfo in live_quotes.items():
                cmp_v = float(qinfo.get("cmp", np.nan))
                if pd.isna(cmp_v) or cmp_v <= 0:
                    continue
                mask = study_df["yf_ticker"].astype(str) == yf_s
                if mask.any():
                    study_df.loc[mask, "current_price"] = round(cmp_v, 2)
                    base_px = study_df.loc[mask, "deal_price"].fillna(study_df.loc[mask, "t0_close"])
                    valid_base = base_px > 0
                    if valid_base.any():
                        idx_valid = base_px[valid_base].index
                        study_df.loc[idx_valid, "live_vs_deal_pct"] = (
                            (cmp_v - base_px.loc[idx_valid]) / base_px.loc[idx_valid] * 100.0
                        ).round(2)

            for slug in ["t1_open", "t0_close", "deal_price"]:
                p_out = DATA_DIR / f"historical_event_study_{slug}.csv"
                if p_out.exists() or slug == "t1_open":
                    study_df.to_csv(p_out, index=False)

            if new_candidates_added > 0 and events_path.exists():
                try:
                    ev_df = pd.read_csv(events_path)
                    ev_cols = [c for c in ev_df.columns if c in study_df.columns]
                    study_df[ev_cols].drop_duplicates().to_csv(events_path, index=False)
                except Exception:
                    pass

        # 4. Also refresh live CMPs for 25 Indian Indices & 42 Sectoral ETFs
        try:
            from market_sr_etf_engine import refresh_indices_and_etf_live_cmps
            idx_etf_res = refresh_indices_and_etf_live_cmps()
            cmp_updated_count += int(idx_etf_res.get("indices_cmp_updated", 0)) + int(idx_etf_res.get("etfs_cmp_updated", 0))
        except Exception:
            pass
    except Exception as exc:
        logger.debug("Daily candidate & CMP sync warning: %s", exc)

    sync_payload = {
        "last_sync_date_ist": today_str,
        "last_sync_timestamp_ist": cur_ist.strftime("%Y-%m-%d %H:%M:%S IST (%a)"),
        "last_sync_epoch": round(cur_ist.timestamp(), 2),
        "new_candidates_discovered": new_candidates_added,
        "live_cmps_refreshed_count": cmp_updated_count,
        "auto_refresh_policy": "Daily Auto-Sync (09:20 IST Market Open & 15:00 IST Paper Trade + Offline Catch-Up on Wake)",
        "market_status": mkt_status,
    }
    try:
        sync_state_path.write_text(json.dumps(sync_payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    return sync_payload


def ensure_3pm_ist_background_scheduler() -> None:
    """
    Start a lightweight singleton daemon thread that:
    1. Immediately performs a daily candidate & paper-trade catch-up check if the app was
       closed/asleep across one or more weekdays.
    2. Wakes up automatically on every weekday (Monday–Friday) at:
       - 09:20 IST (Market Open Candidate & Live CMP Auto-Refresh)
       - 15:00 IST / 3:00 PM IST (Scheduled Top-3 Buy/Sell Paper Trade & Table Refresh)
    even when no browser tab is actively clicking buttons.
    """
    global _SCHEDULER_STARTED
    with _SCHEDULER_LOCK:
        if _SCHEDULER_STARTED:
            return
        _SCHEDULER_STARTED = True

    def _next_scheduled_event_ist(cur: dt.datetime) -> Tuple[dt.datetime, str]:
        c_920 = cur.replace(hour=9, minute=20, second=0, microsecond=0)
        c_1500 = cur.replace(hour=15, minute=0, second=0, microsecond=0)
        if cur.weekday() < 5:
            if cur < c_920:
                return c_920, "09:20 IST Morning Candidate & CMP Sync"
            if cur < c_1500:
                return c_1500, "15:00 IST Scheduled Paper Trade & Table Sync"
        nxt = (cur + dt.timedelta(days=1)).replace(hour=9, minute=20, second=0, microsecond=0)
        while nxt.weekday() >= 5:
            nxt += dt.timedelta(days=1)
        return nxt, "09:20 IST Morning Candidate & CMP Sync"

    def _worker() -> None:
        # Startup catch-up check in background if today's sync or paper trade was missed while app was asleep
        try:
            sync_state_path = DATA_DIR / "daily_candidate_sync_state.json"
            today_str = now_ist().strftime("%Y-%m-%d")
            need_catchup = True
            if sync_state_path.exists():
                st_data = json.loads(sync_state_path.read_text(encoding="utf-8"))
                if st_data.get("last_sync_date_ist") == today_str:
                    need_catchup = False
            if need_catchup:
                run_cli_scheduled_paper_trade(force_rebuild=False, sync_live_candidates=True)
        except Exception as exc:
            logger.debug("Startup catch-up check warning: %s", exc)

        while True:
            try:
                now_dt = now_ist()
                nxt_dt, _ = _next_scheduled_event_ist(now_dt)
                sleep_secs = max(5.0, (nxt_dt - now_dt).total_seconds())
                while sleep_secs > 0:
                    step = min(300.0, sleep_secs)
                    threading.Event().wait(step)
                    now_dt = now_ist()
                    sleep_secs = (nxt_dt - now_dt).total_seconds()
                # Trigger candidate discovery + live CMP refresh + 3:00 PM IST paper trade evaluation
                run_cli_scheduled_paper_trade(force_rebuild=False, sync_live_candidates=True)
            except Exception as exc:
                logger.debug("Background scheduler warning: %s", exc)
                threading.Event().wait(60.0)

    t = threading.Thread(target=_worker, name="PaperTradeAndCandidateAutoScheduler", daemon=True)
    t.start()


def run_cli_scheduled_paper_trade(
    force_rebuild: bool = False,
    sync_live_candidates: bool = True,
) -> Dict[str, Any]:
    """
    Standalone CLI entry point for cron / background thread to sync new daily candidates,
    refresh live CMPs, and execute the 3:00 PM IST weekday paper trade using live data.
    """
    from impact_analyzer import ImmediateEntrySuccessPredictor, get_top3_buy_and_sell_recommendations
    if sync_live_candidates:
        sync_daily_live_candidates_and_tables(force_network=force_rebuild)
    study_path = DATA_DIR / "historical_event_study_t1_open.csv"
    if not study_path.exists():
        return {}
    study_df = pd.read_csv(study_path)
    predictor = ImmediateEntrySuccessPredictor(target_horizon_days=20)
    predictor.fit(study_df)
    live_scored = predictor.score_live_signals_table(study_df, max_rows=400)
    top_buys, top_sells = get_top3_buy_and_sell_recommendations(live_scored, horizon_days=20)
    return build_or_update_paper_trade_ledger(
        study_df=study_df,
        live_signals_scored=live_scored,
        top_buys=top_buys,
        top_sells=top_sells,
        force_rebuild=force_rebuild,
    )


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run 3:00 PM IST Weekday Paper Trade & Daily Candidate Sync Engine")
    parser.add_argument("--rebuild", action="store_true", help="Force full 10-year + live rebuild of paper trade ledger")
    args = parser.parse_args()
    payload = run_cli_scheduled_paper_trade(force_rebuild=args.rebuild, sync_live_candidates=True)
    summary = compute_paper_trade_performance_summary(payload)
    print("KPIs:", json.dumps(summary.get("kpis", {}), indent=2))
    if not summary.get("horizon_df", pd.DataFrame()).empty:
        print("\nMulti-Horizon Performance Table:")
        print(summary["horizon_df"].to_string(index=False))


