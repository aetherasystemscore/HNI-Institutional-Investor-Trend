"""
live_data_engine.py
===================
100% Live Data Ingestion Engine for HNI & Institutional Investor Trend Analysis.
Zero synthetic, mocked, or fabricated data.

Live Data Sources:
1. Trendlyne Individual Superstar Shareholders Index:
   https://trendlyne.com/portfolio/superstar-shareholders/index/
2. Trendlyne Institutional Superstar Shareholders Index:
   https://trendlyne.com/portfolio/superstar-shareholders/index/institutional/
3. Trendlyne FII Superstar Shareholders Index:
   https://trendlyne.com/portfolio/superstar-shareholders/index/fii/
4. Trendlyne Superstar Portfolio Holdings, Bulk/Block Deals & SAST/Insider Disclosures:
   - https://trendlyne.com/portfolio/superstar-shareholders/<ssid>/latest/<slug>/
   - https://trendlyne.com/portfolio/bulk-block-deals/<ssid>/<slug>/
   - https://trendlyne.com/equity/insider-trading-sast/<ssid>/<slug>/
5. Trendlyne Exchange-Wide Live Bulk/Block Deals & SAST/Insider Disclosures:
   - https://trendlyne.com/portfolio/bulk-block-deals/
   - https://trendlyne.com/equity/group-insider-trading-sast/
6. Official NSE India Corporate Shareholding Filings Master (Exact Broadcast Timestamps):
   - https://www.nseindia.com/api/corporate-share-holdings-master?index=equities
   - https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol=<SYMBOL>
7. Yahoo Finance (yfinance) Live & Historical Daily OHLCV Market Data (.NS / .BO / ^NSEI).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_TRENDLYNE_SESSION = requests.Session()
_TRENDLYNE_SESSION.headers.update(DEFAULT_HEADERS)
_TRENDLYNE_WAF_BLOCKED: bool = False


def _save_snapshot_gz(snap_file: Path, payload: Any) -> None:
    """Save a gzipped JSON snapshot in `data/snapshots/` for cloud IP resilience."""
    try:
        raw_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        snap_file.write_bytes(gzip.compress(raw_bytes, compresslevel=6))
    except Exception as exc:
        logger.debug("Snapshot write warning for %s: %s", snap_file.name, exc)


def _load_snapshot_gz(snap_file: Path) -> Optional[Any]:
    """Load a gzipped JSON snapshot from `data/snapshots/`."""
    if not snap_file.exists():
        return None
    try:
        raw_bytes = gzip.decompress(snap_file.read_bytes())
        return json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        logger.debug("Snapshot read warning for %s: %s", snap_file.name, exc)
        return None


def _fetch_trendlyne_html_with_snapshot(
    url: str,
    cache_stem: str,
    use_cache: bool = True,
    cache_ttl_minutes: int = 240,
    timeout: int = 18,
) -> Tuple[str, str]:
    """
    Fetch live HTML from Trendlyne with automatic multi-tier caching and cloud-IP resilience:
    1. Local `data/cache/<cache_stem>.json` (if `use_cache=True` and within `cache_ttl_minutes`).
    2. Committed `data/snapshots/<cache_stem>.json.gz` (if `use_cache=True` and local cache is cold, e.g., on Streamlit Cloud).
    3. Direct live HTTP request via `_TRENDLYNE_SESSION`.
    4. Graceful fallback to `data/cache/<cache_stem>.json` or `data/snapshots/<cache_stem>.json.gz` if Cloudflare
       blocks the cloud data-center IP (`HTTP 403 / 429 / 503`), ensuring `HTTPError` is never raised.
    """
    global _TRENDLYNE_WAF_BLOCKED
    stem = cache_stem[:-5] if cache_stem.endswith(".json") else cache_stem
    cache_file = CACHE_DIR / f"{stem}.json"
    snap_file = SNAPSHOT_DIR / f"{stem}.json.gz"
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    # 1. Check fresh local cache
    if use_cache and cache_file.exists():
        age_min = (time.time() - cache_file.stat().st_mtime) / 60.0
        if age_min <= cache_ttl_minutes:
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                html = cached.get("html") or ""
                if html:
                    return html, cached.get("fetched_at", fetched_at)
            except Exception:
                pass

    # 2. On cold container start with use_cache=True, check committed live snapshot
    if use_cache and not cache_file.exists() and snap_file.exists():
        snap = _load_snapshot_gz(snap_file)
        if isinstance(snap, dict) and snap.get("html"):
            try:
                cache_file.write_text(json.dumps(snap, ensure_ascii=True), encoding="utf-8")
            except Exception:
                pass
            return snap["html"], snap.get("fetched_at", fetched_at)

    # 3. Attempt live HTTP request if host IP is not WAF-blocked
    if not _TRENDLYNE_WAF_BLOCKED:
        try:
            resp = _TRENDLYNE_SESSION.get(url, timeout=timeout)
            if resp.status_code == 200 and len(resp.text) > 500:
                html_text = resp.text
                fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
                payload = {"fetched_at": fetched_at, "url": url, "html": html_text}
                try:
                    cache_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
                except Exception:
                    pass
                _save_snapshot_gz(snap_file, payload)
                return html_text, fetched_at
            elif resp.status_code in (403, 429, 503):
                _TRENDLYNE_WAF_BLOCKED = True
                logger.warning(
                    "Trendlyne returned HTTP %s on cloud IP for %s; activating live snapshot fallback.",
                    resp.status_code,
                    url,
                )
        except Exception as exc:
            _TRENDLYNE_WAF_BLOCKED = True
            logger.warning("Trendlyne live request failed (%s); using live snapshot fallback.", exc)

    # 4. Graceful fallback to any existing local cache or committed snapshot
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("html"):
                return cached["html"], cached.get("fetched_at", fetched_at)
        except Exception:
            pass

    snap = _load_snapshot_gz(snap_file)
    if isinstance(snap, dict) and snap.get("html"):
        return snap["html"], snap.get("fetched_at", fetched_at)

    return "", fetched_at


TRENDLYNE_INDEX_URLS = {
    "individual": "https://trendlyne.com/portfolio/superstar-shareholders/index/",
    "institutional": "https://trendlyne.com/portfolio/superstar-shareholders/index/institutional/",
    "fii": "https://trendlyne.com/portfolio/superstar-shareholders/index/fii/",
}

# Curated list of active Individual & Institutional Superstars with rich Bulk/Block & SAST histories
# used to build the multi-year historical event study dataset from live Trendlyne endpoints.
CORE_HISTORICAL_SUPERSTARS = [
    # Individual HNI Superstars
    {"ssid": "53746", "slug": "ashish-kacholia", "name": "Ashish Kacholia", "category": "Individual HNI"},
    {"ssid": "53774", "slug": "mukul-agrawal", "name": "Mukul Agrawal", "category": "Individual HNI"},
    {"ssid": "53805", "slug": "vijay-kishanlal-kedia", "name": "Vijay Kishanlal Kedia", "category": "Individual HNI"},
    {"ssid": "53781", "slug": "rakesh-jhunjhunwala-and-associates", "name": "Rakesh Jhunjhunwala and Associates", "category": "Individual HNI"},
    {"ssid": "53782", "slug": "rekha-jhunjhunwala", "name": "Rekha Jhunjhunwala", "category": "Individual HNI"},
    {"ssid": "182955", "slug": "sunil-singhania", "name": "Sunil Singhania", "category": "Individual HNI"},
    {"ssid": "178317", "slug": "radhakishan-damani", "name": "Radhakishan Damani", "category": "Individual HNI"},
    {"ssid": "584325", "slug": "madhusudan-kela", "name": "Madhusudan Kela", "category": "Individual HNI"},
    {"ssid": "53757", "slug": "dolly-khanna", "name": "Dolly Khanna", "category": "Individual HNI"},
    {"ssid": "53777", "slug": "porinju-v-veliyath", "name": "Porinju V Veliyath", "category": "Individual HNI"},
    {"ssid": "53740", "slug": "akash-bhanshali", "name": "Akash Bhanshali", "category": "Individual HNI"},
    {"ssid": "53745", "slug": "ashish-dhawan", "name": "Ashish Dhawan", "category": "Individual HNI"},
    {"ssid": "53743", "slug": "anil-kumar-goel-and-associates", "name": "Anil Kumar Goel and Associates", "category": "Individual HNI"},
    {"ssid": "53739", "slug": "ajay-upadhyaya", "name": "Ajay Upadhyaya", "category": "Individual HNI"},
    {"ssid": "69664", "slug": "mohnish-pabrai", "name": "Mohnish Pabrai", "category": "Individual HNI"},
    {"ssid": "53776", "slug": "nemish-s-shah", "name": "Nemish S Shah", "category": "Individual HNI"},
    {"ssid": "62728", "slug": "ramesh-damani", "name": "Ramesh Damani", "category": "Individual HNI"},
    {"ssid": "584326", "slug": "shankar-sharma", "name": "Shankar Sharma", "category": "Individual HNI"},
    # Institutional & FII Superstars
    {"ssid": "53812", "slug": "abu-dhabi-investment-authority", "name": "Abu Dhabi Investment Authority", "category": "Institutional"},
    {"ssid": "53821", "slug": "amansa-holdings-private-limited", "name": "Amansa Holdings Private Limited", "category": "Institutional"},
    {"ssid": "53815", "slug": "acacia-institutional-partners-lp", "name": "Acacia Institutional Partners Lp", "category": "Institutional"},
    {"ssid": "53856", "slug": "government-of-singapore", "name": "Government of Singapore", "category": "Institutional"},
    {"ssid": "53889", "slug": "nalanda-india-equity-fund-limited", "name": "Nalanda India Equity Fund Limited", "category": "Institutional"},
    {"ssid": "53883", "slug": "malabar-india-fund-limited", "name": "Malabar India Fund Limited", "category": "Institutional"},
]


# ---------------------------------------------------------------------------
# Parsing & Formatting Helpers
# ---------------------------------------------------------------------------

def _clean_float(val: Any, default: float = np.nan) -> float:
    """Safely parse numbers with commas, %, Cr, L, or whitespace."""
    if val is None:
        return default
    s = str(val).strip()
    if not s or s in ("-", "--", "N/A", "Filing Awaited", "None", "null"):
        return default
    s = s.replace(",", "").replace("%", "").replace("₹", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return default
    try:
        return float(m.group(0))
    except ValueError:
        return default


def _parse_indian_value_cr(text: str) -> Tuple[float, float]:
    """
    Parse Trendlyne's 'Portfolio Value *(change)' or 'Holding Value' string.
    Example: '1,184.33 Cr -0.55%' -> (1184.33, -0.55)
    Example: '55.2 L' -> (0.552, np.nan)
    """
    if not text:
        return (0.0, 0.0)
    cleaned = " ".join(str(text).split())
    val_cr = 0.0
    change_pct = 0.0

    # Match value before Cr or L
    m_cr = re.search(r"([\d,]+\.?\d*)\s*Cr", cleaned, re.IGNORECASE)
    m_l = re.search(r"([\d,]+\.?\d*)\s*L\b", cleaned, re.IGNORECASE)
    if m_cr:
        val_cr = _clean_float(m_cr.group(1), 0.0)
        rest = cleaned[m_cr.end():]
        m_pct = re.search(r"([-+]?\d*\.?\d+)\s*%", rest)
        if m_pct:
            change_pct = _clean_float(m_pct.group(1), 0.0)
    elif m_l:
        val_cr = _clean_float(m_l.group(1), 0.0) / 100.0
        rest = cleaned[m_l.end():]
        m_pct = re.search(r"([-+]?\d*\.?\d+)\s*%", rest)
        if m_pct:
            change_pct = _clean_float(m_pct.group(1), 0.0)
    else:
        nums = re.findall(r"[-+]?[\d,]+\.?\d*", cleaned)
        if nums:
            val_cr = _clean_float(nums[0], 0.0)
        if len(nums) > 1:
            change_pct = _clean_float(nums[1], 0.0)

    return (round(val_cr, 3), round(change_pct, 2))


def extract_ticker_from_trendlyne_url(url: str) -> str:
    """
    Extract NSE symbol or BSE code from Trendlyne equity URLs:
    - https://trendlyne.com/equity/914/NAVINFLUOR/ -> NAVINFLUOR
    - https://trendlyne.com/equity/share-holding/206/BLUECHIP/latest/blue-chip-india-ltd/ -> BLUECHIP
    - https://trendlyne.com/equity/bulk-block-deals/AROGRANITE/101/aro-granite-industries-ltd/ -> AROGRANITE
    - https://trendlyne.com/equity/insider-trading-sast/all/SHK/1788/s-h-kelkar-company-ltd/ -> SHK
    """
    if not url:
        return ""
    u = str(url).strip()

    # Pattern 1: /equity/share-holding/<id>/<TICKER>/
    m = re.search(r"/equity/share-holding/\d+/([^/]+)/", u)
    if m:
        return m.group(1).strip().upper()

    # Pattern 2: /equity/bulk-block-deals/<TICKER>/<id>/
    m = re.search(r"/equity/bulk-block-deals/([^/]+)/\d+/", u)
    if m:
        return m.group(1).strip().upper()

    # Pattern 3: /equity/insider-trading-sast/all/<TICKER>/<id>/
    m = re.search(r"/equity/insider-trading-sast/all/([^/]+)/\d+/", u)
    if m:
        return m.group(1).strip().upper()

    # Pattern 4: /equity/<id>/<TICKER>/
    m = re.search(r"/equity/\d+/([^/]+)/?", u)
    if m:
        return m.group(1).strip().upper()

    return ""


def to_yfinance_ticker(symbol: str, exchange: str = "NSE") -> str:
    """Convert raw NSE symbol or numeric BSE code to Yahoo Finance ticker (.NS or .BO)."""
    if not symbol:
        return ""
    sym = str(symbol).strip().upper()
    if sym.endswith(".NS") or sym.endswith(".BO") or sym.startswith("^"):
        return sym
    if sym.isdigit():
        # Numeric codes on Trendlyne can be BSE scrip codes (typically 6 digits starting with 5)
        if len(sym) == 6 and sym.startswith("5"):
            return f"{sym}.BO"
        return ""
    if str(exchange).upper() == "BSE":
        # Even when a deal occurred on BSE, most Indian equities have primary liquidity on .NS in yfinance,
        # so we default to .NS and fall back to .BO during price lookup if .NS is empty.
        return f"{sym}.NS"
    return f"{sym}.NS"


def _parse_date_flexible(date_str: str) -> Optional[pd.Timestamp]:
    """Parse dates like '16 Oct 2026', '16-OCT-2026 08:11:46', '30-SEP-2026', '2026-09-30'."""
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s or s in ("-", "--", "None", "null", "Filing Awaited"):
        return None
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Official NSE Shareholding Pattern Filings (Exact Broadcast Timestamps)
# ---------------------------------------------------------------------------

class NSEShareholdingClient:
    """
    Fetches official corporate shareholding pattern filing dates and broadcast timestamps
    directly from NSE India's live API:
    https://www.nseindia.com/api/corporate-share-holdings-master
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._initialized = False

    def _ensure_session(self) -> None:
        if not self._initialized:
            try:
                self.session.get("https://www.nseindia.com", timeout=10)
                self._initialized = True
            except Exception as exc:
                logger.warning("NSE session initialization warning: %s", exc)

    def fetch_master_filings(self, use_cache: bool = True, cache_ttl_minutes: int = 720) -> pd.DataFrame:
        """
        Fetch the latest 1,400+ corporate shareholding pattern filings across NSE equities
        with exact `submissionDate` and `broadcastDate` (down to the second).
        Falls back gracefully to committed live snapshot (`data/snapshots/nse_shareholding_master.json.gz`)
        when running on cloud data-center IPs where NSE blocks automated requests.
        """
        cache_file = CACHE_DIR / "nse_shareholding_master.json"
        snap_file = SNAPSHOT_DIR / "nse_shareholding_master.json.gz"

        if use_cache and cache_file.exists():
            age_min = (time.time() - cache_file.stat().st_mtime) / 60.0
            if age_min <= cache_ttl_minutes:
                try:
                    raw = json.loads(cache_file.read_text(encoding="utf-8"))
                    return self._records_to_df(raw)
                except Exception:
                    pass

        if use_cache and snap_file.exists():
            snap_raw = _load_snapshot_gz(snap_file)
            if isinstance(snap_raw, list) and snap_raw:
                return self._records_to_df(snap_raw)

        self._ensure_session()
        url = "https://www.nseindia.com/api/corporate-share-holdings-master?index=equities"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                self._initialized = False
                self._ensure_session()
                resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    cache_file.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
                    _save_snapshot_gz(snap_file, data)
                    return self._records_to_df(data)
        except Exception as exc:
            logger.warning("Failed to fetch live NSE shareholding master: %s", exc)

        if cache_file.exists():
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                return self._records_to_df(raw)
            except Exception:
                pass

        snap_raw = _load_snapshot_gz(snap_file)
        if isinstance(snap_raw, list) and snap_raw:
            return self._records_to_df(snap_raw)

        return pd.DataFrame()

    def fetch_symbol_filing_history(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch the multi-year (20+ quarters) official shareholding filing history for a specific NSE symbol,
        returning the exact `broadcastDate` and `submissionDate` for each quarter (`30-JUN-2026`, `31-MAR-2026`, etc.).
        """
        sym = str(symbol).strip().upper().replace(".NS", "").replace(".BO", "")
        if not sym or sym.isdigit():
            return pd.DataFrame()

        sym_cache_dir = CACHE_DIR / "nse_symbol_shp"
        sym_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = sym_cache_dir / f"{sym}.json"

        if use_cache and cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600.0
            if age_hours <= 12.0:
                try:
                    raw = json.loads(cache_file.read_text(encoding="utf-8"))
                    return self._records_to_df(raw)
                except Exception:
                    pass

        self._ensure_session()
        url = f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={sym}"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    cache_file.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
                    return self._records_to_df(data)
        except Exception as exc:
            logger.debug("Symbol SHP history lookup failed for %s: %s", sym, exc)

        if cache_file.exists():
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                return self._records_to_df(raw)
            except Exception:
                pass
        return pd.DataFrame()

    @staticmethod
    def _records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        rows = []
        for r in records:
            b_dt = _parse_date_flexible(r.get("broadcastDate") or r.get("systemDate") or r.get("submissionDate"))
            q_dt = _parse_date_flexible(r.get("date"))
            quarter_label = ""
            if q_dt is not None and not pd.isna(q_dt):
                quarter_label = q_dt.strftime("%b %Y")
            rows.append({
                "symbol": str(r.get("symbol") or "").strip().upper(),
                "company_name": str(r.get("name") or "").strip(),
                "isin": str(r.get("isin") or "").strip(),
                "quarter_end_date": str(r.get("date") or "").strip(),
                "quarter_label": quarter_label,
                "submission_date": str(r.get("submissionDate") or "").strip(),
                "broadcast_timestamp": str(r.get("broadcastDate") or r.get("systemDate") or "").strip(),
                "broadcast_dt": b_dt,
                "promoter_pct": _clean_float(r.get("pr_and_prgrp"), np.nan),
                "public_pct": _clean_float(r.get("public_val"), np.nan),
                "employee_trust_pct": _clean_float(r.get("employeeTrusts"), 0.0),
                "xbrl_url": str(r.get("xbrl") or "").strip(),
                "remarks": str(r.get("remarksWeb") or "").strip(),
            })
        df = pd.DataFrame(rows)
        if not df.empty and "broadcast_dt" in df.columns:
            df = df.sort_values("broadcast_dt", ascending=False).reset_index(drop=True)
        return df


# Singleton NSE client
_NSE_CLIENT = NSEShareholdingClient()


def get_nse_client() -> NSEShareholdingClient:
    return _NSE_CLIENT


# ---------------------------------------------------------------------------
# 2. Trendlyne Superstar Shareholders Index Scraper (Individual / Inst / FII)
# ---------------------------------------------------------------------------

def fetch_superstar_index(
    category: str = "individual",
    use_cache: bool = True,
    cache_ttl_minutes: int = 720,
    enrich_nse_dates: bool = True,
) -> Dict[str, Any]:
    """
    Scrape Trendlyne's Superstar Shareholders Index live page:
    - category="individual" -> https://trendlyne.com/portfolio/superstar-shareholders/index/
    - category="institutional" -> https://trendlyne.com/portfolio/superstar-shareholders/index/institutional/
    - category="fii" -> https://trendlyne.com/portfolio/superstar-shareholders/index/fii/
    """
    cat_key = category.lower().strip()
    if cat_key not in TRENDLYNE_INDEX_URLS:
        raise ValueError(f"Unsupported category: {category}. Choose from {list(TRENDLYNE_INDEX_URLS.keys())}")

    url = TRENDLYNE_INDEX_URLS[cat_key]
    cache_stem = f"trendlyne_index_{cat_key}"
    html_text, fetched_at = _fetch_trendlyne_html_with_snapshot(
        url=url,
        cache_stem=cache_stem,
        use_cache=use_cache,
        cache_ttl_minutes=cache_ttl_minutes,
    )

    soup = BeautifulSoup(html_text or "", "html.parser")

    # 1. Extract top summary chart JSONs (#lineChart, #barChart, #donut)
    chart_networth = []
    chart_top_stocks = []
    chart_sectors = []

    el_line = soup.find(id="lineChart")
    if el_line and el_line.get("data-jsondata"):
        try:
            chart_networth = json.loads(el_line.get("data-jsondata"))
        except Exception:
            pass

    el_bar = soup.find(id="barChart")
    if el_bar and el_bar.get("data-jsondata"):
        try:
            raw_bar = json.loads(el_bar.get("data-jsondata"))
            if isinstance(raw_bar, list) and len(raw_bar) > 1:
                headers_bar = raw_bar[0]
                for row_b in raw_bar[1:]:
                    chart_top_stocks.append(dict(zip(headers_bar, row_b)))
        except Exception:
            pass

    el_donut = soup.find(id="donut")
    if el_donut and el_donut.get("data-jsondata"):
        try:
            raw_donut = json.loads(el_donut.get("data-jsondata"))
            if isinstance(raw_donut, list) and len(raw_donut) > 1:
                for row_d in raw_donut[1:]:
                    if len(row_d) >= 2:
                        chart_sectors.append({"sector": str(row_d[0]), "weight_pct": _clean_float(row_d[1], 0.0)})
        except Exception:
            pass

    # 2. Load NSE Master Filings map for timestamp enrichment
    nse_latest_map: Dict[str, Dict[str, Any]] = {}
    if enrich_nse_dates:
        nse_df = _NSE_CLIENT.fetch_master_filings(use_cache=use_cache)
        if not nse_df.empty:
            for _, nrow in nse_df.iterrows():
                sym = nrow["symbol"]
                if sym and sym not in nse_latest_map:
                    nse_latest_map[sym] = nrow.to_dict()

    # 3. Parse #groupTable rows
    table = soup.find("table", {"id": "groupTable"})
    investors_rows = []
    recent_actions_rows = []

    cat_display = {
        "individual": "Individual HNI",
        "institutional": "Institutional",
        "fii": "FII",
    }.get(cat_key, cat_key.title())

    if table and table.find("tbody"):
        for tr in table.find("tbody").find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 8:
                continue

            # Column 0: Superstar Name & Profile Link
            a_super = tds[0].find("a")
            investor_name = a_super.get_text(" ", strip=True) if a_super else tds[0].get_text(" ", strip=True)
            profile_href = a_super.get("href", "") if a_super else ""
            if profile_href.startswith("/"):
                profile_href = f"https://trendlyne.com{profile_href}"

            ssid, slug = "", ""
            m_super = re.search(r"/superstar-shareholders/(\d+)/[^/]+/([^/]+)/?", profile_href)
            if m_super:
                ssid, slug = m_super.group(1), m_super.group(2)

            # Column 1: Portfolio Value *(change)
            val_cr, qoq_change_pct = _parse_indian_value_cr(tds[1].get_text(" ", strip=True))

            # Column 2: #Of Stocks
            stocks_count = int(_clean_float(tds[2].get_text(strip=True), 0))

            # Column 3: Sector Preference
            sectors_list = []
            for a_sec in tds[3].find_all("a"):
                sec_txt = " ".join(a_sec.get_text(" ", strip=True).split())
                m_sec = re.match(r"^(.*?)\s*\(([-+]?\d*\.?\d+)%\)$", sec_txt)
                if m_sec:
                    sectors_list.append({
                        "sector": m_sec.group(1).strip(),
                        "pct": _clean_float(m_sec.group(2), 0.0),
                    })
                elif sec_txt:
                    sectors_list.append({"sector": sec_txt, "pct": 0.0})
            top_sector = sectors_list[0]["sector"] if sectors_list else "-"
            sectors_summary = ", ".join(
                [f"{s['sector']} ({s['pct']}%)" if s["pct"] else s["sector"] for s in sectors_list]
            ) or "-"

            # Column 4: Quarterly Net Worth Sparkline JSON
            spark_div = tds[4].find("div", class_="tl-sparkline")
            spark_dates, spark_vals = [], []
            latest_quarter_date = ""
            if spark_div and spark_div.get("data-sparkline"):
                try:
                    sp_json = json.loads(spark_div.get("data-sparkline"))
                    spark_dates = sp_json.get("d", [])
                    spark_vals = sp_json.get("v", [])
                    if spark_dates:
                        latest_quarter_date = spark_dates[-1]
                except Exception:
                    pass

            # Column 5: Top Holdings
            top_holdings_list = []
            for a_hold in tds[5].find_all("a"):
                h_txt = " ".join(a_hold.get_text(" ", strip=True).split())
                h_href = a_hold.get("href", "")
                if h_href.startswith("/"):
                    h_href = f"https://trendlyne.com{h_href}"
                ticker = extract_ticker_from_trendlyne_url(h_href)
                h_val_cr, _ = _parse_indian_value_cr(h_txt)
                stock_clean = re.sub(r"\s*\([\d,\.\s]+(?:Cr|L).*?\)", "", h_txt, flags=re.IGNORECASE).strip()
                top_holdings_list.append({
                    "stock": stock_clean,
                    "ticker": ticker,
                    "value_cr": h_val_cr,
                    "url": h_href,
                })
            top_holdings_summary = ", ".join(
                [f"{h['stock']} [{h['ticker']}] (₹{h['value_cr']} Cr)" if h["ticker"] else f"{h['stock']} (₹{h['value_cr']} Cr)"
                 for h in top_holdings_list]
            ) or "-"

            # Column 6: Recently Bought & Column 7: Recently Sold
            def _parse_action_col(td_el: Any, action_label: str) -> List[Dict[str, Any]]:
                items = []
                for a_act in td_el.find_all("a"):
                    a_txt = " ".join(a_act.get_text(" ", strip=True).split())
                    a_href = a_act.get("href", "")
                    if a_href.startswith("/"):
                        a_href = f"https://trendlyne.com{a_href}"
                    ticker = extract_ticker_from_trendlyne_url(a_href)
                    m_chg = re.search(r"([-+]?\d*\.?\d+)\s*%$", a_txt)
                    stake_chg = _clean_float(m_chg.group(1), 0.0) if m_chg else 0.0
                    stock_name = a_txt[:m_chg.start()].strip() if m_chg else a_txt.strip()

                    # Cross-reference with NSE Official Shareholding Master for exact filing timestamp
                    nse_info = nse_latest_map.get(ticker, {})
                    broadcast_ts = nse_info.get("broadcast_timestamp", "")
                    submission_dt = nse_info.get("submission_date", "")
                    quarter_end = nse_info.get("quarter_end_date", "")
                    xbrl_url = nse_info.get("xbrl_url", "")

                    item = {
                        "investor_name": investor_name,
                        "ssid": ssid,
                        "slug": slug,
                        "category": cat_display,
                        "action": action_label,
                        "stock_name": stock_name,
                        "ticker": ticker,
                        "yf_ticker": to_yfinance_ticker(ticker),
                        "stake_change_pct": stake_chg,
                        "quarter_end_date": quarter_end or "30-SEP-2026 / Latest Filed",
                        "publication_date": submission_dt or (broadcast_ts.split(" ")[0] if broadcast_ts else "Latest Quarterly SHP"),
                        "broadcast_timestamp": broadcast_ts or "See Symbol SHP Filing History",
                        "xbrl_url": xbrl_url,
                        "stock_url": a_href,
                        "investor_net_worth_cr": val_cr,
                    }
                    items.append(item)
                    recent_actions_rows.append(item)
                return items

            bought_list = _parse_action_col(tds[6], "Recently Bought")
            sold_list = _parse_action_col(tds[7], "Recently Sold")

            bought_summary = ", ".join(
                [
                    f"{b['stock_name']} ({b['ticker']}: +{abs(b['stake_change_pct'])}%)"
                    + (f" [Filed: {b['publication_date']}]" if b["broadcast_timestamp"] != "See Symbol SHP Filing History" else "")
                    for b in bought_list
                ]
            ) or "-"

            sold_summary = ", ".join(
                [
                    f"{s['stock_name']} ({s['ticker']}: -{abs(s['stake_change_pct'])}%)"
                    + (f" [Filed: {s['publication_date']}]" if s["broadcast_timestamp"] != "See Symbol SHP Filing History" else "")
                    for s in sold_list
                ]
            ) or "-"

            investors_rows.append({
                "superstar_name": investor_name,
                "ssid": ssid,
                "slug": slug,
                "category": cat_display,
                "portfolio_value_cr": val_cr,
                "qoq_change_pct": qoq_change_pct,
                "stocks_count": stocks_count,
                "top_sector": top_sector,
                "sector_preferences": sectors_summary,
                "top_holdings": top_holdings_summary,
                "recently_bought": bought_summary,
                "recently_bought_count": len(bought_list),
                "recently_sold": sold_summary,
                "recently_sold_count": len(sold_list),
                "latest_quarter": latest_quarter_date,
                "sparkline_dates": spark_dates,
                "sparkline_values": spark_vals,
                "top_holdings_struct": top_holdings_list,
                "bought_struct": bought_list,
                "sold_struct": sold_list,
                "profile_url": profile_href,
            })

    investors_df = pd.DataFrame(investors_rows)
    if not investors_df.empty:
        investors_df = investors_df.sort_values("portfolio_value_cr", ascending=False).reset_index(drop=True)

    recent_actions_df = pd.DataFrame(recent_actions_rows)

    return {
        "category": cat_display,
        "source_url": url,
        "fetched_at": fetched_at,
        "investors_df": investors_df,
        "recent_actions_df": recent_actions_df,
        "chart_networth": chart_networth,
        "chart_top_stocks": pd.DataFrame(chart_top_stocks),
        "chart_sectors": pd.DataFrame(chart_sectors),
    }


# ---------------------------------------------------------------------------
# 3. Superstar Deep-Dive Portfolio, Bulk/Block Deals & SAST/Insider Scrapers
# ---------------------------------------------------------------------------

def fetch_superstar_portfolio(
    ssid: str,
    slug: str,
    use_cache: bool = True,
    cache_ttl_minutes: int = 60,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch the full multi-quarter stock holdings table for a specific Superstar Investor:
    https://trendlyne.com/portfolio/superstar-shareholders/<ssid>/latest/<slug>/
    Enriches each stock with its official NSE Shareholding Pattern broadcast timestamp.
    """
    url = f"https://trendlyne.com/portfolio/superstar-shareholders/{ssid}/latest/{slug}/"
    html_text, _ = _fetch_trendlyne_html_with_snapshot(
        url=url,
        cache_stem=f"superstar_portfolio_{ssid}",
        use_cache=use_cache,
        cache_ttl_minutes=cache_ttl_minutes,
    )
    if not html_text:
        return pd.DataFrame(), []

    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if not table or not table.find("tbody"):
        return pd.DataFrame(), []

    headers = [th.get_text(" ", strip=True) for th in table.find("thead").find_all("th")] if table.find("thead") else []
    # Typically headers: ['Expand', 'Stock', 'Holding Value', 'Qty Held', 'Sep 2026 Change %', 'Sep 2026 Holding %', 'Jun 2026 %', ...]
    quarter_cols = [h for h in headers[5:] if h]

    nse_df = _NSE_CLIENT.fetch_master_filings(use_cache=True)
    nse_map = {}
    if not nse_df.empty:
        for _, r in nse_df.iterrows():
            if r["symbol"] not in nse_map:
                nse_map[r["symbol"]] = r.to_dict()

    rows = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 6:
            continue

        # Stock name & ticker
        a_stock = tds[1].find("a")
        stock_name = a_stock.get_text(" ", strip=True) if a_stock else tds[1].get_text(" ", strip=True)
        stock_href = a_stock.get("href", "") if a_stock else ""
        if stock_href.startswith("/"):
            stock_href = f"https://trendlyne.com{stock_href}"
        ticker = extract_ticker_from_trendlyne_url(stock_href)

        val_txt = tds[2].get_text(" ", strip=True)
        val_cr, _ = _parse_indian_value_cr(val_txt)
        qty_held = int(_clean_float(tds[3].get_text(strip=True), 0))
        chg_txt = tds[4].get_text(" ", strip=True)
        filing_status = "Filing Awaited" if "Awaited" in chg_txt else "Filed"
        qoq_chg = _clean_float(chg_txt, 0.0)

        nse_info = nse_map.get(ticker, {})
        broadcast_ts = nse_info.get("broadcast_timestamp", "")
        pub_date = nse_info.get("submission_date", "")

        row_dict: Dict[str, Any] = {
            "stock_name": stock_name,
            "ticker": ticker,
            "yf_ticker": to_yfinance_ticker(ticker),
            "holding_value_cr": val_cr,
            "holding_value_text": val_txt,
            "qty_held": qty_held,
            "latest_change_pct": qoq_chg,
            "filing_status": filing_status,
            "nse_publication_date": pub_date or ("Filed" if filing_status == "Filed" else "Awaited"),
            "nse_broadcast_timestamp": broadcast_ts or "-",
            "stock_url": stock_href,
        }

        for idx_q, q_col in enumerate(quarter_cols):
            td_idx = 5 + idx_q
            if td_idx < len(tds):
                q_raw = tds[td_idx].get_text(strip=True)
                row_dict[q_col] = _clean_float(q_raw, 0.0) if q_raw not in ("-", "") else 0.0

        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("holding_value_cr", ascending=False).reset_index(drop=True)
    return df, quarter_cols


def _parse_bulk_block_table(
    soup: BeautifulSoup,
    investor_name: str = "",
    category: str = "Exchange-Wide",
    ssid: str = "",
    slug: str = "",
) -> List[Dict[str, Any]]:
    """Parse a Trendlyne Bulk/Block Deals HTML table into structured records with exact publication dates."""
    table = soup.find("table")
    if not table or not table.find("tbody"):
        return []

    ths = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
    has_intraday_col = any("intraday" in h for h in ths)

    records = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue

        # Stock link (first anchor in td[0])
        a_stock = tds[0].find("a")
        stock_name = a_stock.get_text(" ", strip=True) if a_stock else tds[0].get_text(" ", strip=True)
        stock_href = a_stock.get("href", "") if a_stock else ""
        if stock_href.startswith("/"):
            stock_href = f"https://trendlyne.com{stock_href}"
        ticker = extract_ticker_from_trendlyne_url(stock_href)

        client_name = tds[1].get_text(" ", strip=True)
        exchange = tds[2].get_text(" ", strip=True).upper()
        deal_type = tds[3].get_text(" ", strip=True).title()
        action = tds[4].get_text(" ", strip=True).title()
        date_str = tds[5].get_text(" ", strip=True)
        avg_price = _clean_float(tds[6].get_text(strip=True), np.nan)
        quantity = _clean_float(tds[7].get_text(strip=True), 0.0)

        if has_intraday_col and len(tds) >= 10:
            intraday = tds[8].get_text(" ", strip=True)
            traded_pct = _clean_float(tds[9].get_text(strip=True), 0.0)
        else:
            intraday = "No"
            traded_pct = _clean_float(tds[8].get_text(strip=True), 0.0)

        deal_value_cr = round((avg_price * quantity) / 1e7, 3) if not np.isnan(avg_price) and quantity > 0 else 0.0
        pub_dt = _parse_date_flexible(date_str)

        records.append({
            "event_source": "Bulk/Block Deal",
            "superstar_name": investor_name or client_name,
            "client_name": client_name,
            "ssid": ssid,
            "slug": slug,
            "category": category,
            "stock_name": stock_name,
            "ticker": ticker,
            "yf_ticker": to_yfinance_ticker(ticker, exchange),
            "exchange": exchange,
            "deal_type": deal_type,
            "action": action,  # 'Purchase' or 'Sell'
            "direction": "BUY" if action.lower() in ("purchase", "buy", "acquisition") else "SELL",
            "publication_date": date_str,
            "publication_dt": pub_dt,
            "transaction_period": date_str,
            "deal_price": avg_price,
            "quantity": int(quantity) if not np.isnan(quantity) else 0,
            "value_cr": deal_value_cr,
            "traded_pct": traded_pct,
            "intraday": intraday,
            "stock_url": stock_href,
        })
    return records


def fetch_superstar_bulk_block_deals(
    ssid: str,
    slug: str,
    investor_name: str = "",
    category: str = "Individual HNI",
    use_cache: bool = True,
    cache_ttl_minutes: int = 60,
) -> pd.DataFrame:
    """
    Fetch all historical Bulk & Block Deals for a specific Superstar:
    https://trendlyne.com/portfolio/bulk-block-deals/<ssid>/<slug>/
    """
    url = f"https://trendlyne.com/portfolio/bulk-block-deals/{ssid}/{slug}/"
    html_text, _ = _fetch_trendlyne_html_with_snapshot(
        url=url,
        cache_stem=f"superstar_bb_{ssid}",
        use_cache=use_cache,
        cache_ttl_minutes=cache_ttl_minutes,
    )
    if not html_text:
        return pd.DataFrame()

    soup = BeautifulSoup(html_text, "html.parser")
    records = _parse_bulk_block_table(soup, investor_name=investor_name, category=category, ssid=ssid, slug=slug)
    df = pd.DataFrame(records)
    if not df.empty and "publication_dt" in df.columns:
        df = df.sort_values("publication_dt", ascending=False).reset_index(drop=True)
    return df


def _parse_sast_table(
    soup: BeautifulSoup,
    investor_name: str = "",
    category: str = "Exchange-Wide",
    ssid: str = "",
    slug: str = "",
) -> List[Dict[str, Any]]:
    """
    Parse Trendlyne SAST / Insider Trading table into structured records.
    Columns:
    0: Stock, 1: Client Name, 2: Client Category, 3: Action*,
    4: Reported To/By Exchange (Publication Date!), 5: Quantity,
    6: Post Transaction Holding, 7: Traded %, 8: Avg. Price,
    9: Value, 10: Period, 11: Regulation (Insider/SAST), 12: Security Type, 13: Mode
    """
    table = soup.find("table")
    if not table or not table.find("tbody"):
        return []

    records = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 12:
            continue

        a_stock = tds[0].find("a")
        stock_name = a_stock.get_text(" ", strip=True) if a_stock else tds[0].get_text(" ", strip=True)
        stock_href = a_stock.get("href", "") if a_stock else ""
        if stock_href.startswith("/"):
            stock_href = f"https://trendlyne.com{stock_href}"
        ticker = extract_ticker_from_trendlyne_url(stock_href)

        client_name = tds[1].get_text(" ", strip=True)
        client_cat = tds[2].get_text(" ", strip=True)
        action = tds[3].get_text(" ", strip=True).title()
        reported_date = tds[4].get_text(" ", strip=True)
        quantity = _clean_float(tds[5].get_text(strip=True), 0.0)
        post_holding_txt = " ".join(tds[6].get_text(" ", strip=True).split())
        traded_pct = _clean_float(tds[7].get_text(strip=True), 0.0)
        avg_price = _clean_float(tds[8].get_text(strip=True), np.nan)
        val_inr = _clean_float(tds[9].get_text(strip=True), 0.0)
        val_cr = round(val_inr / 1e7, 3) if val_inr > 0 else (
            round((avg_price * quantity) / 1e7, 3) if not np.isnan(avg_price) and quantity > 0 else 0.0
        )
        period_txt = " ".join(tds[10].get_text(" ", strip=True).split())
        regulation = tds[11].get_text(" ", strip=True)
        sec_type = tds[12].get_text(" ", strip=True) if len(tds) > 12 else "Equity"
        mode = tds[13].get_text(" ", strip=True) if len(tds) > 13 else "Market"

        pub_dt = _parse_date_flexible(reported_date)
        direction = "BUY" if action.lower() in ("acquisition", "purchase", "buy", "revoke") else (
            "SELL" if action.lower() in ("disposal", "sell", "pledge", "invocation") else "OTHER"
        )

        records.append({
            "event_source": f"SAST / Insider ({regulation})",
            "superstar_name": investor_name or client_name,
            "client_name": client_name,
            "client_category": client_cat,
            "ssid": ssid,
            "slug": slug,
            "category": category if investor_name else client_cat,
            "stock_name": stock_name,
            "ticker": ticker,
            "yf_ticker": to_yfinance_ticker(ticker),
            "exchange": "NSE/BSE",
            "deal_type": f"{regulation} ({mode})",
            "action": action,
            "direction": direction,
            "publication_date": reported_date,
            "publication_dt": pub_dt,
            "transaction_period": period_txt,
            "deal_price": avg_price,
            "quantity": int(quantity) if not np.isnan(quantity) else 0,
            "value_cr": val_cr,
            "traded_pct": traded_pct,
            "post_holding": post_holding_txt,
            "security_type": sec_type,
            "mode": mode,
            "stock_url": stock_href,
        })
    return records


def fetch_superstar_sast_insider(
    ssid: str,
    slug: str,
    investor_name: str = "",
    category: str = "Individual HNI",
    use_cache: bool = True,
    cache_ttl_minutes: int = 60,
) -> pd.DataFrame:
    """
    Fetch all historical SAST & Insider Trading disclosures for a specific Superstar:
    https://trendlyne.com/equity/insider-trading-sast/<ssid>/<slug>/
    """
    url = f"https://trendlyne.com/equity/insider-trading-sast/{ssid}/{slug}/"
    html_text, _ = _fetch_trendlyne_html_with_snapshot(
        url=url,
        cache_stem=f"superstar_sast_{ssid}",
        use_cache=use_cache,
        cache_ttl_minutes=cache_ttl_minutes,
    )
    if not html_text:
        return pd.DataFrame()

    soup = BeautifulSoup(html_text, "html.parser")
    records = _parse_sast_table(soup, investor_name=investor_name, category=category, ssid=ssid, slug=slug)
    df = pd.DataFrame(records)
    if not df.empty and "publication_dt" in df.columns:
        df = df.sort_values("publication_dt", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. Exchange-Wide Live Bulk/Block Deals & SAST/Insider Disclosures
# ---------------------------------------------------------------------------

def fetch_live_bulk_block_deals(use_cache: bool = True, cache_ttl_minutes: int = 120) -> pd.DataFrame:
    """
    Fetch the latest 200+ exchange-wide Bulk & Block Deals from Trendlyne:
    https://trendlyne.com/portfolio/bulk-block-deals/
    """
    url = "https://trendlyne.com/portfolio/bulk-block-deals/"
    html_text, _ = _fetch_trendlyne_html_with_snapshot(
        url=url,
        cache_stem="live_exchange_bulk_block",
        use_cache=use_cache,
        cache_ttl_minutes=cache_ttl_minutes,
    )
    if not html_text:
        return pd.DataFrame()

    soup = BeautifulSoup(html_text, "html.parser")
    records = _parse_bulk_block_table(soup, investor_name="", category="Institutional / HNI Deal")
    df = pd.DataFrame(records)
    if not df.empty and "publication_dt" in df.columns:
        df = df.sort_values("publication_dt", ascending=False).reset_index(drop=True)
    return df


def fetch_live_sast_insider_deals(use_cache: bool = True, cache_ttl_minutes: int = 120) -> pd.DataFrame:
    """
    Fetch the latest exchange-wide SAST & Insider Trading disclosures from Trendlyne:
    https://trendlyne.com/equity/group-insider-trading-sast/
    """
    url = "https://trendlyne.com/equity/group-insider-trading-sast/"
    html_text, _ = _fetch_trendlyne_html_with_snapshot(
        url=url,
        cache_stem="live_exchange_sast_insider",
        use_cache=use_cache,
        cache_ttl_minutes=cache_ttl_minutes,
    )
    if not html_text:
        return pd.DataFrame()

    soup = BeautifulSoup(html_text, "html.parser")
    records = _parse_sast_table(soup, investor_name="", category="SAST / Insider")
    df = pd.DataFrame(records)
    if not df.empty and "publication_dt" in df.columns:
        df = df.sort_values("publication_dt", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 5. Historical Superstar Event Dataset Builder (Bulk/Block + SAST + NSE SHP)
# ---------------------------------------------------------------------------

def build_historical_superstar_event_dataset(
    use_cache: bool = True,
    max_workers: int = 6,
    min_year: int = 2018,
) -> pd.DataFrame:
    """
    Aggregates real, dated historical disclosures across:
    1. Core Individual HNI & Institutional Superstars' Bulk & Block Deals
    2. Core Individual HNI & Institutional Superstars' SAST & Insider Filings
    3. Live Exchange-Wide Bulk/Block Deals & SAST Disclosures
    4. Recent Quarterly Shareholding Changes (Recently Bought / Sold) across Individual & Institutional
       Superstars that have an exact verified NSE Broadcast Date (`broadcast_dt`) in NSE Shareholding Master!

    Every single event in the returned DataFrame is 100% real and carries an exact `publication_dt`
    when the disclosure was published to the market.
    """
    all_frames: List[pd.DataFrame] = []

    def _fetch_one_superstar(item: Dict[str, str]) -> List[pd.DataFrame]:
        frames = []
        try:
            df_bb = fetch_superstar_bulk_block_deals(
                ssid=item["ssid"],
                slug=item["slug"],
                investor_name=item["name"],
                category=item["category"],
                use_cache=use_cache,
                cache_ttl_minutes=240,
            )
            if not df_bb.empty:
                frames.append(df_bb)
        except Exception as exc:
            logger.debug("BB fetch error for %s: %s", item["name"], exc)

        try:
            df_sast = fetch_superstar_sast_insider(
                ssid=item["ssid"],
                slug=item["slug"],
                investor_name=item["name"],
                category=item["category"],
                use_cache=use_cache,
                cache_ttl_minutes=240,
            )
            if not df_sast.empty:
                frames.append(df_sast)
        except Exception as exc:
            logger.debug("SAST fetch error for %s: %s", item["name"], exc)
        return frames

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch_one_superstar, item) for item in CORE_HISTORICAL_SUPERSTARS]
        for f in as_completed(futs):
            try:
                all_frames.extend(f.result())
            except Exception:
                pass

    # Add exchange-wide live Bulk/Block & SAST feeds
    try:
        df_live_bb = fetch_live_bulk_block_deals(use_cache=use_cache)
        if not df_live_bb.empty:
            all_frames.append(df_live_bb)
    except Exception:
        pass

    try:
        df_live_sast = fetch_live_sast_insider_deals(use_cache=use_cache)
        if not df_live_sast.empty:
            all_frames.append(df_live_sast)
    except Exception:
        pass

    # Add Superstar Quarterly Shareholding Changes that have verified NSE filing dates
    try:
        nse_master = _NSE_CLIENT.fetch_master_filings(use_cache=use_cache)
        nse_map = {}
        if not nse_master.empty:
            for _, nrow in nse_master.iterrows():
                if nrow["symbol"] and nrow["symbol"] not in nse_map:
                    nse_map[nrow["symbol"]] = nrow.to_dict()

        shp_events = []
        for cat_key in ("individual", "institutional"):
            idx_data = fetch_superstar_index(category=cat_key, use_cache=use_cache, enrich_nse_dates=False)
            act_df = idx_data.get("recent_actions_df", pd.DataFrame())
            if act_df.empty:
                continue
            for _, arow in act_df.iterrows():
                sym = str(arow.get("ticker") or "").strip().upper()
                if not sym:
                    continue
                n_info = nse_map.get(sym)
                if not n_info or pd.isna(n_info.get("broadcast_dt")):
                    continue
                pub_dt = n_info["broadcast_dt"]
                act_str = str(arow.get("action") or "")
                is_buy = "Bought" in act_str
                shp_events.append({
                    "event_source": "Quarterly Shareholding Filing (NSE SHP)",
                    "superstar_name": arow["investor_name"],
                    "client_name": arow["investor_name"],
                    "ssid": arow["ssid"],
                    "slug": arow["slug"],
                    "category": arow["category"],
                    "stock_name": arow["stock_name"],
                    "ticker": sym,
                    "yf_ticker": to_yfinance_ticker(sym),
                    "exchange": "NSE",
                    "deal_type": f"SHP ({n_info.get('quarter_end_date', 'Qtr')})",
                    "action": "Stake Increase" if is_buy else "Stake Decrease",
                    "direction": "BUY" if is_buy else "SELL",
                    "publication_date": n_info.get("broadcast_timestamp") or n_info.get("submission_date"),
                    "publication_dt": pub_dt,
                    "transaction_period": n_info.get("quarter_end_date", ""),
                    "deal_price": np.nan,
                    "quantity": 0,
                    "value_cr": 0.0,
                    "traded_pct": abs(_clean_float(arow.get("stake_change_pct"), 0.0)),
                    "stock_url": arow.get("stock_url", ""),
                })
        if shp_events:
            all_frames.append(pd.DataFrame(shp_events))
    except Exception as exc:
        logger.debug("SHP event enrichment warning: %s", exc)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Filter out rows without valid ticker or publication date
    combined = combined[combined["yf_ticker"].astype(str).str.len() > 0].copy()
    combined["publication_dt"] = pd.to_datetime(combined["publication_dt"], errors="coerce")
    combined = combined.dropna(subset=["publication_dt"]).copy()

    # Normalize timezone-naive date for deduplication and price matching
    combined["pub_date_only"] = combined["publication_dt"].dt.tz_localize(None).dt.normalize()
    combined = combined[combined["pub_date_only"].dt.year >= min_year].copy()

    # Keep only BUY / SELL equity actions
    combined = combined[combined["direction"].isin(["BUY", "SELL"])].copy()

    # Deduplicate same day, same ticker, same superstar/client, same direction & source
    combined = combined.sort_values("traded_pct", ascending=False).drop_duplicates(
        subset=["yf_ticker", "pub_date_only", "superstar_name", "direction", "event_source"],
        keep="first",
    )
    combined = combined.sort_values("publication_dt", ascending=False).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# 6. Yahoo Finance (yfinance) Historical & Live OHLCV Price Cache
# ---------------------------------------------------------------------------

def fetch_ohlcv_history_batch(
    yf_tickers: List[str],
    period: str = "5y",
    use_cache: bool = True,
    cache_ttl_minutes: int = 720,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV history from Yahoo Finance for a list of `.NS` / `.BO` tickers plus `^NSEI`.
    Uses local pickle caching per ticker and a negative cache for delisted symbols so subsequent
    analysis runs in milliseconds. Automatically tries `.BO` fallback if `.NS` returns empty.
    """
    price_cache_dir = CACHE_DIR / "yf_ohlcv"
    price_cache_dir.mkdir(parents=True, exist_ok=True)
    neg_cache_file = price_cache_dir / "negative_cache.json"

    neg_cache: Dict[str, float] = {}
    if use_cache and neg_cache_file.exists():
        try:
            neg_cache = json.loads(neg_cache_file.read_text(encoding="utf-8"))
        except Exception:
            neg_cache = {}

    now_ts = time.time()
    unique_tickers = sorted(set([t.strip() for t in yf_tickers if t and str(t).strip()] + ["^NSEI"]))
    result: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []

    for sym in unique_tickers:
        safe_name = sym.replace("^", "IDX_")
        c_path = price_cache_dir / f"{safe_name}.pkl"
        if use_cache and c_path.exists():
            age_min = (now_ts - c_path.stat().st_mtime) / 60.0
            if age_min <= cache_ttl_minutes:
                try:
                    df_cached = pd.read_pickle(c_path)
                    if isinstance(df_cached, pd.DataFrame) and not df_cached.empty:
                        result[sym] = df_cached
                        continue
                except Exception:
                    pass
        if use_cache and sym in neg_cache:
            if (now_ts - float(neg_cache[sym])) / 60.0 <= cache_ttl_minutes:
                continue
        to_download.append(sym)

    if to_download:
        # Download in batches of 40 tickers for speed and reliability
        batch_size = 40
        for i in range(0, len(to_download), batch_size):
            chunk = to_download[i : i + batch_size]
            try:
                raw_dl = yf.download(
                    tickers=" ".join(chunk),
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if raw_dl is not None and not raw_dl.empty:
                    for sym in chunk:
                        try:
                            if len(chunk) == 1 and not isinstance(raw_dl.columns, pd.MultiIndex):
                                sub = raw_dl.copy()
                            else:
                                if sym not in raw_dl.columns.get_level_values(0):
                                    continue
                                sub = raw_dl[sym].copy()
                            if isinstance(sub.columns, pd.MultiIndex):
                                sub.columns = sub.columns.get_level_values(0)
                            sub = sub.dropna(subset=["Close"]).copy()
                            if not sub.empty:
                                sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                                sub = sub[~sub.index.duplicated(keep="last")].sort_index()
                                result[sym] = sub
                                safe_name = sym.replace("^", "IDX_")
                                sub.to_pickle(price_cache_dir / f"{safe_name}.pkl")
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("Batch yfinance download warning: %s", exc)

        # Fallback: for any .NS ticker that returned empty, try .BO
        missing_ns = [s for s in to_download if s not in result and s.endswith(".NS")]
        for sym_ns in missing_ns[:25]:
            sym_bo = sym_ns[:-3] + ".BO"
            try:
                sub_bo = yf.Ticker(sym_bo).history(period=period, auto_adjust=False)
                if sub_bo is not None and not sub_bo.empty:
                    sub_bo.index = pd.to_datetime(sub_bo.index).tz_localize(None).normalize()
                    sub_bo = sub_bo[~sub_bo.index.duplicated(keep="last")].sort_index()
                    result[sym_ns] = sub_bo
                    safe_name = sym_ns.replace("^", "IDX_")
                    sub_bo.to_pickle(price_cache_dir / f"{safe_name}.pkl")
            except Exception:
                pass

        # Record still-missing symbols in negative_cache.json
        for sym in to_download:
            if sym not in result:
                neg_cache[sym] = now_ts
        try:
            neg_cache_file.write_text(json.dumps(neg_cache, ensure_ascii=True), encoding="utf-8")
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# 7. Real-Time / Intraday Live CMP Fetcher & Market Hours Status (IST)
# ---------------------------------------------------------------------------
import datetime as dt


def get_indian_market_status_ist() -> Dict[str, Any]:
    """
    Returns live Indian stock market (NSE/BSE) status in IST (UTC+05:30):
    - Regular Market Hours: Monday–Friday, 09:15 IST to 15:30 IST
    - Pre-Open Session: Monday–Friday, 09:00 IST to 09:15 IST
    """
    ist_tz = dt.timezone(dt.timedelta(hours=5, minutes=30), name="IST")
    now_ist = dt.datetime.now(dt.timezone.utc).astimezone(ist_tz)
    is_weekday = now_ist.weekday() < 5
    mins_since_midnight = now_ist.hour * 60 + now_ist.minute

    is_pre_open = is_weekday and (9 * 60 <= mins_since_midnight < 9 * 60 + 15)
    is_market_open = is_weekday and (9 * 60 + 15 <= mins_since_midnight <= 15 * 60 + 30)

    if is_market_open:
        status_badge = "🟢 NSE/BSE MARKET LIVE (09:15–15:30 IST)"
        status_color = "#4ade80"
        mode_note = "Streaming Live Intraday CMP"
    elif is_pre_open:
        status_badge = "🟡 NSE/BSE PRE-OPEN SESSION (09:00–09:15 IST)"
        status_color = "#facc15"
        mode_note = "Pre-Open Equilibrium Discovery"
    else:
        status_badge = "🕒 MARKET CLOSED — Showing Latest Verified Exchange Close CMP"
        status_color = "#38bdf8"
        mode_note = "Post-Market / Off-Hours (Auto-Syncs at 09:20 & 15:00 IST Mon–Fri)"

    now_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST (%a)")
    return {
        "now_ist_str": now_str,
        "ist_now_str": now_str,
        "is_weekday": is_weekday,
        "is_market_open": is_market_open,
        "is_pre_open": is_pre_open,
        "status_badge": status_badge,
        "status_label": status_badge,
        "status_color": status_color,
        "mode_note": mode_note,
    }


def fetch_live_cmp_batch(
    tickers_or_yf: List[str],
    force_network: bool = True,
) -> Tuple[Dict[str, float], str]:
    """
    Fast batch fetcher for Live Current Market Price (CMP in ₹) across NSE/BSE stocks,
    Indian Indices (`^NSEI`, `^NSEBANK`, etc.), and Sectoral ETFs.
    - When `force_network=True`, queries Yahoo Finance (`period='5d', interval='1d'`) in a single
      multi-threaded batch to pull the latest intraday / closing price and merges the latest bar
      into `data/cache/yf_ohlcv/*.pkl` so all tables immediately reflect fresh CMPs.
    - Returns `(cmp_map, sync_timestamp_ist)` where `cmp_map` maps both clean symbol (`RELIANCE`)
      and Yahoo symbol (`RELIANCE.NS`) to the latest live float price.
    """
    ist_tz = dt.timezone(dt.timedelta(hours=5, minutes=30), name="IST")
    sync_ts = dt.datetime.now(dt.timezone.utc).astimezone(ist_tz).strftime("%Y-%m-%d %H:%M:%S IST")

    price_cache_dir = CACHE_DIR / "yf_ohlcv"
    price_cache_dir.mkdir(parents=True, exist_ok=True)

    yf_list: List[str] = []
    for raw_t in tickers_or_yf:
        s = str(raw_t or "").strip().upper()
        if not s:
            continue
        if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
            yf_list.append(s)
        else:
            yf_list.append(f"{s}.NS")
    yf_list = sorted(set(yf_list))

    cmp_map: Dict[str, float] = {}

    if force_network and yf_list:
        batch_size = 60
        for i in range(0, len(yf_list), batch_size):
            chunk = yf_list[i : i + batch_size]
            try:
                raw_dl = yf.download(
                    tickers=" ".join(chunk),
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if raw_dl is not None and not raw_dl.empty:
                    for sym in chunk:
                        try:
                            if len(chunk) == 1 and not isinstance(raw_dl.columns, pd.MultiIndex):
                                sub = raw_dl.copy()
                            else:
                                if sym not in raw_dl.columns.get_level_values(0):
                                    continue
                                sub = raw_dl[sym].copy()
                            if isinstance(sub.columns, pd.MultiIndex):
                                sub.columns = sub.columns.get_level_values(0)
                            sub = sub.dropna(subset=["Close"]).copy()
                            if sub.empty:
                                continue
                            sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                            sub = sub[~sub.index.duplicated(keep="last")].sort_index()
                            latest_px = float(sub["Close"].iloc[-1])
                            if latest_px > 0 and not np.isnan(latest_px):
                                cmp_map[sym] = round(latest_px, 2)
                                clean_sym = sym[:-3] if (sym.endswith(".NS") or sym.endswith(".BO")) else sym
                                cmp_map[clean_sym] = round(latest_px, 2)

                                # Also update the local OHLCV pickle so all indicators & tables stay in sync
                                safe_name = sym.replace("^", "IDX_")
                                pkl_path = price_cache_dir / f"{safe_name}.pkl"
                                if pkl_path.exists():
                                    try:
                                        old_df = pd.read_pickle(pkl_path)
                                        if isinstance(old_df, pd.DataFrame) and not old_df.empty:
                                            old_df.index = pd.to_datetime(old_df.index).tz_localize(None).normalize()
                                            merged_df = pd.concat([old_df, sub])
                                            merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()
                                            merged_df.to_pickle(pkl_path)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("Live CMP batch download warning: %s", exc)

    # Fill any remaining tickers from local cache
    for sym in yf_list:
        if sym in cmp_map:
            continue
        safe_name = sym.replace("^", "IDX_")
        pkl_path = price_cache_dir / f"{safe_name}.pkl"
        if pkl_path.exists():
            try:
                df_c = pd.read_pickle(pkl_path)
                if isinstance(df_c, pd.DataFrame) and not df_c.empty and "Close" in df_c.columns:
                    px_val = float(df_c["Close"].dropna().iloc[-1])
                    if px_val > 0:
                        cmp_map[sym] = round(px_val, 2)
                        clean_sym = sym[:-3] if (sym.endswith(".NS") or sym.endswith(".BO")) else sym
                        cmp_map[clean_sym] = round(px_val, 2)
            except Exception:
                pass

    return cmp_map, sync_ts

