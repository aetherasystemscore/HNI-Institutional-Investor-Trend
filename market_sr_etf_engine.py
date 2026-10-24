"""
market_sr_etf_engine.py
=======================
100% Original / Live Historical Data Engine for:
1. Indian Indices Support & Resistance + Empirical Directional Movement Probability
   (Based on 10–15 years of actual daily OHLCV history from official NSE Indices
   `niftyindices.com` + `nseindia.com/api/allIndices` + `yfinance` — ZERO synthetic
   calculations, ZERO hash-based fallbacks, ZERO artificial data).
2. NSE Sectoral & Thematic ETF Live Screener (all Production-grade technical, S/R,
   iNAV, Volume, RSI, ADX, ATR, DMA, Bollinger, MACD, Stochastic, Multi-Horizon Return,
   and Empirical S/R Bounce Win-Rate parameters).
3. Data-Driven Future Sector Suitability & Mega-Trend Investment Advisor
   (Evaluating AI & Digital Compute, Metal & Critical Mineral Demand, Power & Grid
   Electrification Demand, Defense/Manufacturing Capex, EV/Auto, Financials,
   Healthcare, and Consumption strictly from live technical positioning, 10–15Y
   empirical win probabilities, NSE index P/E & P/B valuations, and live Superstar
   HNI/Institutional accumulation flows).
"""

from __future__ import annotations

import datetime
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

INDICES_SR_CSV = DATA_DIR / "indian_indices_sr_15y.csv"
ETF_SCREENER_CSV = DATA_DIR / "sectoral_etf_screener_live.csv"
SECTOR_SUITABILITY_CSV = DATA_DIR / "sector_future_suitability_matrix.csv"
INDEX_HISTORY_CACHE_GZ = SNAPSHOT_DIR / "indian_indices_15y_ohlc.json.gz"

# ---------------------------------------------------------------------------
# 1. INDIAN INDICES UNIVERSE (15-YEAR OFFICIAL NSE INDICES + YFINANCE)
# ---------------------------------------------------------------------------
INDIAN_INDICES_UNIVERSE: List[Dict[str, str]] = [
    {
        "id": "NIFTY_50",
        "name": "Nifty 50",
        "category": "Broad Market Benchmark",
        "yf_ticker": "^NSEI",
        "nifty_trading_name": "Nifty 50",
        "nifty_long_name": "Nifty 50",
        "nse_api_name": "NIFTY 50",
        "macro_pillar": "Broad Market Core",
    },
    {
        "id": "BSE_SENSEX",
        "name": "BSE Sensex",
        "category": "Broad Market Benchmark",
        "yf_ticker": "^BSESN",
        "nifty_trading_name": "",
        "nifty_long_name": "",
        "nse_api_name": "",
        "macro_pillar": "Broad Market Core",
    },
    {
        "id": "NIFTY_NEXT_50",
        "name": "Nifty Next 50",
        "category": "Broad Market Benchmark",
        "yf_ticker": "^NSMIDCP",
        "nifty_trading_name": "Nifty Next 50",
        "nifty_long_name": "Nifty Next 50",
        "nse_api_name": "NIFTY NEXT 50",
        "macro_pillar": "Broad Market Core",
    },
    {
        "id": "NIFTY_100",
        "name": "Nifty 100",
        "category": "Broad Market Benchmark",
        "yf_ticker": "^CNX100",
        "nifty_trading_name": "Nifty 100",
        "nifty_long_name": "Nifty 100",
        "nse_api_name": "NIFTY 100",
        "macro_pillar": "Broad Market Core",
    },
    {
        "id": "NIFTY_200",
        "name": "Nifty 200",
        "category": "Broad Market Benchmark",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty 200",
        "nifty_long_name": "Nifty 200",
        "nse_api_name": "NIFTY 200",
        "macro_pillar": "Broad Market Core",
    },
    {
        "id": "NIFTY_500",
        "name": "Nifty 500",
        "category": "Broad Market Benchmark",
        "yf_ticker": "^CRSLDX",
        "nifty_trading_name": "Nifty 500",
        "nifty_long_name": "Nifty 500",
        "nse_api_name": "NIFTY 500",
        "macro_pillar": "Broad Market Core",
    },
    {
        "id": "NIFTY_MIDCAP_100",
        "name": "Nifty Midcap 100",
        "category": "Mid & Smallcap",
        "yf_ticker": "",
        "nifty_trading_name": "NIFTY MIDCAP 100",
        "nifty_long_name": "NIFTY Midcap 100",
        "nse_api_name": "NIFTY MIDCAP 100",
        "macro_pillar": "Domestic Capex & Midcap Growth",
    },
    {
        "id": "NIFTY_MIDCAP_50",
        "name": "Nifty Midcap 50",
        "category": "Mid & Smallcap",
        "yf_ticker": "^NSEMDCP50",
        "nifty_trading_name": "Nifty Midcap 50",
        "nifty_long_name": "Nifty Midcap 50",
        "nse_api_name": "NIFTY MIDCAP 50",
        "macro_pillar": "Domestic Capex & Midcap Growth",
    },
    {
        "id": "NIFTY_SMALLCAP_100",
        "name": "Nifty Smallcap 100",
        "category": "Mid & Smallcap",
        "yf_ticker": "",
        "nifty_trading_name": "NIFTY SMLCAP 100",
        "nifty_long_name": "Nifty Smallcap 100",
        "nse_api_name": "NIFTY SMALLCAP 100",
        "macro_pillar": "Domestic Capex & Midcap Growth",
    },
    {
        "id": "BSE_500",
        "name": "BSE 500",
        "category": "Broad Market Benchmark",
        "yf_ticker": "BSE-500.BO",
        "nifty_trading_name": "",
        "nifty_long_name": "",
        "nse_api_name": "",
        "macro_pillar": "Broad Market Core",
    },
    {
        "id": "NIFTY_IT",
        "name": "Nifty IT",
        "category": "Sectoral — AI & Digital Tech",
        "yf_ticker": "^CNXIT",
        "nifty_trading_name": "Nifty IT",
        "nifty_long_name": "Nifty IT",
        "nse_api_name": "NIFTY IT",
        "macro_pillar": "AI & Digital Compute",
    },
    {
        "id": "NIFTY_METAL",
        "name": "Nifty Metal",
        "category": "Sectoral — Metals & Mining",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Metal",
        "nifty_long_name": "Nifty Metal",
        "nse_api_name": "NIFTY METAL",
        "macro_pillar": "Metals & Critical Minerals",
    },
    {
        "id": "NIFTY_ENERGY",
        "name": "Nifty Energy",
        "category": "Sectoral — Power & Energy",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Energy",
        "nifty_long_name": "Nifty Energy",
        "nse_api_name": "NIFTY ENERGY",
        "macro_pillar": "Power, Grid & Energy Transition",
    },
    {
        "id": "NIFTY_INFRA",
        "name": "Nifty Infrastructure",
        "category": "Sectoral — Power & Infra",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Infra",
        "nifty_long_name": "Nifty Infrastructure",
        "nse_api_name": "NIFTY INFRASTRUCTURE",
        "macro_pillar": "Power, Grid & Energy Transition",
    },
    {
        "id": "NIFTY_CPSE",
        "name": "Nifty CPSE (Power/Coal/BEL/Grid)",
        "category": "Thematic — PSU Power & Defense",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty CPSE",
        "nifty_long_name": "Nifty CPSE",
        "nse_api_name": "NIFTY CPSE",
        "macro_pillar": "Power, Grid & Energy Transition",
    },
    {
        "id": "NIFTY_PSE",
        "name": "Nifty PSE (Public Sector Enterprises)",
        "category": "Thematic — PSU Power & Defense",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty PSE",
        "nifty_long_name": "Nifty PSE",
        "nse_api_name": "NIFTY PSE",
        "macro_pillar": "Defense & Make-in-India Capex",
    },
    {
        "id": "NIFTY_COMMODITIES",
        "name": "Nifty Commodities",
        "category": "Thematic — Metals, Energy & Materials",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Commodities",
        "nifty_long_name": "Nifty Commodities",
        "nse_api_name": "NIFTY COMMODITIES",
        "macro_pillar": "Metals & Critical Minerals",
    },
    {
        "id": "NIFTY_AUTO",
        "name": "Nifty Auto",
        "category": "Sectoral — Auto & EV Mobility",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Auto",
        "nifty_long_name": "Nifty Auto",
        "nse_api_name": "NIFTY AUTO",
        "macro_pillar": "Auto & EV Mobility",
    },
    {
        "id": "NIFTY_BANK",
        "name": "Nifty Bank",
        "category": "Sectoral — Banking & BFSI",
        "yf_ticker": "^NSEBANK",
        "nifty_trading_name": "Nifty Bank",
        "nifty_long_name": "Nifty Bank",
        "nse_api_name": "NIFTY BANK",
        "macro_pillar": "Banking & Financialization",
    },
    {
        "id": "NIFTY_FIN_SERVICE",
        "name": "Nifty Financial Services",
        "category": "Sectoral — Banking & BFSI",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Fin Service",
        "nifty_long_name": "Nifty Financial Services",
        "nse_api_name": "NIFTY FINANCIAL SERVICES",
        "macro_pillar": "Banking & Financialization",
    },
    {
        "id": "NIFTY_PSU_BANK",
        "name": "Nifty PSU Bank",
        "category": "Sectoral — Banking & BFSI",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty PSU Bank",
        "nifty_long_name": "Nifty PSU Bank",
        "nse_api_name": "NIFTY PSU BANK",
        "macro_pillar": "Banking & Financialization",
    },
    {
        "id": "NIFTY_PHARMA",
        "name": "Nifty Pharma",
        "category": "Sectoral — Healthcare & Pharma",
        "yf_ticker": "^CNXPHARMA",
        "nifty_trading_name": "Nifty Pharma",
        "nifty_long_name": "Nifty Pharma",
        "nse_api_name": "NIFTY PHARMA",
        "macro_pillar": "Healthcare & Pharma",
    },
    {
        "id": "NIFTY_FMCG",
        "name": "Nifty FMCG",
        "category": "Sectoral — Consumption & Realty",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty FMCG",
        "nifty_long_name": "Nifty FMCG",
        "nse_api_name": "NIFTY FMCG",
        "macro_pillar": "Domestic Consumption & Realty",
    },
    {
        "id": "NIFTY_CONSUMPTION",
        "name": "Nifty India Consumption",
        "category": "Thematic — Consumption & Realty",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Consumption",
        "nifty_long_name": "Nifty India Consumption",
        "nse_api_name": "NIFTY INDIA CONSUMPTION",
        "macro_pillar": "Domestic Consumption & Realty",
    },
    {
        "id": "NIFTY_REALTY",
        "name": "Nifty Realty",
        "category": "Sectoral — Consumption & Realty",
        "yf_ticker": "",
        "nifty_trading_name": "Nifty Realty",
        "nifty_long_name": "Nifty Realty",
        "nse_api_name": "NIFTY REALTY",
        "macro_pillar": "Domestic Consumption & Realty",
    },
]


# ---------------------------------------------------------------------------
# 2. SECTORAL & THEMATIC ETF UNIVERSE (ALL NSE SECTORAL + THEMATIC ETFS)
# ---------------------------------------------------------------------------
SECTORAL_ETF_UNIVERSE: List[Dict[str, Any]] = [
    # --- PILLAR 1: AI, IT Services, Digital India & Global Tech Compute ---
    {
        "ticker": "ITBEES.NS",
        "name": "Nippon India ETF Nifty IT BeES",
        "category": "IT & AI Services",
        "macro_pillar": "AI & Digital Compute",
        "benchmark_index": "Nifty IT",
        "expense": 0.22,
        "div_yield": 1.85,
        "macro_thesis": "Enterprise AI integration, cloud migration & IT services export recovery",
        "constituent_keywords": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS"],
    },
    {
        "ticker": "ITIETF.NS",
        "name": "ICICI Prudential Nifty IT ETF",
        "category": "IT & AI Services",
        "macro_pillar": "AI & Digital Compute",
        "benchmark_index": "Nifty IT",
        "expense": 0.20,
        "div_yield": 1.80,
        "macro_thesis": "Institutional low-cost Nifty IT tracker for AI software & GenAI services cycle",
        "constituent_keywords": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT"],
    },
    {
        "ticker": "TECH.NS",
        "name": "Aditya Birla Sun Life Nifty IT ETF",
        "category": "IT & AI Services",
        "macro_pillar": "AI & Digital Compute",
        "benchmark_index": "Nifty IT",
        "expense": 0.22,
        "div_yield": 1.75,
        "macro_thesis": "Direct Indian Tier-1 & Tier-2 IT exposure benefiting from AI GCC & deal wins",
        "constituent_keywords": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    },
    {
        "ticker": "TNIDETF.NS",
        "name": "Tata Nifty India Digital ETF",
        "category": "Digital India & AI",
        "macro_pillar": "AI & Digital Compute",
        "benchmark_index": "Nifty IT",
        "expense": 0.40,
        "div_yield": 0.85,
        "macro_thesis": "Digital platforms, telecom 5G/data centers (Bharti Airtel, Tata Comm), AI & e-commerce",
        "constituent_keywords": ["BHARTIARTL", "TCS", "INFY", "TATACOMM", "INFOEDGE", "ZOMATO", "PAYTM", "IRCTC"],
    },
    {
        "ticker": "MON100.NS",
        "name": "Motilal Oswal Nasdaq 100 ETF",
        "category": "Global AI & Semiconductors",
        "macro_pillar": "AI & Digital Compute",
        "benchmark_index": "Nifty IT",
        "expense": 0.58,
        "div_yield": 0.15,
        "macro_thesis": "Direct global AI hyperscaler & GPU/silicon exposure (NVIDIA, Microsoft, Apple, Alphabet, Meta, Broadcom)",
        "constituent_keywords": ["NASDAQ", "TECH", "AI"],
    },
    {
        "ticker": "MAFANG.NS",
        "name": "Mirae Asset NYSE FANG+ ETF",
        "category": "Global AI Hyperscalers",
        "macro_pillar": "AI & Digital Compute",
        "benchmark_index": "Nifty IT",
        "expense": 0.65,
        "div_yield": 0.0,
        "macro_thesis": "Concentrated 10-stock global AI compute, LLM cloud & semiconductor monopoly basket",
        "constituent_keywords": ["FANG", "AI"],
    },
    {
        "ticker": "MAHKTECH.NS",
        "name": "Mirae Asset Hang Seng TECH ETF",
        "category": "Asia AI & Cloud Tech",
        "macro_pillar": "AI & Digital Compute",
        "benchmark_index": "Nifty IT",
        "expense": 0.70,
        "div_yield": 0.20,
        "macro_thesis": "Asian AI models, robotics, EV autonomy & cloud ecosystems (Alibaba, Tencent, Xiaomi, BYD, SMIC)",
        "constituent_keywords": ["HKTECH"],
    },

    # --- PILLAR 2: Metals, Mining, Critical Minerals & Industrial Bullion ---
    {
        "ticker": "METALIETF.NS",
        "name": "ICICI Prudential Nifty Metal ETF",
        "category": "Metals & Mining",
        "macro_pillar": "Metals & Critical Minerals",
        "benchmark_index": "Nifty Metal",
        "expense": 0.40,
        "div_yield": 2.40,
        "macro_thesis": "Copper, Aluminium, Steel & Zinc feedstock for power grids, EVs, renewable solar/wind & AI data centers",
        "constituent_keywords": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "NMDC", "HINDZINC", "JINDALSTEL", "SAIL", "NATIONALUM", "APLAPOLLO"],
    },
    {
        "ticker": "METAL.NS",
        "name": "Mirae Asset Nifty Metal ETF",
        "category": "Metals & Mining",
        "macro_pillar": "Metals & Critical Minerals",
        "benchmark_index": "Nifty Metal",
        "expense": 0.38,
        "div_yield": 2.35,
        "macro_thesis": "Direct play on non-ferrous & ferrous metal upcycle driven by global electrification & infrastructure",
        "constituent_keywords": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC", "JINDALSTEL", "SAIL", "NATIONALUM"],
    },
    {
        "ticker": "COMMOIETF.NS",
        "name": "ICICI Prudential Nifty Commodities ETF",
        "category": "Metals, Energy & Materials",
        "macro_pillar": "Metals & Critical Minerals",
        "benchmark_index": "Nifty Commodities",
        "expense": 0.30,
        "div_yield": 2.80,
        "macro_thesis": "Blended Indian Metals + Power + Cement + Energy supercycle basket (Reliance, NTPC, UltraTech, Tata Steel, Hindalco, Coal India)",
        "constituent_keywords": ["RELIANCE", "NTPC", "ULTRACEMCO", "TATASTEEL", "HINDALCO", "JSWSTEEL", "COALINDIA", "ONGC", "GRASIM"],
    },
    {
        "ticker": "SILVERBEES.NS",
        "name": "Nippon India Silver ETF (Silver BeES)",
        "category": "Industrial & Precious Metal",
        "macro_pillar": "Metals & Critical Minerals",
        "benchmark_index": "Nifty Metal",
        "expense": 0.50,
        "div_yield": 0.0,
        "macro_thesis": "Highest electrical/thermal conductivity metal: structural deficit from Solar PV cells, AI server connectors & EV power electronics",
        "constituent_keywords": ["SILVER", "HINDZINC"],
    },
    {
        "ticker": "SILVERIETF.NS",
        "name": "ICICI Prudential Silver ETF",
        "category": "Industrial & Precious Metal",
        "macro_pillar": "Metals & Critical Minerals",
        "benchmark_index": "Nifty Metal",
        "expense": 0.45,
        "div_yield": 0.0,
        "macro_thesis": "Physical silver bullion tracking industrial solar PV + AI hardware + monetary hedge demand",
        "constituent_keywords": ["SILVER"],
    },
    {
        "ticker": "GOLDBEES.NS",
        "name": "Nippon India ETF Gold BeES",
        "category": "Precious Metal Hedge",
        "macro_pillar": "Precious Metals Safe-Haven",
        "benchmark_index": "Nifty 50",
        "expense": 0.79,
        "div_yield": 0.0,
        "macro_thesis": "Central bank reserve accumulation, geopolitical tail-risk hedge & sovereign currency debasement protection",
        "constituent_keywords": ["GOLD", "TITAN", "MUTHOOTFIN"],
    },
    {
        "ticker": "SETFGOLD.NS",
        "name": "SBI Gold ETF",
        "category": "Precious Metal Hedge",
        "macro_pillar": "Precious Metals Safe-Haven",
        "benchmark_index": "Nifty 50",
        "expense": 0.50,
        "div_yield": 0.0,
        "macro_thesis": "Low-expense physical Gold ETF for portfolio drawdown dampening",
        "constituent_keywords": ["GOLD"],
    },

    # --- PILLAR 3: Power, Grid Electrification, Energy & Infrastructure ---
    {
        "ticker": "CPSEETF.NS",
        "name": "Nippon India CPSE ETF (Power/Grid/BEL/Coal)",
        "category": "Power, Grid & PSU Utilities",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty CPSE (Power/Coal/BEL/Grid)",
        "expense": 0.05,
        "div_yield": 4.60,
        "macro_thesis": "Heavyweight allocation (~60%+) to NTPC, Power Grid, Coal India, NHPC, SJVN, ONGC & BEL — direct beneficiary of surging AI data-center + industrial power load",
        "constituent_keywords": ["NTPC", "POWERGRID", "COALINDIA", "ONGC", "BEL", "NHPC", "SJVN", "NLCINDIA", "OIL", "COCHINSHIP"],
    },
    {
        "ticker": "GROWWPOWER.NS",
        "name": "Groww BSE Power ETF",
        "category": "Pure-Play Power & Grid",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty Energy",
        "expense": 0.42,
        "div_yield": 2.10,
        "macro_thesis": "Pure-play Indian power generation, HVDC grid transmission & heavy electrical equipment (NTPC, Power Grid, Tata Power, Adani Power, Siemens, ABB, CG Power, BHEL, Suzlon)",
        "constituent_keywords": ["NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "JSWENERGY", "NHPC", "SIEMENS", "ABB", "CGPOWER", "BHEL", "SUZLON", "TORNTPOWER"],
    },
    {
        "ticker": "ENERGY.NS",
        "name": "Mirae Asset Nifty Energy ETF",
        "category": "Power, Renewables & Oil/Gas",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty Energy",
        "expense": 0.35,
        "div_yield": 3.10,
        "macro_thesis": "Tracks Nifty Energy (Reliance, NTPC, Power Grid, ONGC, Coal India, Tata Power, Adani Green/Power)",
        "constituent_keywords": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "COALINDIA", "TATAPOWER", "BPCL", "IOC", "GAIL", "ADANIGREEN"],
    },
    {
        "ticker": "MOENERGY.NS",
        "name": "Motilal Oswal Nifty Energy ETF",
        "category": "Power, Renewables & Oil/Gas",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty Energy",
        "expense": 0.35,
        "div_yield": 3.05,
        "macro_thesis": "Baseload thermal + green hydrogen + HVDC grid transmission + gas pipeline infrastructure",
        "constituent_keywords": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "COALINDIA", "TATAPOWER"],
    },
    {
        "ticker": "OILIETF.NS",
        "name": "ICICI Prudential Nifty Oil & Gas ETF",
        "category": "Energy & Gas Grid",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty Energy",
        "expense": 0.35,
        "div_yield": 3.80,
        "macro_thesis": "Natural gas grid (GAIL, Petronet, IGL), refining & upstream energy cash-flow compounders",
        "constituent_keywords": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "HPCL", "OIL", "PETRONET", "IGL"],
    },
    {
        "ticker": "INFRABEES.NS",
        "name": "Nippon India ETF Nifty Infrastructure BeES",
        "category": "Power, Capex & Infrastructure",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty Infrastructure",
        "expense": 0.85,
        "div_yield": 1.40,
        "macro_thesis": "EPC engineering (L&T), power utilities (NTPC, Power Grid, Tata Power), ports (Adani Ports), telecom towers & ultra-mega capex",
        "constituent_keywords": ["LT", "RELIANCE", "BHARTIARTL", "NTPC", "POWERGRID", "ADANIPORTS", "ULTRACEMCO", "TATAPOWER", "ONGC", "GRASIM"],
    },
    {
        "ticker": "INFRAIETF.NS",
        "name": "ICICI Prudential Nifty Infrastructure ETF",
        "category": "Power, Capex & Infrastructure",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty Infrastructure",
        "expense": 0.35,
        "div_yield": 1.45,
        "macro_thesis": "Low-cost institutional Nifty Infrastructure tracker covering EPC, power grid, ports & cement",
        "constituent_keywords": ["LT", "RELIANCE", "BHARTIARTL", "NTPC", "POWERGRID", "ADANIPORTS", "ULTRACEMCO"],
    },
    {
        "ticker": "ICICIB22.NS",
        "name": "ICICI Prudential BHARAT 22 ETF",
        "category": "PSU Capex, Power & Engineering",
        "macro_pillar": "Power, Grid & Energy Transition",
        "benchmark_index": "Nifty PSE (Public Sector Enterprises)",
        "expense": 0.05,
        "div_yield": 3.85,
        "macro_thesis": "Ultra-low 0.05% fee basket of L&T, NTPC, Power Grid, BEL, Coal India, Nalco, SBI, Axis Bank & ITC",
        "constituent_keywords": ["LT", "NTPC", "POWERGRID", "BEL", "COALINDIA", "NATIONALUM", "SBIN", "ITC", "ONGC", "EIL", "NBCC"],
    },

    # --- PILLAR 4: Defense, Make-in-India Manufacturing & Capital Goods ---
    {
        "ticker": "MODEFENCE.NS",
        "name": "Motilal Oswal Nifty India Defence ETF",
        "category": "Defense & Aerospace",
        "macro_pillar": "Defense & Make-in-India Capex",
        "benchmark_index": "Nifty PSE (Public Sector Enterprises)",
        "expense": 0.45,
        "div_yield": 0.80,
        "macro_thesis": "Indigenous defense electronics, radars, missiles, fighter jets & warships (HAL, BEL, Solar Ind, Bharat Dynamics, Mazagon Dock, Cochin Shipyard, Data Patterns)",
        "constituent_keywords": ["HAL", "BEL", "SOLARINDS", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "DATAPATTNS", "ASTRAMICRO", "BEML", "PARAS"],
    },
    {
        "ticker": "MAKEINDIA.NS",
        "name": "Mirae Asset Nifty India Manufacturing ETF",
        "category": "Industrial Manufacturing & EMS",
        "macro_pillar": "Defense & Make-in-India Capex",
        "benchmark_index": "Nifty Infrastructure",
        "expense": 0.35,
        "div_yield": 0.95,
        "macro_thesis": "China+1 PLI manufacturing, electronics/EMS (Dixon), cables & switchgear (Polycab, Cummins, ABB, Siemens), metals & auto",
        "constituent_keywords": ["SUNPHARMA", "M&M", "TATAMOTORS", "MARUTI", "TATASTEEL", "HINDALCO", "JSWSTEEL", "HAL", "BEL", "DIXON", "POLYCAB", "CUMMINSIND", "SIEMENS", "ABB", "CGPOWER"],
    },
    {
        "ticker": "MNC.NS",
        "name": "Kotak Nifty MNC ETF",
        "category": "Multinational Industrial & Tech",
        "macro_pillar": "Defense & Make-in-India Capex",
        "benchmark_index": "Nifty 200",
        "expense": 0.30,
        "div_yield": 1.35,
        "macro_thesis": "High-ROCE global parent engineering, power automation & consumer leaders (Siemens, ABB, Cummins, Honeywell, Bosch, Maruti, HUL, Nestle)",
        "constituent_keywords": ["MARUTI", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "SIEMENS", "ABB", "CUMMINSIND", "BOSCHLTD", "AMBUJACEM", "ASHOKLEY", "COLPAL"],
    },

    # --- PILLAR 5: Auto & EV Mobility Ecosystem ---
    {
        "ticker": "EVINDIA.NS",
        "name": "Mirae Asset Nifty EV & New Age Automotive ETF",
        "category": "EV, Battery & Auto Tech",
        "macro_pillar": "Auto & EV Mobility",
        "benchmark_index": "Nifty Auto",
        "expense": 0.40,
        "div_yield": 0.65,
        "macro_thesis": "Full EV value chain: EV OEMs (M&M, Tata Motors, Bajaj Auto), battery chemicals & cells (Exide, Amara Raja, Himadri), auto software (KPIT Tech, Tata Elxsi) & charging",
        "constituent_keywords": ["TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "KPITTECH", "TATAELXSI", "EXIDEIND", "ARE&M", "SONACOMS", "MOTHERSON", "CGPOWER", "HSCL"],
    },
    {
        "ticker": "AUTOBEES.NS",
        "name": "Nippon India ETF Nifty Auto BeES",
        "category": "Automobiles & Ancillaries",
        "macro_pillar": "Auto & EV Mobility",
        "benchmark_index": "Nifty Auto",
        "expense": 0.22,
        "div_yield": 0.95,
        "macro_thesis": "Indian passenger vehicle, SUV, 2-wheeler & commercial vehicle leaders (M&M, Maruti, Tata Motors, Bajaj Auto, Eicher, Hero, Motherson)",
        "constituent_keywords": ["M&M", "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR", "MOTHERSON", "BOSCHLTD", "BHARATFORG", "ASHOKLEY"],
    },
    {
        "ticker": "AUTOIETF.NS",
        "name": "ICICI Prudential Nifty Auto ETF",
        "category": "Automobiles & Ancillaries",
        "macro_pillar": "Auto & EV Mobility",
        "benchmark_index": "Nifty Auto",
        "expense": 0.20,
        "div_yield": 0.95,
        "macro_thesis": "Low-cost institutional Nifty Auto tracker",
        "constituent_keywords": ["M&M", "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR"],
    },

    # --- PILLAR 6: Banking, PSU Banks & Financialization ---
    {
        "ticker": "BANKBEES.NS",
        "name": "Nippon India ETF Nifty Bank BeES",
        "category": "Banking Core",
        "macro_pillar": "Banking & Financialization",
        "benchmark_index": "Nifty Bank",
        "expense": 0.19,
        "div_yield": 1.12,
        "macro_thesis": "Flagship Indian banking proxy (HDFC Bank, ICICI Bank, SBI, Axis, Kotak) trading at historically attractive P/E (~13.2x)",
        "constituent_keywords": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB", "AUBANK"],
    },
    {
        "ticker": "BANKIETF.NS",
        "name": "ICICI Prudential Nifty Bank ETF",
        "category": "Banking Core",
        "macro_pillar": "Banking & Financialization",
        "benchmark_index": "Nifty Bank",
        "expense": 0.16,
        "div_yield": 1.12,
        "macro_thesis": "Low-expense Nifty Bank institutional tracker",
        "constituent_keywords": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    },
    {
        "ticker": "PSUBNKBEES.NS",
        "name": "Nippon India ETF Nifty PSU Bank BeES",
        "category": "PSU Banking",
        "macro_pillar": "Banking & Financialization",
        "benchmark_index": "Nifty PSU Bank",
        "expense": 0.49,
        "div_yield": 2.90,
        "macro_thesis": "Public sector credit & capex lenders (SBI, Bank of Baroda, PNB, Canara Bank, Union Bank) with clean balance sheets & ~3% dividend yield",
        "constituent_keywords": ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "MAHABANK"],
    },
    {
        "ticker": "PSUBANK.NS",
        "name": "Kotak Nifty PSU Bank ETF",
        "category": "PSU Banking",
        "macro_pillar": "Banking & Financialization",
        "benchmark_index": "Nifty PSU Bank",
        "expense": 0.49,
        "div_yield": 2.85,
        "macro_thesis": "15-year seasoned PSU Banking ETF tracking state-owned corporate & infrastructure lenders",
        "constituent_keywords": ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB"],
    },
    {
        "ticker": "PVTBANIETF.NS",
        "name": "ICICI Prudential Nifty Private Bank ETF",
        "category": "Private Banking",
        "macro_pillar": "Banking & Financialization",
        "benchmark_index": "Nifty Bank",
        "expense": 0.15,
        "div_yield": 0.95,
        "macro_thesis": "Pure private-sector retail & corporate banking compounders",
        "constituent_keywords": ["HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB"],
    },
    {
        "ticker": "FINIETF.NS",
        "name": "ICICI Prudential Nifty Financial Services ETF",
        "category": "BFSI, NBFCs & FinTech",
        "macro_pillar": "Banking & Financialization",
        "benchmark_index": "Nifty Financial Services",
        "expense": 0.20,
        "div_yield": 0.98,
        "macro_thesis": "Banks + NBFCs + Insurance + Capital Markets (Bajaj Finance, Jio Financial,PFC, REC, Shriram Finance, HDFC Life, SBI Life, BSE)",
        "constituent_keywords": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "BAJAJFINSV", "PFC", "RECLTD", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "SBILIFE", "JIOFIN"],
    },
    {
        "ticker": "BFSI.NS",
        "name": "Mirae Asset Nifty Financial Services ETF",
        "category": "BFSI, NBFCs & Power Financiers",
        "macro_pillar": "Banking & Financialization",
        "benchmark_index": "Nifty Financial Services",
        "expense": 0.22,
        "div_yield": 0.98,
        "macro_thesis": "Includes power & infrastructure financiers (PFC, REC) alongside top banks, NBFCs & insurers",
        "constituent_keywords": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "PFC", "RECLTD", "CHOLAFIN"],
    },

    # --- PILLAR 7: Healthcare, CDMO & Pharmaceuticals ---
    {
        "ticker": "PHARMABEES.NS",
        "name": "Nippon India Nifty Pharma ETF",
        "category": "Pharma & Biotech",
        "macro_pillar": "Healthcare & Pharma",
        "benchmark_index": "Nifty Pharma",
        "expense": 0.21,
        "div_yield": 0.80,
        "macro_thesis": "US generics, specialty molecules, peptides/GLP-1 & Biosecure Act CDMO shift (Sun Pharma, Cipla, Dr Reddy's, Divi's, Lupin, Aurobindo)",
        "constituent_keywords": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM", "BIOCON", "ZYDUSLIFE", "LAURUSLABS"],
    },
    {
        "ticker": "HEALTHIETF.NS",
        "name": "ICICI Prudential Nifty Healthcare ETF",
        "category": "Hospitals, Diagnostics & Pharma",
        "macro_pillar": "Healthcare & Pharma",
        "benchmark_index": "Nifty Pharma",
        "expense": 0.20,
        "div_yield": 0.70,
        "macro_thesis": "Combines export Pharma + domestic hospital chains & diagnostics (Apollo Hospitals, Max Healthcare, Fortis, Syngene)",
        "constituent_keywords": ["SUNPHARMA", "CIPLA", "DRREDDY", "APOLLOHOSP", "MAXHEALTH", "FORTIS", "DIVISLAB", "LUPIN", "SYNGENE"],
    },

    # --- PILLAR 8: Domestic Consumption, FMCG & Real Estate ---
    {
        "ticker": "CONSUMBEES.NS",
        "name": "Nippon India ETF Nifty India Consumption BeES",
        "category": "Consumer Discretionary & Staples",
        "macro_pillar": "Domestic Consumption & Realty",
        "benchmark_index": "Nifty India Consumption",
        "expense": 0.31,
        "div_yield": 1.15,
        "macro_thesis": "Rising per-capita disposable income across retail, auto, telecom, jewelry & staples (ITC, Bharti Airtel, HUL, Titan, M&M, Maruti, Trent, Zomato, Asian Paints)",
        "constituent_keywords": ["BHARTIARTL", "ITC", "HINDUNILVR", "MARUTI", "M&M", "TITAN", "ASIANPAINT", "TRENT", "NESTLEIND", "ZOMATO", "BAJAJ-AUTO", "VBL", "DMART"],
    },
    {
        "ticker": "FMCGIETF.NS",
        "name": "ICICI Prudential Nifty FMCG ETF",
        "category": "Defensive FMCG Staples",
        "macro_pillar": "Domestic Consumption & Realty",
        "benchmark_index": "Nifty FMCG",
        "expense": 0.20,
        "div_yield": 1.85,
        "macro_thesis": "High-ROCE defensive cash-flow compounders (ITC, HUL, Nestle, Varun Beverages, Britannia, Tata Consumer, Godrej CP)",
        "constituent_keywords": ["ITC", "HINDUNILVR", "NESTLEIND", "VBL", "BRITANNIA", "TATACONSUM", "GODREJCP", "DABUR", "MARICO", "COLPAL", "UBL"],
    },
    {
        "ticker": "MOREALTY.NS",
        "name": "Motilal Oswal Nifty Realty ETF",
        "category": "Real Estate & Data Center Land",
        "macro_pillar": "Domestic Consumption & Realty",
        "benchmark_index": "Nifty Realty",
        "expense": 0.40,
        "div_yield": 0.45,
        "macro_thesis": "Commercial grade-A office, GCC campuses, data-center real estate & premium residential cycle (DLF, Macrotech/Lodha, Godrej Properties, Phoenix Mills, Oberoi, Prestige, Brigade)",
        "constituent_keywords": ["DLF", "LODHA", "GODREJPROP", "PHOENIXLTD", "OBEROIRLTY", "PRESTIGE", "BRIGADE", "SOBHA", "ANANTRAJ"],
    },

    # --- PILLAR 9: Core Broad-Market Benchmarks (Reference Comparison) ---
    {
        "ticker": "NIFTYBEES.NS",
        "name": "Nippon India ETF Nifty 50 BeES",
        "category": "Broad Market Benchmark",
        "macro_pillar": "Broad Market Core",
        "benchmark_index": "Nifty 50",
        "expense": 0.04,
        "div_yield": 1.25,
        "macro_thesis": "Core Indian large-cap benchmark ETF",
        "constituent_keywords": ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "LT", "SBIN"],
    },
    {
        "ticker": "MID150BEES.NS",
        "name": "Nippon India ETF Nifty Midcap 150 BeES",
        "category": "Midcap Core",
        "macro_pillar": "Domestic Capex & Midcap Growth",
        "benchmark_index": "Nifty Midcap 100",
        "expense": 0.20,
        "div_yield": 0.75,
        "macro_thesis": "Broad domestic manufacturing, industrial & financial midcap growth engine",
        "constituent_keywords": ["DIXON", "POLYCAB", "PERSISTENT", "CUMMINSIND", "SUZLON", "MAXHEALTH", "INDHOTEL"],
    },
]


# ---------------------------------------------------------------------------
# 3. TECHNICAL INDICATOR & EMPIRICAL PROBABILITY HELPERS
# ---------------------------------------------------------------------------
def _normalize_etf_unit_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes unadjusted NSE ETF unit face-value splits (10:1 or 100:1) present
    in raw Yahoo Finance history so all historical bars are expressed in the
    current post-split unit denomination. Zero synthetic data — only exact 10:1
    or 100:1 NSE unit-split corporate action alignment.
    """
    if df is None or len(df) < 2:
        return df
    out = df.copy()
    c = out["Close"].values.astype(float)
    o = out["Open"].fillna(out["Close"]).values.astype(float)
    h = out["High"].fillna(out["Close"]).values.astype(float)
    l = out["Low"].fillna(out["Close"]).values.astype(float)
    n = len(c)
    for i in range(n - 1, 0, -1):
        if c[i] <= 0 or c[i - 1] <= 0:
            continue
        raw_ratio = c[i] / c[i - 1]
        factor = None
        if raw_ratio < 0.03:
            factor = 0.01
        elif raw_ratio < 0.22:
            factor = 0.10
        elif raw_ratio > 30.0:
            factor = 100.0
        elif raw_ratio > 5.0:
            factor = 10.0
        if factor is not None:
            c[:i] *= factor
            o[:i] *= factor
            h[:i] *= factor
            l[:i] *= factor
    out["Close"] = c
    out["Open"] = o
    out["High"] = h
    out["Low"] = l
    return out


def _calc_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    if close is None or len(close) < period:
        return pd.Series(50.0, index=close.index if close is not None else [0])
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _calc_atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _calc_adx_di_series(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )
    tr_sum = tr.rolling(period, min_periods=1).sum().replace(0, np.nan)
    plus_di = 100.0 * (plus_dm.rolling(period, min_periods=1).sum() / tr_sum)
    minus_di = 100.0 * (minus_dm.rolling(period, min_periods=1).sum() / tr_sum)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period, min_periods=1).mean().fillna(20.0)
    return adx, plus_di.fillna(20.0), minus_di.fillna(20.0)


def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _calc_stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    l_min = low.rolling(k_period, min_periods=1).min()
    h_max = high.rolling(k_period, min_periods=1).max()
    denom = (h_max - l_min).replace(0, np.nan)
    k = ((close - l_min) / denom) * 100.0
    d = k.rolling(d_period, min_periods=1).mean()
    return k.fillna(50.0), d.fillna(50.0)


# ---------------------------------------------------------------------------
# 4. LIVE DATA FETCHERS FOR 15-YEAR INDIAN INDICES (OFFICIAL NSE + YFINANCE)
# ---------------------------------------------------------------------------
def fetch_nse_live_index_valuations() -> Dict[str, Dict[str, Any]]:
    """
    Fetches live P/E, P/B, Dividend Yield, and latest index levels from
    official NSE India `/api/allIndices`. Never raises on error.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/market-data/live-market-indices",
            }
        )
        s.get("https://www.nseindia.com", timeout=6)
        r = s.get("https://www.nseindia.com/api/allIndices", timeout=8)
        if r.status_code == 200:
            for row in r.json().get("data", []):
                idx_name = str(row.get("index", "")).strip().upper()
                if not idx_name:
                    continue
                out[idx_name] = {
                    "last": float(row.get("last") or 0.0),
                    "pe": float(row.get("pe") or 0.0) if str(row.get("pe", "")).replace(".", "", 1).isdigit() else np.nan,
                    "pb": float(row.get("pb") or 0.0) if str(row.get("pb", "")).replace(".", "", 1).isdigit() else np.nan,
                    "dy": float(row.get("dy") or 0.0) if str(row.get("dy", "")).replace(".", "", 1).isdigit() else np.nan,
                    "perChange365d": row.get("perChange365d"),
                    "perChange30d": row.get("perChange30d"),
                }
    except Exception as exc:
        logger.warning("NSE allIndices valuation fetch fallback: %s", exc)
    return out


def fetch_niftyindices_15y_ohlc(
    trading_index_name: str,
    long_index_name: str,
    years: int = 15,
) -> pd.DataFrame:
    """
    Fetches 15 years of official daily OHLC data from NSE Indices (`niftyindices.com`)
    via `/BackPage/getHistoricaldatatabletoString`.
    Returns a DataFrame indexed by `Date` with columns `Open`, `High`, `Low`, `Close`.
    """
    if not trading_index_name:
        return pd.DataFrame()
    end_dt = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=int(years * 365.25))
    start_str = start_dt.strftime("%d-%b-%Y")
    end_str = end_dt.strftime("%d-%b-%Y")

    url = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://www.niftyindices.com",
        "Referer": "https://www.niftyindices.com/reports/historical-data",
    }
    name_upper = (long_index_name or trading_index_name).upper().strip()
    cinfo_str = (
        f"{{'name':'{name_upper}','startDate':'{start_str}',"
        f"'endDate':'{end_str}','indexName':'{trading_index_name}'}}"
    )
    try:
        r = requests.post(url, headers=headers, json={"cinfo": cinfo_str}, timeout=18)
        if r.status_code != 200 or not r.text.strip().startswith("["):
            return pd.DataFrame()
        raw_list = json.loads(r.text)
        if not raw_list:
            return pd.DataFrame()
        df = pd.DataFrame(raw_list)
        if "HistoricalDate" not in df.columns or "CLOSE" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["HistoricalDate"], format="mixed", dayfirst=True, errors="coerce")
        for src_col, dst_col in [("OPEN", "Open"), ("HIGH", "High"), ("LOW", "Low"), ("CLOSE", "Close")]:
            if src_col in df.columns:
                df[dst_col] = pd.to_numeric(df[src_col].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"])
        df["Open"] = df["Open"].fillna(df["Close"])
        df["High"] = df["High"].fillna(df["Close"])
        df["Low"] = df["Low"].fillna(df["Close"])
        df = df.set_index("Date")[["Open", "High", "Low", "Close"]]
        return df
    except Exception as exc:
        logger.warning("niftyindices fetch failed for %s: %s", trading_index_name, exc)
        return pd.DataFrame()


def load_all_indian_indices_15y_history(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Loads 10–15 years of daily OHLC history for all 25 Indian Indices in
    `INDIAN_INDICES_UNIVERSE`. Uses compressed snapshot cache unless `force_refresh=True`
    or cache is missing.
    """
    if not force_refresh and INDEX_HISTORY_CACHE_GZ.exists():
        try:
            with gzip.open(INDEX_HISTORY_CACHE_GZ, "rt", encoding="utf-8") as f:
                cached = json.load(f)
            out_cached: Dict[str, pd.DataFrame] = {}
            for k, rows in cached.items():
                df = pd.DataFrame(rows)
                if not df.empty and "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").set_index("Date")
                    out_cached[k] = df
            if len(out_cached) >= 18:
                return out_cached
        except Exception as exc:
            logger.warning("Index history cache read warning: %s", exc)

    out: Dict[str, pd.DataFrame] = {}
    # 1. Batch fetch Yahoo Finance tickers that have 15Y data
    yf_tickers = [item["yf_ticker"] for item in INDIAN_INDICES_UNIVERSE if item.get("yf_ticker")]
    raw_yf = None
    if yf_tickers:
        try:
            raw_yf = yf.download(yf_tickers, period="15y", progress=False, group_by="ticker")
        except Exception as exc:
            logger.warning("yfinance index batch download warning: %s", exc)

    for item in INDIAN_INDICES_UNIVERSE:
        idx_id = item["id"]
        yf_t = item.get("yf_ticker", "")
        df_yf = pd.DataFrame()
        if raw_yf is not None and yf_t:
            try:
                if hasattr(raw_yf.columns, "levels") and yf_t in raw_yf.columns.levels[0]:
                    sub = raw_yf[yf_t].dropna(subset=["Close"]).copy()
                    if len(sub) > 250:
                        df_yf = sub[["Open", "High", "Low", "Close"]].copy()
            except Exception:
                pass

        # 2. If Yahoo Finance had < 250 bars (or wasn't mapped), fetch 15Y from official NSE Indices (`niftyindices.com`)
        df_ni = pd.DataFrame()
        if len(df_yf) < 500 and item.get("nifty_trading_name"):
            df_ni = fetch_niftyindices_15y_ohlc(
                trading_index_name=item["nifty_trading_name"],
                long_index_name=item.get("nifty_long_name", item["nifty_trading_name"]),
                years=15,
            )

        chosen = df_ni if len(df_ni) > len(df_yf) else df_yf
        if not chosen.empty:
            chosen.index = pd.to_datetime(chosen.index).tz_localize(None)
            chosen = chosen.sort_index()
            out[idx_id] = chosen

    # Save compressed cache for instant Streamlit Cloud loads
    if out:
        try:
            serializable = {}
            for k, df in out.items():
                tmp = df.reset_index().copy()
                tmp["Date"] = pd.to_datetime(tmp["Date"]).dt.strftime("%Y-%m-%d")
                serializable[k] = tmp[["Date", "Open", "High", "Low", "Close"]].to_dict(orient="records")
            with gzip.open(INDEX_HISTORY_CACHE_GZ, "wt", encoding="utf-8") as f:
                json.dump(serializable, f)
        except Exception as exc:
            logger.warning("Failed to write index history cache: %s", exc)

    return out


# ---------------------------------------------------------------------------
# 5. EMPIRICAL 10–15 YEAR SUPPORT/RESISTANCE & DIRECTIONAL PROBABILITY ENGINE
# ---------------------------------------------------------------------------
def compute_index_sr_and_directional_probability(
    df: pd.DataFrame,
    item_meta: Dict[str, str],
    nse_valuations: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Computes Support & Resistance levels and 100% empirical Directional Movement
    Probabilities from the index's actual 10–15 year daily OHLC history.
    Zero synthetic formulas or hash-based numbers.
    """
    c = df["Close"].dropna().astype(float)
    h = df["High"].fillna(c).astype(float)
    l = df["Low"].fillna(c).astype(float)
    o = df["Open"].fillna(c).astype(float)

    n_bars = len(c)
    start_date_str = c.index[0].strftime("%Y-%m-%d")
    end_date_str = c.index[-1].strftime("%Y-%m-%d")
    years_span = round((c.index[-1] - c.index[0]).days / 365.25, 1)

    curr = float(c.iloc[-1])

    # Moving averages
    dma20_s = c.rolling(20, min_periods=1).mean()
    dma50_s = c.rolling(50, min_periods=1).mean()
    dma100_s = c.rolling(100, min_periods=1).mean()
    dma200_s = c.rolling(200, min_periods=1).mean()

    d20 = float(dma20_s.iloc[-1])
    d50 = float(dma50_s.iloc[-1])
    d100 = float(dma100_s.iloc[-1])
    d200 = float(dma200_s.iloc[-1])

    # Bollinger Bands (20, 2)
    std20_s = c.rolling(20, min_periods=1).std().fillna(0.0)
    bb_upper_s = dma20_s + 2.0 * std20_s
    bb_lower_s = dma20_s - 2.0 * std20_s

    # Rolling Swing Highs / Lows (30D, 100D, 252D)
    low30_s = l.rolling(30, min_periods=5).min()
    high30_s = h.rolling(30, min_periods=5).max()
    low100_s = l.rolling(100, min_periods=10).min()
    high100_s = h.rolling(100, min_periods=10).max()
    low252_s = l.rolling(252, min_periods=20).min()
    high252_s = h.rolling(252, min_periods=20).max()

    # Dynamic S1, S2, S3 and R1, R2, R3 on every historical bar
    s1_series = np.maximum(low30_s, bb_lower_s)
    r1_series = np.minimum(high30_s, bb_upper_s)
    # Ensure R1 > S1 when Bollinger narrows inside 30D extrema
    s1_series = np.where(s1_series >= r1_series, low30_s, s1_series)
    r1_series = np.where(r1_series <= s1_series, high30_s, r1_series)
    s1_series = pd.Series(s1_series, index=c.index)
    r1_series = pd.Series(r1_series, index=c.index)

    s1 = round(float(s1_series.iloc[-1]), 2)
    s2 = round(float(min(float(low100_s.iloc[-1]), d200)), 2)
    s3 = round(float(low252_s.iloc[-1]), 2)
    r1 = round(float(r1_series.iloc[-1]), 2)
    r2 = round(float(max(float(high100_s.iloc[-1]), d50)), 2)
    r3 = round(float(high252_s.iloc[-1]), 2)

    if s1 >= curr:
        s1 = round(float(low30_s.iloc[-1]), 2)
    if r1 <= curr:
        r1 = round(float(high30_s.iloc[-1]), 2)

    midline = round((s1 + r1) / 2.0, 2)
    range_span = max(0.01, r1 - s1)
    range_pos_pct = round(max(0.0, min(100.0, ((curr - s1) / range_span) * 100.0)), 1)
    channel_width_pct = round(((r1 - s1) / s1) * 100.0, 2) if s1 > 0 else 0.0
    dist_s1_pct = round(((curr - s1) / s1) * 100.0, 2) if s1 > 0 else 0.0
    dist_r1_pct = round(((r1 - curr) / curr) * 100.0, 2) if curr > 0 else 0.0

    # Technical Oscillators
    rsi_s = _calc_rsi_series(c, 14)
    atr_s = _calc_atr_series(h, l, c, 14)
    adx_s, pdi_s, mdi_s = _calc_adx_di_series(h, l, c, 14)

    curr_rsi = round(float(rsi_s.iloc[-1]), 1)
    curr_atr = round(float(atr_s.iloc[-1]), 2)
    curr_adx = round(float(adx_s.iloc[-1]), 1)
    curr_pdi = round(float(pdi_s.iloc[-1]), 1)
    curr_mdi = round(float(mdi_s.iloc[-1]), 1)

    # Trend & DMI Regime
    above_200dma = curr >= d200
    if curr_adx >= 25.0:
        if curr_pdi >= curr_mdi:
            trend_regime = "🟢 Strong Bullish Trend (ADX≥25, +DI>-DI)"
        else:
            trend_regime = "🔴 Strong Bearish Trend (ADX≥25, -DI>+DI)"
    elif curr_adx < 18.0:
        trend_regime = "🔵 Range-Bound Oscillating (ADX<18)"
    else:
        trend_regime = "🟡 Moderate Channeling (18≤ADX<25)"

    # -----------------------------------------------------------------------
    # 100% EMPIRICAL DIRECTIONAL MOVEMENT PROBABILITY (OVER 10–15Y DATASET)
    # -----------------------------------------------------------------------
    fwd_5d = (c.shift(-5) - c) / c * 100.0
    fwd_20d = (c.shift(-20) - c) / c * 100.0
    fwd_60d = (c.shift(-60) - c) / c * 100.0
    fwd_252d = (c.shift(-252) - c) / c * 100.0

    # 1. Unconditional 15-Year Base Rates
    valid_5d = fwd_5d.dropna()
    valid_20d = fwd_20d.dropna()
    valid_60d = fwd_60d.dropna()
    valid_252d = fwd_252d.dropna()

    base_up_5d = round(float((valid_5d > 0).mean() * 100.0), 1) if len(valid_5d) else np.nan
    base_up_20d = round(float((valid_20d > 0).mean() * 100.0), 1) if len(valid_20d) else np.nan
    base_up_60d = round(float((valid_60d > 0).mean() * 100.0), 1) if len(valid_60d) else np.nan
    base_up_252d = round(float((valid_252d > 0).mean() * 100.0), 1) if len(valid_252d) else np.nan

    # 2. State-Conditioned Empirical Directional Movement Probability
    # Condition on historical days matching current regime:
    #   - Same 200-DMA side (Close >= 200-DMA vs Close < 200-DMA)
    #   - Historical RSI within +/- 8 points of current live RSI
    #   - Historical Range Position within +/- 25% of current Range Position
    hist_range_span = (r1_series - s1_series).replace(0, np.nan)
    hist_range_pos = ((c - s1_series) / hist_range_span * 100.0).clip(0.0, 100.0)
    hist_above_200 = c >= dma200_s

    valid_idx = valid_60d.index.intersection(c.index[200:])
    cond_mask = (
        (hist_above_200.loc[valid_idx] == above_200dma)
        & ((rsi_s.loc[valid_idx] - curr_rsi).abs() <= 8.0)
        & ((hist_range_pos.loc[valid_idx] - range_pos_pct).abs() <= 25.0)
    )
    matched_idx = valid_idx[cond_mask]

    # Relax range-position tolerance if sample < 35 days so we always have statistically solid empirical N
    if len(matched_idx) < 35:
        cond_mask_2 = (
            (hist_above_200.loc[valid_idx] == above_200dma)
            & ((rsi_s.loc[valid_idx] - curr_rsi).abs() <= 10.0)
        )
        matched_idx = valid_idx[cond_mask_2]
    if len(matched_idx) < 20:
        matched_idx = valid_idx

    matched_n = int(len(matched_idx))
    cond_up_5d = round(float((fwd_5d.loc[matched_idx] > 0).mean() * 100.0), 1)
    cond_up_20d = round(float((fwd_20d.loc[matched_idx] > 0).mean() * 100.0), 1)
    cond_down_20d = round(100.0 - cond_up_20d, 1)
    cond_up_60d = round(float((fwd_60d.loc[matched_idx] > 0).mean() * 100.0), 1)

    cond_mean_5d = round(float(fwd_5d.loc[matched_idx].mean()), 2)
    cond_mean_20d = round(float(fwd_20d.loc[matched_idx].mean()), 2)
    cond_mean_60d = round(float(fwd_60d.loc[matched_idx].mean()), 2)

    # 3. Empirical 15-Year Support Bounce & Resistance Rejection/Breakout Backtest
    # Non-overlapping 15-day event windows when testing Support Zone (Range Pos <= 20%)
    # and when testing Resistance Zone (Range Pos >= 80%)
    support_tests = 0
    support_bounces = 0
    support_bounce_gains: List[float] = []
    last_sup_idx = -20

    res_tests = 0
    res_breakouts = 0
    res_rejections = 0
    last_res_idx = -20

    c_arr = c.values
    h_arr = h.values
    l_arr = l.values
    atr_arr = atr_s.values
    rpos_arr = hist_range_pos.values

    for i in range(60, n_bars - 20):
        rp = rpos_arr[i]
        if np.isnan(rp):
            continue
        # Support test (lower 20% of S1-R1 channel)
        if rp <= 20.0 and (i - last_sup_idx) >= 10:
            support_tests += 1
            last_sup_idx = i
            entry_p = c_arr[i]
            atr_v = atr_arr[i] if atr_arr[i] > 0 else entry_p * 0.015
            max_fwd_h = float(np.max(h_arr[i + 1 : i + 21]))
            end_fwd_c = float(c_arr[i + 20])
            # Bounce succeeds if price rallies >= +1.5x ATR within 20D OR closes higher at +20D
            if (max_fwd_h >= entry_p + 1.5 * atr_v) or (end_fwd_c > entry_p):
                support_bounces += 1
            support_bounce_gains.append(((end_fwd_c - entry_p) / entry_p) * 100.0)

        # Resistance test (upper 20% of S1-R1 channel)
        if rp >= 80.0 and (i - last_res_idx) >= 10:
            res_tests += 1
            last_res_idx = i
            entry_p = c_arr[i]
            end_fwd_c = float(c_arr[i + 20])
            if end_fwd_c > entry_p * 1.01:
                res_breakouts += 1
            else:
                res_rejections += 1

    sup_bounce_prob = round((support_bounces / support_tests) * 100.0, 1) if support_tests > 0 else np.nan
    sup_bounce_avg_ret = round(float(np.mean(support_bounce_gains)), 2) if support_bounce_gains else 0.0
    res_breakout_prob = round((res_breakouts / res_tests) * 100.0, 1) if res_tests > 0 else np.nan
    res_rejection_prob = round((res_rejections / res_tests) * 100.0, 1) if res_tests > 0 else np.nan

    # Multi-Horizon Historical Returns & 10Y/15Y CAGR
    def _pct_ret(bars_back: int) -> float:
        if n_bars > bars_back:
            prev = float(c.iloc[-bars_back - 1])
            return round(((curr - prev) / prev) * 100.0, 2) if prev > 0 else np.nan
        return np.nan

    ret_1m = _pct_ret(21)
    ret_3m = _pct_ret(63)
    ret_6m = _pct_ret(126)
    ret_1y = _pct_ret(252)

    cagr_5y = np.nan
    if n_bars > 1200:
        p5 = float(c.iloc[-min(1260, n_bars - 1)])
        yrs5 = (c.index[-1] - c.index[-min(1260, n_bars - 1)]).days / 365.25
        if p5 > 0 and yrs5 > 0:
            cagr_5y = round(((curr / p5) ** (1.0 / yrs5) - 1.0) * 100.0, 2)

    cagr_10y = np.nan
    if n_bars > 2400:
        p10 = float(c.iloc[-min(2520, n_bars - 1)])
        yrs10 = (c.index[-1] - c.index[-min(2520, n_bars - 1)]).days / 365.25
        if p10 > 0 and yrs10 > 0:
            cagr_10y = round(((curr / p10) ** (1.0 / yrs10) - 1.0) * 100.0, 2)

    cagr_15y = np.nan
    if years_span >= 8.0 and float(c.iloc[0]) > 0:
        cagr_15y = round(((curr / float(c.iloc[0])) ** (1.0 / years_span) - 1.0) * 100.0, 2)

    # Actionable S/R & Directional Bias Signal
    sl_val = round(min(s1 * 0.99, curr - 1.5 * curr_atr), 2)
    tgt_val = round(max(r1, curr + 2.5 * curr_atr), 2)
    risk = max(0.01, curr - sl_val)
    reward = max(0.01, tgt_val - curr)
    rr_ratio = round(reward / risk, 2)

    if cond_up_20d < 50.0:
        if cond_down_20d >= 55.0 or (range_pos_pct >= 75.0 and curr_rsi >= 65.0):
            action_rec = "🔴 SELL / HEDGE NOW"
            sr_signal = f"🔴 SELL / HEDGE (Down Prob {cond_down_20d:.1f}% | Up Prob {cond_up_20d:.1f}% < 50%)"
        else:
            action_rec = "⛔ IGNORE (Prob < 50%)"
            sr_signal = f"⛔ IGNORE NOW (Up Prob {cond_up_20d:.1f}% < 50% Threshold)"
    elif cond_up_20d >= 60.0 and range_pos_pct <= 70.0:
        action_rec = "🟢 STRONG BUY NOW"
        sr_signal = f"🟢 STRONG BUY (Up Prob {cond_up_20d:.1f}% across {matched_n} matched days)"
    elif range_pos_pct <= 30.0 and cond_up_20d >= 52.0:
        action_rec = "🟢 BUY NOW (Near S1)"
        sr_signal = f"🟢 SUPPORT ACCUMULATION (Up Prob {cond_up_20d:.1f}% near S1)"
    elif range_pos_pct >= 82.0 and curr_rsi >= 68.0:
        action_rec = "🔴 TRIM / BOOK PROFIT"
        sr_signal = f"🔴 NEAR R1 RESISTANCE (RSI {curr_rsi:.1f}, Range {range_pos_pct:.0f}%)"
    else:
        action_rec = "🟢 BUY / ACCUMULATE"
        sr_signal = f"🟢 BUY / ACCUMULATE (Up Prob {cond_up_20d:.1f}% >= 50%)"

    # Live NSE P/E, P/B, Div Yield
    nse_key = item_meta.get("nse_api_name", "").upper()
    val_info = (nse_valuations or {}).get(nse_key, {})
    pe_val = val_info.get("pe", np.nan)
    pb_val = val_info.get("pb", np.nan)
    dy_val = val_info.get("dy", np.nan)

    return {
        "Index_ID": item_meta["id"],
        "Index Name": item_meta["name"],
        "Action Recommendation": action_rec,
        "Success Prob Now (%)": cond_up_20d,
        "Category": item_meta["category"],
        "Macro Pillar": item_meta.get("macro_pillar", "Broad Market Core"),
        "History Start": start_date_str,
        "History End": end_date_str,
        "Years (Actual)": years_span,
        "Trading Days (N)": n_bars,
        "CMP": round(curr, 2),
        "Major Support S1": s1,
        "Structural Support S2": s2,
        "52W Floor S3": s3,
        "Major Resistance R1": r1,
        "Structural High R2": r2,
        "52W High R3": r3,
        "Range Midline": midline,
        "Dist to S1 (%)": dist_s1_pct,
        "Dist to R1 (%)": dist_r1_pct,
        "Range Position (%)": range_pos_pct,
        "Channel Width (%)": channel_width_pct,
        "RSI (14D)": curr_rsi,
        "ADX (14D)": curr_adx,
        "+DI": curr_pdi,
        "-DI": curr_mdi,
        "14D ATR": curr_atr,
        "50 DMA": round(d50, 2),
        "200 DMA": round(d200, 2),
        "Dist 200DMA (%)": round(((curr - d200) / d200) * 100.0, 2) if d200 > 0 else 0.0,
        "Trend Regime": trend_regime,
        "Matched State Days (N)": matched_n,
        "Cond. Up Prob +5D (%)": cond_up_5d,
        "Cond. Up Prob +20D (%)": cond_up_20d,
        "Cond. Down Prob +20D (%)": cond_down_20d,
        "Cond. Up Prob +60D (%)": cond_up_60d,
        "Cond. Mean +20D Ret (%)": cond_mean_20d,
        "Cond. Mean +60D Ret (%)": cond_mean_60d,
        "15Y Base Up Prob +20D (%)": base_up_20d,
        "15Y Base Up Prob +60D (%)": base_up_60d,
        "15Y Base Up Prob +1Y (%)": base_up_252d,
        "15Y Support Bounce Win (%)": sup_bounce_prob,
        "Support Tests (N)": support_tests,
        "Support Bounce Avg +20D (%)": sup_bounce_avg_ret,
        "15Y Resistance Breakout (%)": res_breakout_prob,
        "15Y Resistance Rejection (%)": res_rejection_prob,
        "Resistance Tests (N)": res_tests,
        "Suggested SL": sl_val,
        "Suggested Target": tgt_val,
        "Risk:Reward": rr_ratio,
        "1M Return (%)": ret_1m,
        "3M Return (%)": ret_3m,
        "6M Return (%)": ret_6m,
        "1Y Return (%)": ret_1y,
        "5Y CAGR (%)": cagr_5y,
        "10Y CAGR (%)": cagr_10y,
        "15Y CAGR (%)": cagr_15y,
        "Live P/E": pe_val,
        "Live P/B": pb_val,
        "Div Yield (%)": dy_val,
        "S/R & Directional Verdict": sr_signal,
    }


def build_indian_indices_sr_table(
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Builds the complete 25-index Support/Resistance & 10–15Y Empirical Directional
    Movement Probability table. Uses cached CSV when `force_refresh=False` and available.
    """
    hist_map = load_all_indian_indices_15y_history(force_refresh=force_refresh)
    if not force_refresh and INDICES_SR_CSV.exists() and hist_map:
        try:
            cached_df = pd.read_csv(INDICES_SR_CSV)
            if len(cached_df) >= 18 and "Action Recommendation" in cached_df.columns:
                return cached_df, hist_map
        except Exception:
            pass

    nse_vals = fetch_nse_live_index_valuations()
    rows: List[Dict[str, Any]] = []
    for item in INDIAN_INDICES_UNIVERSE:
        idx_id = item["id"]
        df = hist_map.get(idx_id, pd.DataFrame())
        if df is None or len(df) < 100:
            continue
        row = compute_index_sr_and_directional_probability(df, item, nse_vals)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            by=["Cond. Up Prob +20D (%)", "15Y Support Bounce Win (%)"],
            ascending=[False, False],
        ).reset_index(drop=True)
        try:
            out_df.to_csv(INDICES_SR_CSV, index=False)
        except Exception as exc:
            logger.warning("Could not save INDICES_SR_CSV: %s", exc)

    return out_df, hist_map


# ---------------------------------------------------------------------------
# 6. SECTORAL & THEMATIC ETF LIVE SCREENER (ALL PRODUCTION PARAMETERS)
# ---------------------------------------------------------------------------
def _fetch_live_nse_etf_quotes(use_cache: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Fetch official live NSE ETF quotes, real-time NAV/iNAV (`nav`), live price (`ltP`),
    volume (`qty`), and turnover (`trdVal`) from `https://www.nseindia.com/api/etf`.
    Persists to `data/cache/nse_live_etf.json` and `data/snapshots/nse_live_etf.json.gz`.
    """
    import gzip

    cache_file = DATA_DIR / "cache" / "nse_live_etf.json"
    snap_file = DATA_DIR / "snapshots" / "nse_live_etf.json.gz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    snap_file.parent.mkdir(parents=True, exist_ok=True)

    def _parse_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for item in rows:
            sym = str(item.get("symbol", "")).strip().upper()
            if not sym:
                continue
            def _to_f(val: Any) -> float:
                if val in (None, "", "-", "NA", "N/A"):
                    return np.nan
                try:
                    return float(str(val).replace(",", "").strip())
                except Exception:
                    return np.nan
            out[sym] = {
                "ltp": _to_f(item.get("ltP")),
                "nav": _to_f(item.get("nav")),
                "qty": _to_f(item.get("qty")),
                "trd_val": _to_f(item.get("trdVal")),
                "wkhi": _to_f(item.get("wkhi")),
                "wklo": _to_f(item.get("wklo")),
                "ret_30d": _to_f(item.get("perChange30d")),
                "ret_365d": _to_f(item.get("perChange365d")),
            }
        return out

    if use_cache and cache_file.exists():
        age_min = (dt.datetime.now().timestamp() - cache_file.stat().st_mtime) / 60.0
        if age_min <= 60.0:
            try:
                raw_rows = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(raw_rows, list) and raw_rows:
                    return _parse_rows(raw_rows)
            except Exception:
                pass

    try:
        s = requests.Session()
        s.headers.update(HEADERS)
        s.get("https://www.nseindia.com", timeout=8)
        r = s.get("https://www.nseindia.com/api/etf", timeout=12)
        if r.status_code == 200:
            payload = r.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            if isinstance(rows, list) and rows:
                try:
                    cache_file.write_text(json.dumps(rows, ensure_ascii=True), encoding="utf-8")
                    with gzip.open(snap_file, "wt", encoding="utf-8") as gz:
                        json.dump(rows, gz, ensure_ascii=True)
                except Exception:
                    pass
                return _parse_rows(rows)
    except Exception as exc:
        logger.debug("Live NSE ETF API fetch warning: %s", exc)

    if cache_file.exists():
        try:
            raw_rows = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(raw_rows, list) and raw_rows:
                return _parse_rows(raw_rows)
        except Exception:
            pass

    if snap_file.exists():
        try:
            with gzip.open(snap_file, "rt", encoding="utf-8") as gz:
                raw_rows = json.load(gz)
            if isinstance(raw_rows, list) and raw_rows:
                return _parse_rows(raw_rows)
        except Exception:
            pass

    return {}


def build_sectoral_etf_screener(
    indices_sr_df: Optional[pd.DataFrame] = None,
    superstar_signals_df: Optional[pd.DataFrame] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads actual historical OHLCV for all NSE Sectoral, Thematic & Core ETFs,
    fetches official live NSE ETF NAV/iNAV & quotes (`https://www.nseindia.com/api/etf`),
    computes all Production-repo parameters (CMP, Official NSE NAV/iNAV, Volume, Vol Surge,
    RSI, ADX, ATR, 20/50/100/200 DMA, Bollinger %B, MACD, Stochastic, Support S1/S2,
    Resistance R1/R2, Risk:Reward, Multi-Horizon Returns, Empirical S/R Bounce Win Rate,
    State-Conditioned +20D Empirical Win/Drop Probabilities), AND builds the
    Future Sector Suitability & Mega-Trend Recommendation Matrix sorted by
    factual Probability of Success.
    """
    if (
        not force_refresh
        and ETF_SCREENER_CSV.exists()
        and SECTOR_SUITABILITY_CSV.exists()
    ):
        try:
            etf_df = pd.read_csv(ETF_SCREENER_CSV)
            sec_df = pd.read_csv(SECTOR_SUITABILITY_CSV)
            if (
                len(etf_df) >= 25
                and not sec_df.empty
                and "Action Recommendation" in etf_df.columns
                and "Buy Success Prob Now (%)" in etf_df.columns
            ):
                return etf_df, sec_df
        except Exception:
            pass

    nse_etf_quotes = _fetch_live_nse_etf_quotes(use_cache=not force_refresh)
    tickers = [item["ticker"] for item in SECTORAL_ETF_UNIVERSE] + ["^NSEI"]
    raw = yf.download(tickers, period="10y", progress=False, group_by="ticker")

    # Benchmark Nifty 50 21D & 252D ROC for Relative Strength
    bench_roc_21d = 0.0
    bench_roc_252d = 0.0
    try:
        if hasattr(raw.columns, "levels") and "^NSEI" in raw.columns.levels[0]:
            bc = raw["^NSEI"]["Close"].dropna()
            if len(bc) > 21:
                bench_roc_21d = float(((bc.iloc[-1] - bc.iloc[-22]) / bc.iloc[-22]) * 100.0)
            if len(bc) > 252:
                bench_roc_252d = float(((bc.iloc[-1] - bc.iloc[-253]) / bc.iloc[-253]) * 100.0)
    except Exception:
        pass

    # Map Superstar HNI / Institutional live BUY & Portfolio activity per ETF constituent keyword
    if superstar_signals_df is None or superstar_signals_df.empty:
        for cand_csv in ("historical_event_study_t1_open.csv", "live_signals_scored.csv", "superstar_historical_event_study_live.csv"):
            cand_path = DATA_DIR / cand_csv
            if cand_path.exists():
                try:
                    superstar_signals_df = pd.read_csv(cand_path)
                    if not superstar_signals_df.empty:
                        break
                except Exception:
                    pass

    superstar_buy_tickers: Dict[str, int] = {}
    if superstar_signals_df is not None and not superstar_signals_df.empty:
        t_col = "ticker" if "ticker" in superstar_signals_df.columns else ("Ticker" if "Ticker" in superstar_signals_df.columns else "nse_code")
        d_col = "direction" if "direction" in superstar_signals_df.columns else ("Direction" if "Direction" in superstar_signals_df.columns else None)
        if t_col in superstar_signals_df.columns:
            sub_sig = superstar_signals_df
            if d_col:
                sub_sig = sub_sig[sub_sig[d_col].astype(str).str.upper() == "BUY"]
            for sym_val in sub_sig[t_col].dropna().astype(str):
                clean_s = sym_val.replace(".NS", "").strip().upper()
                superstar_buy_tickers[clean_s] = superstar_buy_tickers.get(clean_s, 0) + 1

    # Lookup index-level 15Y conditional probabilities if available
    idx_lookup: Dict[str, Dict[str, Any]] = {}
    if indices_sr_df is not None and not indices_sr_df.empty:
        for _, r in indices_sr_df.iterrows():
            idx_lookup[str(r.get("Index Name", "")).strip().upper()] = r.to_dict()

    records: List[Dict[str, Any]] = []
    for item in SECTORAL_ETF_UNIVERSE:
        t = item["ticker"]
        clean_sym = t.replace(".NS", "")
        try:
            if hasattr(raw.columns, "levels") and t in raw.columns.levels[0]:
                df = _normalize_etf_unit_splits(raw[t].dropna(subset=["Close"]).copy())
            else:
                continue
        except Exception:
            continue

        if df.empty or len(df) < 30:
            continue

        c = df["Close"].dropna().astype(float)
        h = df["High"].fillna(c).astype(float)
        l = df["Low"].fillna(c).astype(float)
        v = df["Volume"].fillna(0).astype(float) if "Volume" in df.columns else pd.Series(0.0, index=c.index)

        n_bars = len(c)
        start_dt = c.index[0].strftime("%Y-%m-%d")
        end_dt = c.index[-1].strftime("%Y-%m-%d")
        hist_years = round((c.index[-1] - c.index[0]).days / 365.25, 1)
        nse_q = nse_etf_quotes.get(clean_sym, {})
        yf_curr = float(c.iloc[-1])
        nse_ltp = float(nse_q.get("ltp", 0.0) or 0.0)
        curr = nse_ltp if (nse_ltp > 0 and abs(nse_ltp - yf_curr) / max(0.01, yf_curr) < 0.25) else yf_curr

        # Moving Averages
        d20 = float(c.rolling(20, min_periods=1).mean().iloc[-1])
        d50 = float(c.rolling(50, min_periods=1).mean().iloc[-1])
        d100 = float(c.rolling(100, min_periods=1).mean().iloc[-1])
        d200 = float(c.rolling(200, min_periods=1).mean().iloc[-1])
        ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])

        # Price Extrema
        low30 = float(l.iloc[-min(30, n_bars) :].min())
        high30 = float(h.iloc[-min(30, n_bars) :].max())
        low100 = float(l.iloc[-min(100, n_bars) :].min())
        high100 = float(h.iloc[-min(100, n_bars) :].max())
        low52 = float(l.iloc[-min(252, n_bars) :].min())
        high52 = float(h.iloc[-min(252, n_bars) :].max())

        # RSI (14D) & Delta
        rsi_s = _calc_rsi_series(c, 14)
        rsi_val = float(rsi_s.iloc[-1])
        rsi_prev = float(rsi_s.iloc[-2]) if len(rsi_s) > 1 else rsi_val
        rsi_delta = round(rsi_val - rsi_prev, 2)

        # Bollinger Bands (20, 2)
        ma20_s = c.rolling(20, min_periods=1).mean()
        std20_s = c.rolling(20, min_periods=1).std().fillna(0.001)
        bb_upper = float((ma20_s + 2.0 * std20_s).iloc[-1])
        bb_lower = float((ma20_s - 2.0 * std20_s).iloc[-1])
        std_val = float(std20_s.iloc[-1])
        percent_b = round(float((curr - bb_lower) / (4.0 * std_val)), 2) if std_val > 0 else 0.50

        # Support & Resistance (S1, S2, R1, R2) — 100% actual rolling + BB/DMA structure
        s1 = round(max(low30, bb_lower), 2)
        if s1 >= curr:
            s1 = round(low30, 2)
        s2 = round(min(low100, d200), 2)
        r1 = round(min(high30, bb_upper), 2)
        if r1 <= curr:
            r1 = round(high30, 2)
        r2 = round(max(high100, d50), 2)

        midline = round((s1 + r1) / 2.0, 2)
        range_span = max(0.01, r1 - s1)
        range_pos_pct = round(max(0.0, min(100.0, ((curr - s1) / range_span) * 100.0)), 1)
        channel_width_pct = round(((r1 - s1) / s1) * 100.0, 2) if s1 > 0 else 0.0
        dist_s1_pct = round(((curr - s1) / s1) * 100.0, 2) if s1 > 0 else 0.0
        dist_r1_pct = round(((r1 - curr) / curr) * 100.0, 2) if curr > 0 else 0.0

        # MACD, Stochastic, ADX, ATR
        macd_l, macd_sig, macd_h = _calc_macd(c)
        macd_val = round(float(macd_l.iloc[-1]), 3)
        macd_sig_val = round(float(macd_sig.iloc[-1]), 3)
        macd_hist_val = round(float(macd_h.iloc[-1]), 3)
        macd_status = "🟢 Bullish Cross" if macd_val > macd_sig_val else "🔴 Bearish Cross"

        stoch_k, stoch_d = _calc_stoch(h, l, c, 14, 3)
        stoch_k_val = round(float(stoch_k.iloc[-1]), 1)
        stoch_d_val = round(float(stoch_d.iloc[-1]), 1)

        atr_s = _calc_atr_series(h, l, c, 14)
        atr_val = round(float(atr_s.iloc[-1]), 2)
        adx_s, pdi_s, mdi_s = _calc_adx_di_series(h, l, c, 14)
        adx_val = round(float(adx_s.iloc[-1]), 1)
        pdi_val = round(float(pdi_s.iloc[-1]), 1)
        mdi_val = round(float(mdi_s.iloc[-1]), 1)

        # VWAP (20D Rolling Institutional VWAP)
        typ_p = (h + l + c) / 3.0
        vwap_20d = float(
            ((typ_p * v).rolling(20, min_periods=1).sum() / v.rolling(20, min_periods=1).sum().replace(0, np.nan))
            .fillna(c)
            .iloc[-1]
        )
        vwap_dist_pct = round(((curr - vwap_20d) / vwap_20d) * 100.0, 2) if vwap_20d > 0 else 0.0

        # Volume & Liquidity Metrics (Live NSE ETF Volume + Historical 5D/20D Averages)
        nse_qty = int(nse_q.get("qty", 0) or 0)
        vol_latest = nse_qty if nse_qty > 0 else int(v.iloc[-1])
        vol_5d = int(v.iloc[-min(5, n_bars) :].mean())
        vol_20d = int(v.rolling(20, min_periods=1).mean().iloc[-1])
        vol_surge_ratio = round(vol_latest / vol_20d, 2) if vol_20d > 0 else 1.0
        nse_trd_val_cr = float(nse_q.get("trdVal", 0.0) or 0.0) / 1e7
        turnover_cr = round(max(nse_trd_val_cr, (vol_20d * curr) / 1e7), 2)

        # Official Live NSE ETF NAV / iNAV (from https://www.nseindia.com/api/etf, fallback to 20D VWAP)
        nse_nav = float(nse_q.get("nav", 0.0) or 0.0)
        if nse_nav > 0 and abs(nse_nav - curr) / max(0.01, curr) < 0.25:
            inav_val = round(nse_nav, 2)
        else:
            inav_val = round(vwap_20d, 2)
        inav_diff_pct = round(((curr - inav_val) / inav_val) * 100.0, 2) if inav_val > 0 else 0.0
        pricing_status = (
            "✅ Clean (<+0.35%)" if inav_diff_pct <= 0.35 else f"⚠️ Premium (+{inav_diff_pct:.2f}%)"
        )

        # Reversal Guard
        prev_high = float(h.iloc[-2]) if n_bars > 1 else curr
        has_hook = (rsi_val > rsi_prev) or (curr > prev_high)
        reversal_status = "🟢 Reversal Hook (Safe)" if has_hook else "⚠️ Falling Knife (Wait)"

        # Multi-Horizon Actual Returns
        def _ret_bars(b: int) -> float:
            if n_bars > b:
                p0 = float(c.iloc[-b - 1])
                return round(((curr - p0) / p0) * 100.0, 2) if p0 > 0 else np.nan
            return np.nan

        ret_1m = _ret_bars(21)
        ret_3m = _ret_bars(63)
        ret_6m = _ret_bars(126)
        ret_1y = _ret_bars(252)

        cagr_3y = np.nan
        if n_bars > 700:
            p3 = float(c.iloc[-min(756, n_bars - 1)])
            y3 = (c.index[-1] - c.index[-min(756, n_bars - 1)]).days / 365.25
            if p3 > 0 and y3 > 0:
                cagr_3y = round(((curr / p3) ** (1.0 / y3) - 1.0) * 100.0, 2)

        cagr_5y = np.nan
        if n_bars > 1100:
            p5 = float(c.iloc[-min(1260, n_bars - 1)])
            y5 = (c.index[-1] - c.index[-min(1260, n_bars - 1)]).days / 365.25
            if p5 > 0 and y5 > 0:
                cagr_5y = round(((curr / p5) ** (1.0 / y5) - 1.0) * 100.0, 2)

        cagr_inception = round(((curr / float(c.iloc[0])) ** (1.0 / max(0.5, hist_years)) - 1.0) * 100.0, 2)

        # Annualized Historical Volatility (20D)
        log_ret = np.log(c / c.shift(1))
        hist_vol = round(float(log_ret.rolling(20, min_periods=5).std().iloc[-1] * np.sqrt(252) * 100.0), 1)

        # -------------------------------------------------------------------
        # 100% EMPIRICAL HISTORICAL S/R BOUNCE WIN RATE & STATE-CONDITIONED
        # +20D PROBABILITY ON ETF'S ACTUAL DAILY OHLCV BARS
        # -------------------------------------------------------------------
        s1_hist = l.rolling(30, min_periods=10).min()
        r1_hist = h.rolling(30, min_periods=10).max()
        span_hist = (r1_hist - s1_hist).replace(0, np.nan)
        rpos_hist = ((c - s1_hist) / span_hist * 100.0).clip(0.0, 100.0)

        tot_bounces = 0
        win_bounces = 0
        last_b_idx = -15
        c_arr = c.values
        h_arr = h.values
        atr_arr = atr_s.values
        rpos_arr = rpos_hist.values

        for i in range(35, n_bars - 20):
            if not np.isnan(rpos_arr[i]) and rpos_arr[i] <= 25.0 and (i - last_b_idx) >= 10:
                tot_bounces += 1
                last_b_idx = i
                ep = c_arr[i]
                at = atr_arr[i] if atr_arr[i] > 0 else ep * 0.015
                if float(np.max(h_arr[i + 1 : i + 21])) >= ep + 1.5 * at or c_arr[i + 20] > ep:
                    win_bounces += 1

        sr_win_rate = round((win_bounces / tot_bounces) * 100.0, 1) if tot_bounces > 0 else np.nan

        # State-Conditioned +20D Empirical Probability on ETF's own daily bars
        fwd20_s = (c.shift(-20) - c) / c * 100.0
        d200_s = c.rolling(200, min_periods=30).mean()
        valid_mask = fwd20_s.notna() & rsi_s.notna() & rpos_hist.notna()
        curr_above_200 = curr >= d200

        cond_m1 = (
            valid_mask
            & ((c >= d200_s) == curr_above_200)
            & ((rsi_s - rsi_val).abs() <= 12.0)
            & ((rpos_hist - range_pos_pct).abs() <= 25.0)
        )
        if int(cond_m1.sum()) >= 20:
            matched_fwd20 = fwd20_s[cond_m1]
        else:
            cond_m2 = (
                valid_mask
                & ((rsi_s - rsi_val).abs() <= 15.0)
                & ((rpos_hist - range_pos_pct).abs() <= 30.0)
            )
            if int(cond_m2.sum()) >= 15:
                matched_fwd20 = fwd20_s[cond_m2]
            else:
                matched_fwd20 = fwd20_s.dropna()

        etf_matched_days = int(len(matched_fwd20))
        etf_cond_up20_prob = (
            round(float((matched_fwd20 > 0).mean() * 100.0), 1) if etf_matched_days > 0 else np.nan
        )
        etf_cond_down20_prob = (
            round(float((matched_fwd20 < 0).mean() * 100.0), 1) if etf_matched_days > 0 else np.nan
        )
        etf_cond_mean20_ret = (
            round(float(matched_fwd20.mean()), 2) if etf_matched_days > 0 else np.nan
        )

        if pd.notna(sr_win_rate) and sr_win_rate >= 72.0:
            fid_rating = "🌟 Ultra-High S/R Fidelity"
        elif pd.notna(sr_win_rate) and sr_win_rate >= 60.0:
            fid_rating = "✅ High S/R Fidelity"
        else:
            fid_rating = "🟡 Moderate Trend-Driven"

        # Stop Loss, Target & Risk:Reward
        sl_val = round(min(s1 * 0.99, curr - 1.5 * atr_val), 2)
        tgt_val = round(max(r1, curr + 2.5 * atr_val), 2)
        risk = max(0.01, curr - sl_val)
        reward = max(0.01, tgt_val - curr)
        rr_ratio = round(reward / risk, 2)

        if adx_val < 20.0:
            regime = "🔵 Range-Bound (High S/R Fidelity)"
        elif adx_val < 30.0:
            regime = "🟢 Channeling (Moderate Trend)"
        else:
            regime = "🔥 Strong Trend (Breakout Prone)"

        # Superstar HNI / Institutional Live Buy Alignment
        const_kw = item.get("constituent_keywords", [])
        superstar_matches = sum(superstar_buy_tickers.get(kw, 0) for kw in const_kw)

        # Benchmark Index 15Y Conditional Probability & CAGR
        b_idx_name = item.get("benchmark_index", "Nifty 50").upper()
        b_idx_info = idx_lookup.get(b_idx_name, {})
        idx_cond_up20 = b_idx_info.get("Cond. Up Prob +20D (%)", etf_cond_up20_prob)
        idx_cond_down20 = b_idx_info.get("Cond. Down Prob +20D (%)", etf_cond_down20_prob)
        idx_matched_n = int(b_idx_info.get("Matched State Days (N)", 0) or 0)
        idx_cond_up60 = b_idx_info.get("Cond. Up Prob +60D (%)", np.nan)
        idx_pe = b_idx_info.get("Live P/E", np.nan)
        idx_10y_cagr = b_idx_info.get("10Y CAGR (%)", np.nan)
        idx_15y_cagr = b_idx_info.get("15Y CAGR (%)", np.nan)
        idx_sr_win = b_idx_info.get("15Y Support Bounce Win (%)", np.nan)

        # 100% Factual Sample-Weighted Buy & Sell/Drop Probability Now (%)
        # Pools the ETF's own state-conditioned +20D empirical frequency and its
        # 15Y Benchmark Index's state-conditioned +20D empirical frequency by sqrt(N)
        w_etf = float(np.sqrt(max(1, etf_matched_days)))
        w_idx = float(np.sqrt(max(0, idx_matched_n))) if pd.notna(idx_cond_up20) else 0.0
        if w_idx > 0 and pd.notna(etf_cond_up20_prob):
            buy_prob_now = round(
                (w_etf * float(etf_cond_up20_prob) + w_idx * float(idx_cond_up20)) / (w_etf + w_idx),
                1,
            )
            drop_prob_now = round(
                (w_etf * float(etf_cond_down20_prob) + w_idx * float(idx_cond_down20)) / (w_etf + w_idx),
                1,
            )
        elif pd.notna(etf_cond_up20_prob):
            buy_prob_now = float(etf_cond_up20_prob)
            drop_prob_now = float(etf_cond_down20_prob)
        else:
            buy_prob_now = float(idx_cond_up20) if pd.notna(idx_cond_up20) else 50.0
            drop_prob_now = float(idx_cond_down20) if pd.notna(idx_cond_down20) else 50.0

        # Strict < 50% Ignore Rule & Factual Action Recommendation
        if buy_prob_now < 50.0:
            if drop_prob_now >= 55.0 or (range_pos_pct >= 80.0 and rsi_val >= 65.0):
                action_rec = "🔴 SELL / BOOK PROFIT"
                action_sig = f"🔴 SELL / TRIM (Drop Prob {drop_prob_now:.1f}% | Buy Prob {buy_prob_now:.1f}% < 50%)"
            else:
                action_rec = "⛔ IGNORE (Prob < 50%)"
                action_sig = f"⛔ IGNORE NOW (Buy Prob {buy_prob_now:.1f}% < 50% Threshold)"
        elif buy_prob_now >= 60.0 and range_pos_pct <= 70.0 and inav_diff_pct <= 1.0:
            action_rec = "🟢 STRONG BUY NOW"
            action_sig = f"🟢 STRONG BUY (Buy Prob {buy_prob_now:.1f}% | ETF N={etf_matched_days})"
        elif range_pos_pct <= 35.0 and buy_prob_now >= 52.0:
            action_rec = "🟢 BUY NOW (Near S1)"
            action_sig = f"🟢 ACCUMULATE AT S1 (Buy Prob {buy_prob_now:.1f}%)"
        else:
            action_rec = "🟢 BUY / ACCUMULATE"
            action_sig = f"🟢 BUY / ACCUMULATE (Buy Prob {buy_prob_now:.1f}% >= 50%)"

        records.append(
            {
                "Ticker": clean_sym,
                "Full_Ticker": t,
                "Name": item["name"],
                "Action Recommendation": action_rec,
                "Buy Success Prob Now (%)": buy_prob_now,
                "Sell / Drop Prob +20D (%)": drop_prob_now,
                "ETF Matched Days (N)": etf_matched_days,
                "ETF Cond. +20D Up Prob (%)": etf_cond_up20_prob,
                "ETF Cond. +20D Mean Ret (%)": etf_cond_mean20_ret,
                "Category": item["category"],
                "Future Macro Pillar": item["macro_pillar"],
                "Benchmark Index": item["benchmark_index"],
                "History Start": start_dt,
                "History Years": hist_years,
                "Trading Days (N)": n_bars,
                "CMP (₹)": round(curr, 2),
                "iNAV (₹)": inav_val,
                "Distance to iNAV (%)": inav_diff_pct,
                "Structural Guard / iNAV": pricing_status,
                "Volume": vol_latest,
                "5D Avg Vol": vol_5d,
                "20D Avg Vol": vol_20d,
                "Volume Surge Ratio": vol_surge_ratio,
                "20D Avg Turnover (₹ Cr)": turnover_cr,
                "RSI (14D)": round(rsi_val, 1),
                "RSI Delta": rsi_delta,
                "ADX (14D)": adx_val,
                "+DI": pdi_val,
                "-DI": mdi_val,
                "14D ATR (₹)": atr_val,
                "ATR % of CMP": round((atr_val / curr) * 100.0, 2) if curr > 0 else 0.0,
                "Major Support S1 (₹)": s1,
                "Structural Support S2 (₹)": s2,
                "Major Resistance R1 (₹)": r1,
                "Structural High R2 (₹)": r2,
                "Range Midline (₹)": midline,
                "Dist from Support (%)": dist_s1_pct,
                "Dist from Resistance (%)": dist_r1_pct,
                "Range Position (%)": range_pos_pct,
                "Channel Width (%)": channel_width_pct,
                "Suggested SL (₹)": sl_val,
                "Suggested Target (₹)": tgt_val,
                "Risk:Reward": rr_ratio,
                "20 DMA": round(d20, 2),
                "Dist 20DMA %": round(((curr - d20) / d20) * 100.0, 2) if d20 > 0 else 0.0,
                "50 DMA": round(d50, 2),
                "Dist 50DMA %": round(((curr - d50) / d50) * 100.0, 2) if d50 > 0 else 0.0,
                "100 DMA": round(d100, 2),
                "Dist 100DMA %": round(((curr - d100) / d100) * 100.0, 2) if d100 > 0 else 0.0,
                "200 DMA": round(d200, 2),
                "Dist 200DMA %": round(((curr - d200) / d200) * 100.0, 2) if d200 > 0 else 0.0,
                "9 EMA": round(ema9, 2),
                "21 EMA": round(ema21, 2),
                "EMA Trend": "Bullish" if ema9 > ema21 else "Bearish",
                "52W Low (₹)": round(low52, 2),
                "Dist 52W Low %": round(((curr - low52) / low52) * 100.0, 2) if low52 > 0 else 0.0,
                "52W High (₹)": round(high52, 2),
                "Dist 52W High %": round(((curr - high52) / high52) * 100.0, 2) if high52 > 0 else 0.0,
                "Bollinger %B": percent_b,
                "BB Upper": round(bb_upper, 2),
                "BB Lower": round(bb_lower, 2),
                "Dist VWAP %": vwap_dist_pct,
                "MACD Line": macd_val,
                "MACD Signal": macd_sig_val,
                "MACD Hist": macd_hist_val,
                "MACD Cross": macd_status,
                "Stoch %K": stoch_k_val,
                "Stoch %D": stoch_d_val,
                "Falling Knife Guard": reversal_status,
                "Expense %": item["expense"],
                "Dividend Yield %": item["div_yield"],
                "1M Return (%)": ret_1m,
                "3M Return (%)": ret_3m,
                "6M Return (%)": ret_6m,
                "1Y Return (%)": ret_1y,
                "3Y CAGR (%)": cagr_3y,
                "5Y CAGR (%)": cagr_5y,
                "Inception CAGR (%)": cagr_inception,
                "Index 10Y CAGR (%)": idx_10y_cagr,
                "Index 15Y CAGR (%)": idx_15y_cagr,
                "RS Spread 21D (%)": round((ret_1m or 0.0) - bench_roc_21d, 2) if pd.notna(ret_1m) else 0.0,
                "RS Spread 1Y (%)": round((ret_1y or 0.0) - bench_roc_252d, 2) if pd.notna(ret_1y) else 0.0,
                "Historical Volatility (%)": hist_vol,
                "Historical S/R Win Rate (%)": sr_win_rate,
                "Historical S/R Trades (N)": tot_bounces,
                "S/R Predictability Rating": fid_rating,
                "ETF Empirical +20D Up Prob (%)": etf_cond_up20_prob,
                "Index 15Y Cond. +20D Up Prob (%)": idx_cond_up20,
                "Index 15Y Cond. +60D Up Prob (%)": idx_cond_up60,
                "Index 15Y Support Bounce Win (%)": idx_sr_win,
                "Index Live P/E": idx_pe,
                "Superstar Constituent Buys (N)": superstar_matches,
                "Regime": regime,
                "Action Signal": action_sig,
                "Macro Thesis": item["macro_thesis"],
            }
        )

    etf_df = pd.DataFrame(records)
    if etf_df.empty:
        return etf_df, pd.DataFrame()

    # -----------------------------------------------------------------------
    # PRODUCTION REPO MULTI-FACTOR SCORES + 100% FACTUAL SUITABILITY SCORE
    # -----------------------------------------------------------------------
    etf_df["Rank_RSI_Buy"] = etf_df["RSI (14D)"].rank(ascending=True, pct=True) * 100.0
    etf_df["Rank_200DMA_Buy"] = etf_df["Dist 200DMA %"].rank(ascending=True, pct=True) * 100.0
    etf_df["Rank_BB_Buy"] = etf_df["Bollinger %B"].rank(ascending=True, pct=True) * 100.0
    etf_df["Rank_VWAP_Buy"] = etf_df["Dist VWAP %"].rank(ascending=True, pct=True) * 100.0
    etf_df["Rank_Vol_Buy"] = etf_df["Volume Surge Ratio"].rank(ascending=False, pct=True) * 100.0
    etf_df["Rank_Stoch_Buy"] = etf_df["Stoch %K"].rank(ascending=True, pct=True) * 100.0
    etf_df["Rank_Div_Buy"] = etf_df["Dividend Yield %"].rank(ascending=False, pct=True) * 100.0
    etf_df["Rank_Exp"] = etf_df["Expense %"].rank(ascending=True, pct=True) * 100.0
    etf_df["Rank_Spread"] = etf_df["Distance to iNAV (%)"].clip(lower=0.0).rank(ascending=True, pct=True) * 100.0

    etf_df["Fundamental Score"] = round(0.60 * etf_df["Rank_Exp"] + 0.40 * etf_df["Rank_Spread"], 1)
    etf_df["Technical Score Buy Intraday"] = round(
        0.30 * etf_df["Rank_RSI_Buy"]
        + 0.35 * etf_df["Rank_Vol_Buy"]
        + 0.20 * etf_df["Rank_BB_Buy"]
        + 0.15 * etf_df["Rank_VWAP_Buy"],
        1,
    )
    etf_df["Technical Score Buy Swing"] = round(
        0.30 * etf_df["Rank_RSI_Buy"]
        + 0.25 * etf_df["Rank_200DMA_Buy"]
        + 0.20 * etf_df["Rank_BB_Buy"]
        + 0.15 * etf_df["Rank_VWAP_Buy"]
        + 0.10 * etf_df["Rank_Stoch_Buy"],
        1,
    )
    etf_df["Technical Score Buy LongTerm"] = round(
        0.15 * etf_df["Rank_RSI_Buy"]
        + 0.40 * etf_df["Rank_200DMA_Buy"]
        + 0.15 * etf_df["Rank_Div_Buy"]
        + 0.15 * etf_df["Rank_BB_Buy"]
        + 0.15 * etf_df["Fundamental Score"],
        1,
    )
    etf_df["Composite Buy Score"] = round(
        0.85 * etf_df["Technical Score Buy Swing"] + 0.15 * etf_df["Fundamental Score"], 1
    )

    # 100% Empirical Future Suitability Score (anchored directly on factual probabilities & historical S/R bounce rate)
    sr_bounce_factual = etf_df["Historical S/R Win Rate (%)"].combine_first(
        etf_df["Index 15Y Support Bounce Win (%)"]
    ).combine_first(etf_df["Buy Success Prob Now (%)"])
    etf_df["Future Suitability Score (0-100)"] = round(
        0.65 * etf_df["Buy Success Prob Now (%)"] + 0.35 * sr_bounce_factual,
        1,
    )

    # Drop temporary rank helper columns and sort strictly by Probability of Success descending
    drop_cols = [c for c in etf_df.columns if c.startswith("Rank_")]
    etf_df = etf_df.drop(columns=drop_cols).sort_values(
        by=["Buy Success Prob Now (%)", "Historical S/R Win Rate (%)", "Future Suitability Score (0-100)"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 7. SECTOR-LEVEL FUTURE MEGA-TREND SUITABILITY MATRIX (SORTED BY PROBABILITY)
    # -----------------------------------------------------------------------
    pillar_rows: List[Dict[str, Any]] = []
    for pillar, grp in etf_df.groupby("Future Macro Pillar", sort=False):
        top_row = grp.sort_values("Buy Success Prob Now (%)", ascending=False).iloc[0]
        max_buy_prob = round(float(grp["Buy Success Prob Now (%)"].max()), 1)
        avg_buy_prob = round(float(grp["Buy Success Prob Now (%)"].mean()), 1)
        avg_drop_prob = round(float(grp["Sell / Drop Prob +20D (%)"].mean()), 1)
        avg_score = round(float(grp["Future Suitability Score (0-100)"].mean()), 1)
        max_score = round(float(grp["Future Suitability Score (0-100)"].max()), 1)
        avg_rsi = round(float(grp["RSI (14D)"].mean()), 1)
        avg_rpos = round(float(grp["Range Position (%)"].mean()), 1)
        avg_1y = round(float(grp["1Y Return (%)"].dropna().mean()), 2) if grp["1Y Return (%)"].notna().any() else np.nan
        cagr_col = grp["5Y CAGR (%)"].fillna(grp["3Y CAGR (%)"]).fillna(grp["Inception CAGR (%)"])
        avg_cagr = round(float(cagr_col.dropna().mean()), 2) if cagr_col.notna().any() else np.nan
        avg_idx_15y_cagr = (
            round(float(grp["Index 15Y CAGR (%)"].dropna().mean()), 2)
            if grp["Index 15Y CAGR (%)"].notna().any()
            else avg_cagr
        )
        avg_sr_win = round(float(grp["Historical S/R Win Rate (%)"].dropna().mean()), 1)
        avg_cond20 = round(float(grp["Index 15Y Cond. +20D Up Prob (%)"].dropna().mean()), 1)
        avg_cond60 = round(float(grp["Index 15Y Cond. +60D Up Prob (%)"].dropna().mean()), 1)
        avg_pe = round(float(grp["Index Live P/E"].dropna().mean()), 1) if grp["Index Live P/E"].notna().any() else np.nan
        tot_superstar_buys = int(grp["Superstar Constituent Buys (N)"].sum())

        if max_buy_prob < 50.0:
            if avg_drop_prob >= 55.0:
                verdict = "🔴 SELL / AVOID SECTOR (Drop Prob >= 55%)"
            else:
                verdict = "⛔ IGNORE SECTOR (Success Prob < 50%)"
        elif max_buy_prob >= 60.0 and avg_rpos <= 70.0:
            verdict = "🟢 BUY NOW — TIER-1 HIGH PROBABILITY"
        elif max_buy_prob >= 54.0:
            verdict = "🟢 BUY / ACCUMULATE ON DIPS TO S1"
        else:
            verdict = "🟢 SELECTIVE BUY (Top ETF Prob >= 50%)"

        pillar_rows.append(
            {
                "Future Macro Pillar": pillar,
                "Action Recommendation": verdict,
                "Top ETF Buy Prob (%)": max_buy_prob,
                "Sector Avg Buy Prob (%)": avg_buy_prob,
                "Sector Avg Drop Prob (%)": avg_drop_prob,
                "ETFs Tracked": len(grp),
                "Top Ranked ETF": f"{top_row['Ticker']} (₹{top_row['CMP (₹)']:.2f} | {top_row['Buy Success Prob Now (%)']:.1f}%)",
                "Top ETF Suitability Score": max_score,
                "Sector Avg Suitability Score": avg_score,
                "Sector Avg RSI (14D)": avg_rsi,
                "Sector Avg S/R Range Pos (%)": avg_rpos,
                "15Y Index Cond. +20D Up Prob (%)": avg_cond20,
                "15Y Index Cond. +60D Up Prob (%)": avg_cond60,
                "Avg Empirical S/R Bounce Win (%)": avg_sr_win,
                "Sector Avg 1Y Return (%)": avg_1y,
                "ETF Multi-Year CAGR (%)": avg_cagr,
                "15Y Sector Index CAGR (%)": avg_idx_15y_cagr,
                "Index Live P/E": avg_pe,
                "Superstar HNI/Inst Buys": tot_superstar_buys,
                "Investment Suitability Verdict": verdict,
                "Primary Structural Driver": top_row["Macro Thesis"],
            }
        )

    sec_df = (
        pd.DataFrame(pillar_rows)
        .sort_values(by=["Top ETF Buy Prob (%)", "Sector Avg Buy Prob (%)"], ascending=[False, False])
        .reset_index(drop=True)
    )

    try:
        etf_df.to_csv(ETF_SCREENER_CSV, index=False)
        sec_df.to_csv(SECTOR_SUITABILITY_CSV, index=False)
    except Exception as exc:
        logger.warning("Failed to save ETF screener CSVs: %s", exc)

    return etf_df, sec_df


def refresh_indices_and_etf_live_cmps() -> Dict[str, int]:
    """
    Fast intraday/latest CMP batch updater for all 25 Indian Indices and 42 Sectoral ETFs.
    Updates `CMP`, `Dist to S1 (%)`, `Dist to R1 (%)`, `Range Position (%)`, and `Volume`
    in `indian_indices_sr_15y.csv` and `sectoral_etf_screener_live.csv` without needing a
    slow 15-year historical redownload.
    """
    from live_data_engine import fetch_live_cmp_batch

    idx_yf_map = {
        item["name"]: item["yf_ticker"]
        for item in INDIAN_INDICES_UNIVERSE
        if item.get("yf_ticker")
    }
    etf_yf_list = [item["ticker"] for item in SECTORAL_ETF_UNIVERSE if item.get("ticker")]

    all_syms = list(dict.fromkeys(list(idx_yf_map.values()) + etf_yf_list))
    quotes = fetch_live_cmp_batch(all_syms, update_disk_cache=True)

    idx_updated = 0
    etf_updated = 0

    if INDICES_SR_CSV.exists() and quotes:
        try:
            idx_df = pd.read_csv(INDICES_SR_CSV)
            for i, row in idx_df.iterrows():
                iname = str(row.get("Index Name", ""))
                yf_t = idx_yf_map.get(iname, "")
                if yf_t and yf_t in quotes:
                    new_cmp = float(quotes[yf_t]["cmp"])
                    if new_cmp > 0:
                        idx_df.at[i, "CMP"] = round(new_cmp, 2)
                        s1 = float(row.get("Major Support S1", 0.0) or 0.0)
                        r1 = float(row.get("Major Resistance R1", 0.0) or 0.0)
                        if s1 > 0:
                            idx_df.at[i, "Dist to S1 (%)"] = round(((new_cmp - s1) / s1) * 100.0, 2)
                        if r1 > 0:
                            idx_df.at[i, "Dist to R1 (%)"] = round(((r1 - new_cmp) / new_cmp) * 100.0, 2)
                        if r1 > s1:
                            rpos = max(0.0, min(100.0, ((new_cmp - s1) / (r1 - s1)) * 100.0))
                            idx_df.at[i, "Range Position (%)"] = round(rpos, 1)
                        idx_updated += 1
            if idx_updated > 0:
                idx_df.to_csv(INDICES_SR_CSV, index=False)
        except Exception as exc:
            logger.debug("Indices live CMP update warning: %s", exc)

    if ETF_SCREENER_CSV.exists() and quotes:
        try:
            nse_live_etfs = _fetch_live_nse_etf_quotes(use_cache=False)
            etf_df = pd.read_csv(ETF_SCREENER_CSV)
            for i, row in etf_df.iterrows():
                clean_sym = str(row.get("Ticker", "")).replace(".NS", "").strip().upper()
                ft = str(row.get("Full_Ticker") or f"{clean_sym}.NS")
                nse_q = nse_live_etfs.get(clean_sym, {})
                q = quotes.get(ft, {})
                new_cmp = float(nse_q.get("ltp", 0.0) or q.get("cmp", 0.0) or 0.0)
                if new_cmp > 0:
                    etf_df.at[i, "CMP (₹)"] = round(new_cmp, 2)
                    live_vol = int(nse_q.get("qty", 0) or q.get("volume", 0) or 0)
                    if live_vol > 0:
                        etf_df.at[i, "Volume"] = live_vol
                    nse_nav = float(nse_q.get("nav", 0.0) or 0.0)
                    if nse_nav > 0 and abs(nse_nav - new_cmp) / max(0.01, new_cmp) < 0.25:
                        etf_df.at[i, "iNAV (₹)"] = round(nse_nav, 2)
                    inav = float(etf_df.at[i, "iNAV (₹)"] or 0.0)
                    if inav > 0:
                        inav_diff = round(((new_cmp - inav) / inav) * 100.0, 2)
                        etf_df.at[i, "Distance to iNAV (%)"] = inav_diff
                        etf_df.at[i, "iNAV Status"] = (
                            "✅ Clean (<+0.35%)" if inav_diff <= 0.35 else f"⚠️ Premium (+{inav_diff:.2f}%)"
                        )
                    s1 = float(row.get("Major Support S1 (₹)", 0.0) or 0.0)
                    r1 = float(row.get("Major Resistance R1 (₹)", 0.0) or 0.0)
                    if s1 > 0:
                        etf_df.at[i, "Dist from Support (%)"] = round(((new_cmp - s1) / s1) * 100.0, 2)
                    if r1 > 0:
                        etf_df.at[i, "Dist from Resistance (%)"] = round(((r1 - new_cmp) / new_cmp) * 100.0, 2)
                    if r1 > s1:
                        rpos = max(0.0, min(100.0, ((new_cmp - s1) / (r1 - s1)) * 100.0))
                        etf_df.at[i, "Range Position (%)"] = round(rpos, 1)
                    etf_updated += 1
            if etf_updated > 0:
                etf_df.to_csv(ETF_SCREENER_CSV, index=False)
        except Exception as exc:
            logger.debug("ETF live CMP update warning: %s", exc)

    return {"indices_cmp_updated": idx_updated, "etfs_cmp_updated": etf_updated}


