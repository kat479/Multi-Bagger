"""
Nifty 500 → Yahoo Finance Symbol Fetcher
=========================================
Fetches the official Nifty 500 constituent list from NSE India,
converts every symbol to its Yahoo Finance format (appends .NS or .BO),
validates each symbol against yfinance, and saves the result to CSV.

Install dependencies:
    pip install requests pandas yfinance tqdm

Run:
    python fetch_nifty500_symbols.py

Output:
    nifty500_yahoo_symbols.csv
"""

import requests
import pandas as pd
import yfinance as yf
import time
import os
from io import StringIO
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# STEP 1: Download Nifty 500 constituent list from NSE
# ─────────────────────────────────────────────────────────

NSE_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

# Known special-case symbol mappings (NSE symbol → Yahoo suffix override)
# Some NSE symbols differ slightly on Yahoo Finance
SYMBOL_OVERRIDES = {
    "M&M":           "M%26M.NS",
    "M&MFIN":        "M%26MFIN.NS",
    "L&TFH":         "L%26TFH.NS",
    "BAJAJ-AUTO":    "BAJAJ-AUTO.NS",
    "NIFTY 50":      None,   # Index, skip
}


def download_nifty500() -> pd.DataFrame:
    """Download the Nifty 500 CSV from NSE and return as DataFrame."""
    print("📥 Downloading Nifty 500 list from NSE India...")
    try:
        session = requests.Session()
        # First hit the main page to get cookies
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=15)
        time.sleep(1)
        response = session.get(NSE_URL, headers=HEADERS, timeout=20)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        print(f"✅ Downloaded {len(df)} stocks from NSE.")
        return df
    except Exception as e:
        print(f"❌ Failed to download from NSE: {e}")
        print("💡 Trying fallback: loading from local file if available...")
        if os.path.exists("ind_nifty500list.csv"):
            df = pd.read_csv("ind_nifty500list.csv")
            print(f"✅ Loaded {len(df)} stocks from local file.")
            return df
        raise


# ─────────────────────────────────────────────────────────
# STEP 2: Convert NSE symbols to Yahoo Finance format
# ─────────────────────────────────────────────────────────

def to_yahoo_symbol(nse_symbol: str) -> str:
    """Convert an NSE symbol to Yahoo Finance ticker (appends .NS)."""
    nse_symbol = str(nse_symbol).strip().upper()

    # Check override table
    if nse_symbol in SYMBOL_OVERRIDES:
        return SYMBOL_OVERRIDES[nse_symbol]  # May be None (skip)

    # Yahoo Finance uses .NS for NSE-listed stocks
    return f"{nse_symbol}.NS"


# ─────────────────────────────────────────────────────────
# STEP 3: (Optional) Validate symbols against yfinance
# ─────────────────────────────────────────────────────────

def validate_symbol(yahoo_symbol: str) -> dict:
    """
    Quick validation: fetch basic info from yfinance.
    Returns dict with name, sector, current price.
    """
    try:
        ticker = yf.Ticker(yahoo_symbol)
        info = ticker.fast_info  # faster than .info
        price = getattr(info, "last_price", None)
        return {
            "valid":   price is not None and price > 0,
            "price":   round(price, 2) if price else None,
        }
    except Exception:
        return {"valid": False, "price": None}


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    # ── Download ──────────────────────────────────────────
    df_nse = download_nifty500()

    # Standardise column names (NSE CSV columns can vary slightly)
    df_nse.columns = df_nse.columns.str.strip()
    print(f"\n📋 Columns in NSE file: {list(df_nse.columns)}")

    # NSE CSV typically has: Company Name, Industry, Symbol, Series, ISIN Code
    symbol_col   = next((c for c in df_nse.columns if "symbol" in c.lower()), None)
    name_col     = next((c for c in df_nse.columns if "company" in c.lower() or "name" in c.lower()), None)
    industry_col = next((c for c in df_nse.columns if "industry" in c.lower() or "sector" in c.lower()), None)
    isin_col     = next((c for c in df_nse.columns if "isin" in c.lower()), None)

    if not symbol_col:
        raise ValueError("Could not find Symbol column in NSE CSV. Columns: " + str(list(df_nse.columns)))

    # ── Build base table ──────────────────────────────────
    records = []
    for _, row in df_nse.iterrows():
        nse_sym   = str(row[symbol_col]).strip().upper()
        yahoo_sym = to_yahoo_symbol(nse_sym)

        if yahoo_sym is None:
            continue  # Skip overridden/invalid entries

        records.append({
            "Company Name":   row[name_col].strip()     if name_col     else "",
            "NSE Symbol":     nse_sym,
            "Yahoo Symbol":   yahoo_sym,
            "BSE Symbol":     yahoo_sym.replace(".NS", ".BO"),  # BSE equivalent
            "Industry":       row[industry_col].strip() if industry_col else "",
            "ISIN":           row[isin_col].strip()     if isin_col     else "",
        })

    df_out = pd.DataFrame(records)
    print(f"\n🔁 Converted {len(df_out)} symbols to Yahoo Finance format.")

    # ── Optional: Validate symbols (adds ~5-10 min for 500 stocks) ───
    validate = input("\n❓ Validate all symbols against yfinance? This takes ~5-10 mins. (y/n): ").strip().lower()

    if validate == "y":
        print("\n🔍 Validating symbols... (this may take a while)\n")
        prices  = []
        valid   = []

        for sym in tqdm(df_out["Yahoo Symbol"], desc="Validating"):
            result = validate_symbol(sym)
            valid.append(result["valid"])
            prices.append(result["price"])
            time.sleep(0.1)  # Be gentle with the API

        df_out["Valid"]         = valid
        df_out["Current Price"] = prices

        valid_count   = sum(valid)
        invalid_count = len(valid) - valid_count
        print(f"\n✅ Valid symbols:   {valid_count}")
        print(f"❌ Invalid symbols: {invalid_count}")

        # Show invalid ones
        invalid_df = df_out[df_out["Valid"] == False]
        if not invalid_df.empty:
            print("\n⚠️  Symbols that failed validation:")
            print(invalid_df[["Company Name", "NSE Symbol", "Yahoo Symbol"]].to_string(index=False))
    else:
        print("⏭️  Skipping validation.")

    # ── Save to CSV ───────────────────────────────────────
    output_file = "nifty500_yahoo_symbols.csv"
    df_out.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    print(f"📊 Total stocks: {len(df_out)}")
    print("\n📌 Sample output:")
    print(df_out.head(10).to_string(index=False))

    # ── Summary by industry ───────────────────────────────
    if "Industry" in df_out.columns and df_out["Industry"].any():
        print("\n📈 Stocks by Industry (Top 15):")
        print(df_out["Industry"].value_counts().head(15).to_string())

    return df_out


# ─────────────────────────────────────────────────────────
# BONUS: Helper to load the saved CSV later
# ─────────────────────────────────────────────────────────

def load_nifty500_symbols(filepath="nifty500_yahoo_symbols.csv") -> dict:
    """
    Load saved Nifty 500 symbols as a dict:
        { "Company Name": "SYMBOL.NS", ... }

    Usage:
        from fetch_nifty500_symbols import load_nifty500_symbols
        watchlist = load_nifty500_symbols()
    """
    df = pd.read_csv(filepath)
    return dict(zip(df["Company Name"], df["Yahoo Symbol"]))


def load_by_industry(industry: str, filepath="nifty500_yahoo_symbols.csv") -> dict:
    """
    Load only stocks from a specific industry sector.

    Usage:
        it_stocks = load_by_industry("Information Technology")
    """
    df = pd.read_csv(filepath)
    filtered = df[df["Industry"].str.contains(industry, case=False, na=False)]
    return dict(zip(filtered["Company Name"], filtered["Yahoo Symbol"]))


if __name__ == "__main__":
    main()
