"""
🚀 Indian Stock Multibagger Monitor — Nifty 500 Edition
========================================================
Fetches all Nifty 500 stocks from NSE, scores them on multibagger
criteria, and displays results on an interactive Streamlit dashboard.

Install dependencies:
    pip install streamlit yfinance pandas plotly requests tqdm

Run:
    streamlit run multibagger_dashboard.py

How data loading works:
  - First run: Downloads Nifty 500 list from NSE, then fetches
    fundamentals in batches of 50. Results saved to nifty500_cache.csv.
  - Subsequent runs: Loads from cache instantly. Refresh when needed.
  - Cache TTL: Configurable (default 24 hours).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import time
from io import StringIO
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Nifty 500 Multibagger Monitor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CACHE_FILE      = "nifty500_cache.csv"
SYMBOLS_FILE    = "nifty500_yahoo_symbols.csv"
CACHE_TTL_HOURS = 24
BATCH_SIZE      = 50   # stocks per yfinance batch download
NSE_URL         = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border-radius: 12px;
        padding: 14px 18px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 10px;
    }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 10px 16px;
        border: 1px solid #2d3452;
    }
    .stProgress > div > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEP 1 — LOAD / DOWNLOAD NIFTY 500 SYMBOL LIST
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_nifty500_symbols() -> pd.DataFrame:
    """
    Returns DataFrame with columns: Company Name, NSE Symbol, Yahoo Symbol, Industry, ISIN
    Tries: (1) local symbols CSV, (2) NSE live download.
    """
    # Use pre-fetched symbols file if available
    if os.path.exists(SYMBOLS_FILE):
        df = pd.read_csv(SYMBOLS_FILE)
        if {"Company Name", "Yahoo Symbol"}.issubset(df.columns):
            return df

    # Download fresh from NSE
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        time.sleep(0.5)
        r = session.get(NSE_URL, headers=NSE_HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = df.columns.str.strip()

        sym_col  = next((c for c in df.columns if "symbol"  in c.lower()), None)
        name_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)
        ind_col  = next((c for c in df.columns if "industry" in c.lower()), None)
        isin_col = next((c for c in df.columns if "isin"    in c.lower()), None)

        records = []
        for _, row in df.iterrows():
            sym = str(row[sym_col]).strip().upper()
            records.append({
                "Company Name": row[name_col].strip() if name_col else sym,
                "NSE Symbol":   sym,
                "Yahoo Symbol": f"{sym}.NS",
                "Industry":     row[ind_col].strip()  if ind_col  else "",
                "ISIN":         row[isin_col].strip() if isin_col else "",
            })

        result = pd.DataFrame(records)
        result.to_csv(SYMBOLS_FILE, index=False)
        return result

    except Exception as e:
        st.error(f"❌ Could not download Nifty 500 list: {e}\n\nPlease place `nifty500_yahoo_symbols.csv` in the same folder.")
        st.stop()


# ─────────────────────────────────────────────
# STEP 2 — BATCH FETCH FUNDAMENTALS
# ─────────────────────────────────────────────


def _calc_peg(info: dict):
    """
    Calculate PEG ratio from available data.
    Priority:
      1. yfinance pegRatio (when available)
      2. trailingPE / earningsGrowth (as %)
      3. trailingPE / revenueGrowth  (fallback)
    Returns None if cannot be calculated or result is negative/implausible.
    """
    # Use yfinance value if present and sensible
    peg_yf = info.get("pegRatio")
    if peg_yf is not None and 0 < peg_yf < 50:
        return peg_yf

    pe = info.get("trailingPE")
    if pe is None or pe <= 0:
        return None

    # Prefer earnings growth, fall back to revenue growth
    growth = info.get("earningsGrowth") or info.get("revenueGrowth")
    if growth is None or growth <= 0:
        return None

    # growth is a decimal (e.g. 0.25 = 25%), PEG = PE / growth_pct
    peg = pe / (growth * 100)
    return round(peg, 2) if 0 < peg < 50 else None


def fetch_batch_fundamentals(tickers: list) -> dict:
    """
    Uses yfinance.download() for price data (fast, batch).
    Uses yf.Tickers() for fundamentals (info).
    Returns dict keyed by ticker symbol.
    """
    results = {}

    # Batch download 1-year price history (very fast)
    ticker_str = " ".join(tickers)
    try:
        hist_all = yf.download(
            ticker_str,
            period="1y",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        hist_all = pd.DataFrame()

    # Fetch fundamentals via Tickers object
    try:
        tickers_obj = yf.Tickers(ticker_str)
    except Exception:
        tickers_obj = None

    for ticker in tickers:
        try:
            info = {}
            if tickers_obj:
                t = tickers_obj.tickers.get(ticker)
                if t:
                    info = t.info or {}

            # Extract price history for this ticker
            hist = pd.DataFrame()
            if not hist_all.empty:
                if len(tickers) == 1:
                    hist = hist_all
                elif ticker in hist_all.columns.get_level_values(0):
                    hist = hist_all[ticker].dropna(how="all")

            high_52w = hist["High"].max() if not hist.empty and "High" in hist.columns else None
            low_52w  = hist["Low"].min()  if not hist.empty and "Low"  in hist.columns else None
            current  = info.get("currentPrice") or info.get("regularMarketPrice")

            if current is None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                current = float(closes.iloc[-1]) if len(closes) else None

            pct_from_high = ((current - high_52w) / high_52w * 100) if (current and high_52w) else None
            pct_from_low  = ((current - low_52w)  / low_52w  * 100) if (current and low_52w)  else None

            results[ticker] = {
                "ticker":        ticker,
                "name":          info.get("longName", ticker),
                "sector":        info.get("sector", ""),
                "industry":      info.get("industry", ""),
                "current_price": current,
                "market_cap":    info.get("marketCap"),
                "52w_high":      high_52w,
                "52w_low":       low_52w,
                "pct_from_high": pct_from_high,
                "pct_from_low":  pct_from_low,
                "pe":            info.get("trailingPE"),
                "fwd_pe":        info.get("forwardPE"),
                "peg":           _calc_peg(info),
                "pb":            info.get("priceToBook"),
                "ps":            info.get("priceToSalesTrailing12Months"),
                "rev_growth":    info.get("revenueGrowth"),
                "earn_growth":   info.get("earningsGrowth"),
                "roe":           info.get("returnOnEquity"),
                "roa":           info.get("returnOnAssets"),
                "debt_equity":   info.get("debtToEquity"),
                "free_cashflow": info.get("freeCashflow"),
                "inst_holding":  info.get("heldPercentInstitutions"),
                "insider_hold":  info.get("heldPercentInsiders"),
                "div_yield":     info.get("dividendYield"),
                "error":         None,
            }
        except Exception as e:
            results[ticker] = {"ticker": ticker, "error": str(e)}

    return results


# ─────────────────────────────────────────────
# STEP 3 — CACHE MANAGEMENT
# ─────────────────────────────────────────────

def cache_is_fresh() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS)


def load_cache() -> pd.DataFrame:
    return pd.read_csv(CACHE_FILE) if os.path.exists(CACHE_FILE) else pd.DataFrame()


def save_cache(df: pd.DataFrame):
    df.to_csv(CACHE_FILE, index=False)


def fetch_all_nifty500(symbols_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch fundamentals for all Nifty 500 stocks in batches.
    Shows a progress bar in Streamlit.
    """
    tickers = symbols_df["Yahoo Symbol"].tolist()
    total   = len(tickers)
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    st.info(f"📡 Fetching data for **{total} stocks** in {len(batches)} batches of {BATCH_SIZE}. This takes ~3–5 minutes and is cached for {CACHE_TTL_HOURS} hours.")
    progress_bar = st.progress(0, text="Starting...")
    status_text  = st.empty()

    all_results = {}
    for i, batch in enumerate(batches):
        status_text.markdown(f"⏳ Fetching batch **{i+1}/{len(batches)}** ({batch[0]} → {batch[-1]})")
        batch_results = fetch_batch_fundamentals(batch)
        all_results.update(batch_results)
        progress_bar.progress((i + 1) / len(batches), text=f"Batch {i+1}/{len(batches)} done")
        time.sleep(0.3)  # polite delay

    progress_bar.empty()
    status_text.empty()

    # Merge with symbols metadata
    rows = []
    sym_lookup = symbols_df.set_index("Yahoo Symbol").to_dict("index")
    for ticker, d in all_results.items():
        meta = sym_lookup.get(ticker, {})
        rows.append({
            "Company Name": meta.get("Company Name", d.get("name", ticker)),
            "NSE Symbol":   meta.get("NSE Symbol", ticker.replace(".NS", "")),
            "Yahoo Symbol": ticker,
            "Industry":     meta.get("Industry", d.get("industry", "")),
            "ISIN":         meta.get("ISIN", ""),
            **{k: v for k, v in d.items() if k not in ("ticker", "name", "industry", "error")},
            "fetch_error":  d.get("error"),
        })

    df = pd.DataFrame(rows)
    save_cache(df)
    st.success(f"✅ Data fetched and cached for {len(df)} stocks!")
    return df


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

def multibagger_score(row) -> tuple:
    score  = 0
    checks = []

    rg = row.get("rev_growth")
    if rg is not None and pd.notna(rg):
        if rg > 0.25:   score += 20; checks.append(("✅", f"Revenue growth: {rg*100:.1f}% (>25%)"))
        elif rg > 0.15: score += 12; checks.append(("🟡", f"Revenue growth: {rg*100:.1f}% (>15%)"))
        else:                         checks.append(("❌", f"Revenue growth: {rg*100:.1f}% (<15%)"))
    else:
        checks.append(("⚪", "Revenue growth: N/A"))

    eg = row.get("earn_growth")
    if eg is not None and pd.notna(eg):
        if eg > 0.25:   score += 20; checks.append(("✅", f"Earnings growth: {eg*100:.1f}% (>25%)"))
        elif eg > 0.15: score += 10; checks.append(("🟡", f"Earnings growth: {eg*100:.1f}% (>15%)"))
        else:                         checks.append(("❌", f"Earnings growth: {eg*100:.1f}% (<15%)"))
    else:
        checks.append(("⚪", "Earnings growth: N/A"))

    roe = row.get("roe")
    if roe is not None and pd.notna(roe):
        if roe > 0.20:   score += 15; checks.append(("✅", f"ROE: {roe*100:.1f}% (>20%)"))
        elif roe > 0.15: score += 8;  checks.append(("🟡", f"ROE: {roe*100:.1f}% (>15%)"))
        else:                          checks.append(("❌", f"ROE: {roe*100:.1f}% (<15%)"))
    else:
        checks.append(("⚪", "ROE: N/A"))

    de = row.get("debt_equity")
    if de is not None and pd.notna(de):
        if de < 0.3:  score += 15; checks.append(("✅", f"D/E: {de/100:.2f} (very low)"))
        elif de < 50: score += 8;  checks.append(("🟡", f"D/E: {de/100:.2f} (moderate)"))
        else:                       checks.append(("❌", f"D/E: {de/100:.2f} (high)"))
    else:
        checks.append(("⚪", "D/E: N/A"))

    peg = row.get("peg")
    if peg is not None and pd.notna(peg) and peg > 0:
        if peg < 1.0:   score += 15; checks.append(("✅", f"PEG: {peg:.2f} (<1 — undervalued)"))
        elif peg < 1.5: score += 8;  checks.append(("🟡", f"PEG: {peg:.2f} (<1.5 — fair value)"))
        else:                         checks.append(("❌", f"PEG: {peg:.2f} (>1.5 — expensive)"))
    else:
        checks.append(("⚪", "PEG: N/A"))

    ins = row.get("insider_hold")
    if ins is not None and pd.notna(ins):
        if ins > 0.50:   score += 15; checks.append(("✅", f"Insider/Promoter: {ins*100:.1f}% (>50%)"))
        elif ins > 0.35: score += 8;  checks.append(("🟡", f"Insider/Promoter: {ins*100:.1f}% (>35%)"))
        else:                          checks.append(("❌", f"Insider/Promoter: {ins*100:.1f}% (<35%)"))
    else:
        checks.append(("⚪", "Insider holding: N/A"))

    return min(score, 100), checks


def fmt_num(val, prefix="", suffix="", decimals=2):
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    try:
        val = float(val)
        if abs(val) >= 1e9:  return f"{prefix}{val/1e9:.{decimals}f}B{suffix}"
        if abs(val) >= 1e7:  return f"{prefix}{val/1e7:.{decimals}f}Cr{suffix}"
        if abs(val) >= 1e6:  return f"{prefix}{val/1e6:.{decimals}f}M{suffix}"
        return f"{prefix}{val:.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 Nifty 500 Monitor")

    st.divider()
    st.subheader("📦 Data")
    cache_fresh = cache_is_fresh()
    if cache_fresh:
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        st.success(f"✅ Cache fresh\n\n{cache_mtime.strftime('%d %b %Y, %H:%M')}")
    else:
        st.warning("⚠️ Cache stale or missing")

    if st.button("🔄 Refresh All Data (3–5 min)"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("🔍 Filters")
    min_score = st.slider("Min Multibagger Score", 0, 100, 40)
    max_pe    = st.slider("Max PE Ratio", 0, 200, 100)
    min_roe   = st.slider("Min ROE %", 0, 50, 0)
    max_de    = st.slider("Max Debt/Equity", 0.0, 5.0, 3.0, step=0.1)

    st.divider()
    st.subheader("⚙️ Chart Settings")
    show_chart_period = st.selectbox("Price Chart Period", ["1mo","3mo","6mo","1y","2y"], index=3)
    top_n = st.slider("Top N stocks in scorecard", 5, 50, 20)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
symbols_df = load_nifty500_symbols()

if cache_is_fresh():
    df_cache = load_cache()
else:
    df_cache = fetch_all_nifty500(symbols_df)

# Coerce numeric columns (CSV load reads everything as strings)
NUMERIC_COLS = [
    "current_price", "market_cap", "52w_high", "52w_low",
    "pct_from_high", "pct_from_low", "pe", "fwd_pe", "peg",
    "pb", "ps", "rev_growth", "earn_growth", "roe", "roa",
    "debt_equity", "free_cashflow", "inst_holding", "insider_hold",
    "div_yield", "score",
]
for col in NUMERIC_COLS:
    if col in df_cache.columns:
        df_cache[col] = pd.to_numeric(df_cache[col], errors="coerce")

# Compute scores
if not df_cache.empty and "score" not in df_cache.columns:
    df_cache["score"] = df_cache.apply(
        lambda row: multibagger_score(row.to_dict())[0], axis=1
    )
    save_cache(df_cache)

# ─────────────────────────────────────────────
# SECTOR FILTER (dynamic, after data loads)
# ─────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📋 Sector Filter")
    industries = sorted(df_cache["Industry"].dropna().unique().tolist()) if "Industry" in df_cache.columns else []
    selected_industries = st.multiselect(
        "Industry / Sector",
        options=industries,
        default=[],
        placeholder="All industries"
    )

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_cache.copy()
df = df[df["score"] >= min_score]

if "pe" in df.columns:
    df = df[df["pe"].isna() | (df["pe"] <= max_pe)]
if "roe" in df.columns:
    df = df[df["roe"].isna() | (df["roe"] >= min_roe / 100)]
if "debt_equity" in df.columns:
    df = df[df["debt_equity"].isna() | (df["debt_equity"] <= max_de * 100)]
if selected_industries:
    df = df[df["Industry"].isin(selected_industries)]

df = df.sort_values("score", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🚀 Nifty 500 Multibagger Monitor")
last_updated = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE)).strftime('%d %b %Y, %H:%M') if os.path.exists(CACHE_FILE) else "N/A"
st.caption(f"Showing **{len(df)}** stocks matching filters · {len(df_cache)} total · Last updated: {last_updated}")

# ─────────────────────────────────────────────
# QUICK STATS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Universe",      len(df_cache))
c2.metric("Passing Filters",     len(df))
c3.metric("Score ≥ 60 (Strong)", int((df["score"] >= 60).sum()) if not df.empty else 0)
c4.metric("Score 40–60 (Watch)", int(((df["score"] >= 40) & (df["score"] < 60)).sum()) if not df.empty else 0)
c5.metric("Avg Score",           f"{df['score'].mean():.1f}" if not df.empty else "N/A")

st.divider()

# ─────────────────────────────────────────────
# TOP SCORERS MINI-CARDS
# ─────────────────────────────────────────────
st.subheader(f"🏆 Top {min(top_n, len(df))} Multibagger Candidates")
top_df = df.head(top_n)
cols_per_row = 5

for row_start in range(0, len(top_df), cols_per_row):
    cols = st.columns(cols_per_row)
    for col_i, (_, row) in enumerate(top_df.iloc[row_start:row_start+cols_per_row].iterrows()):
        score = int(row["score"])
        price = row.get("current_price")
        pfh   = row.get("pct_from_high")
        color = "#00d4aa" if score >= 60 else ("#fbbf24" if score >= 40 else "#ff4b6e")
        price_str = f"₹{float(price):,.1f}" if price and pd.notna(price) else "N/A"
        pfh_str   = f"{float(pfh):.1f}% from 52W high" if pfh and pd.notna(pfh) else ""
        with cols[col_i]:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div style="font-size:12px;color:#9ca3af;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{row['Company Name']}</div>
                <div style="font-size:22px;font-weight:bold;color:{color}">{score}<span style="font-size:13px;color:#6b7280">/100</span></div>
                <div style="font-size:13px;color:#e5e7eb;">{price_str}</div>
                <div style="font-size:11px;color:#6b7280;">{pfh_str}</div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Full Table",
    "🔬 Deep Dive",
    "📉 Price Charts",
    "✅ Criteria Checker",
    "📈 Analytics",
])

# ══════════════════════════════════════════════
# TAB 1 — FULL TABLE
# ══════════════════════════════════════════════
with tab1:
    st.caption(f"{len(df)} stocks · sorted by Multibagger Score")

    table_df = pd.DataFrame()
    table_df["Company"]          = df["Company Name"]
    table_df["Industry"]         = df["Industry"]
    table_df["Score"]            = df["score"].astype(int)
    table_df["Price (₹)"]        = df["current_price"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None)
    table_df["Mkt Cap"]          = df["market_cap"].apply(lambda x: fmt_num(x, prefix="₹"))
    table_df["PE"]               = df["pe"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None)
    table_df["Fwd PE"]           = df["fwd_pe"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None)
    table_df["PEG"]              = df["peg"].apply(lambda x: round(float(x), 2) if pd.notna(x) else None)
    table_df["ROE %"]            = df["roe"].apply(lambda x: round(float(x)*100, 1) if pd.notna(x) else None)
    table_df["Rev Gr%"]          = df["rev_growth"].apply(lambda x: round(float(x)*100, 1) if pd.notna(x) else None)
    table_df["EPS Gr%"]          = df["earn_growth"].apply(lambda x: round(float(x)*100, 1) if pd.notna(x) else None)
    table_df["D/E"]              = df["debt_equity"].apply(lambda x: round(float(x)/100, 2) if pd.notna(x) else None)
    table_df["Insider %"]        = df["insider_hold"].apply(lambda x: round(float(x)*100, 1) if pd.notna(x) else None)
    table_df["↓ From 52W High%"] = df["pct_from_high"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None)

    def score_color(val):
        if pd.isna(val): return ""
        if val >= 60: return "background-color:#0d3d2e;color:#00d4aa;font-weight:bold"
        if val >= 40: return "background-color:#3d2e00;color:#fbbf24;font-weight:bold"
        return "background-color:#3d0d15;color:#ff4b6e"

    def growth_color(val):
        if pd.isna(val): return ""
        return "color:#00d4aa" if val > 0 else "color:#ff4b6e"

    styled = (
        table_df.style
        .applymap(score_color,  subset=["Score"])
        .applymap(growth_color, subset=["Rev Gr%", "EPS Gr%", "↓ From 52W High%"])
        .format(na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=520)

    csv = table_df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "nifty500_multibagger_screener.csv", "text/csv")

# ══════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════
with tab2:
    stock_names = df["Company Name"].tolist()
    if not stock_names:
        st.warning("No stocks match current filters.")
    else:
        selected_name = st.selectbox("Select a stock", stock_names)
        row = df[df["Company Name"] == selected_name].iloc[0].to_dict()
        score, checks = multibagger_score(row)
        color = "#00d4aa" if score >= 60 else ("#fbbf24" if score >= 40 else "#ff4b6e")

        c1, c2, c3, c4, c5 = st.columns(5)
        price = row.get("current_price")
        c1.metric("Current Price",   f"₹{float(price):,.1f}" if price and pd.notna(price) else "N/A")
        c2.metric("Score",           f"{score}/100")
        c3.metric("Market Cap",      fmt_num(row.get("market_cap"), prefix="₹"))
        c4.metric("Industry",        row.get("Industry", "N/A"))
        c5.metric("NSE Symbol",      row.get("NSE Symbol", "N/A"))

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💰 Valuation")
            pe  = row.get('pe');     st.markdown(f"- **PE (Trailing):** {float(pe):.1f}"  if pe  and pd.notna(pe)  else "- **PE:** N/A")
            fpe = row.get('fwd_pe'); st.markdown(f"- **PE (Forward):** {float(fpe):.1f}"  if fpe and pd.notna(fpe) else "- **Fwd PE:** N/A")
            pg  = row.get('peg');    st.markdown(f"- **PEG Ratio:** {float(pg):.2f}"       if pg  and pd.notna(pg)  else "- **PEG:** N/A")
            pb  = row.get('pb');     st.markdown(f"- **Price/Book:** {float(pb):.2f}"      if pb  and pd.notna(pb)  else "- **P/B:** N/A")
            ps  = row.get('ps');     st.markdown(f"- **Price/Sales:** {float(ps):.2f}"     if ps  and pd.notna(ps)  else "- **P/S:** N/A")

            st.markdown("#### 📈 Growth")
            rg = row.get('rev_growth')
            eg = row.get('earn_growth')
            st.markdown(f"- **Revenue Growth:** {'🟢' if rg and rg>0.15 else '🔴'} {float(rg)*100:.1f}%"  if rg and pd.notna(rg) else "- **Revenue Growth:** N/A")
            st.markdown(f"- **Earnings Growth:** {'🟢' if eg and eg>0.20 else '🔴'} {float(eg)*100:.1f}%" if eg and pd.notna(eg) else "- **Earnings Growth:** N/A")

            st.markdown("#### 📊 52-Week Range")
            h52 = row.get('52w_high'); st.markdown(f"- **52W High:** ₹{float(h52):,.1f}" if h52 and pd.notna(h52) else "- **52W High:** N/A")
            l52 = row.get('52w_low');  st.markdown(f"- **52W Low:** ₹{float(l52):,.1f}"  if l52 and pd.notna(l52) else "- **52W Low:** N/A")
            pfh = row.get('pct_from_high'); st.markdown(f"- **From 52W High:** {float(pfh):.1f}%" if pfh and pd.notna(pfh) else "")
            pfl = row.get('pct_from_low');  st.markdown(f"- **From 52W Low:** +{float(pfl):.1f}%" if pfl and pd.notna(pfl) else "")

        with col2:
            st.markdown("#### 🏆 Quality")
            roe = row.get('roe'); st.markdown(f"- **ROE:** {'🟢' if roe and roe>0.18 else '🔴'} {float(roe)*100:.1f}%" if roe and pd.notna(roe) else "- **ROE:** N/A")
            roa = row.get('roa'); st.markdown(f"- **ROA:** {float(roa)*100:.1f}%"  if roa and pd.notna(roa) else "- **ROA:** N/A")
            de  = row.get('debt_equity'); st.markdown(f"- **D/E:** {'🟢' if de and de<50 else '🔴'} {float(de)/100:.2f}" if de and pd.notna(de) else "- **D/E:** N/A")
            fcf = row.get('free_cashflow'); st.markdown(f"- **Free Cash Flow:** {fmt_num(fcf, prefix='₹')}")

            st.markdown("#### 👥 Ownership")
            ins = row.get('insider_hold'); st.markdown(f"- **Promoter/Insider:** {'🟢' if ins and ins>0.50 else '🔴'} {float(ins)*100:.1f}%" if ins and pd.notna(ins) else "- **Insider %:** N/A")
            ist = row.get('inst_holding'); st.markdown(f"- **Institutional:** {float(ist)*100:.1f}%" if ist and pd.notna(ist) else "- **Institutional:** N/A")

        st.divider()
        st.markdown("#### 🎯 Multibagger Criteria Check")
        for icon, text in checks:
            st.markdown(f"{icon} &nbsp; {text}", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Multibagger Score", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40],  "color": "#3d0d15"},
                    {"range": [40, 60], "color": "#3d2e00"},
                    {"range": [60, 100],"color": "#0d3d2e"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 60}
            }
        ))
        fig_gauge.update_layout(height=260, paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — PRICE CHARTS
# ══════════════════════════════════════════════
with tab3:
    top_chart_names = df["Company Name"].head(20).tolist()
    chart_selected  = st.multiselect(
        "Select stocks to compare (max 10 recommended)",
        df["Company Name"].tolist(),
        default=top_chart_names[:5]
    )

    if chart_selected:
        ticker_map = df[df["Company Name"].isin(chart_selected)].set_index("Company Name")["Yahoo Symbol"].to_dict()

        fig = go.Figure()
        for name in chart_selected:
            ticker = ticker_map.get(name)
            if not ticker:
                continue
            try:
                h = yf.Ticker(ticker).history(period=show_chart_period)
                if not h.empty:
                    norm = (h["Close"] / h["Close"].iloc[0]) * 100
                    fig.add_trace(go.Scatter(x=h.index, y=norm, name=name, mode="lines", line={"width": 2}))
            except Exception:
                pass

        fig.update_layout(
            title=f"Normalized Price Performance (Base=100) — {show_chart_period}",
            xaxis_title="Date", yaxis_title="Indexed Return",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            legend={"bgcolor": "#1e2130"}, height=480, hovermode="x unified",
        )
        fig.update_xaxes(gridcolor="#2d3452")
        fig.update_yaxes(gridcolor="#2d3452")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Candlestick Chart")
        candle_name   = st.selectbox("Select stock for candlestick", chart_selected, key="candle")
        candle_ticker = ticker_map.get(candle_name)

        if candle_ticker:
            try:
                h = yf.Ticker(candle_ticker).history(period=show_chart_period)
                if not h.empty:
                    fig2 = go.Figure(go.Candlestick(
                        x=h.index,
                        open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                        name=candle_name,
                        increasing_line_color="#00d4aa",
                        decreasing_line_color="#ff4b6e",
                    ))
                    fig2.add_trace(go.Bar(
                        x=h.index, y=h["Volume"],
                        name="Volume", yaxis="y2",
                        marker_color="rgba(79,142,247,0.2)",
                    ))
                    fig2.update_layout(
                        title=f"{candle_name} — {show_chart_period}",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                        height=500, xaxis_rangeslider_visible=False,
                        yaxis2={"overlaying": "y", "side": "right", "showgrid": False},
                    )
                    fig2.update_xaxes(gridcolor="#2d3452")
                    fig2.update_yaxes(gridcolor="#2d3452")
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load chart: {e}")

# ══════════════════════════════════════════════
# TAB 4 — CRITERIA CHECKER
# ══════════════════════════════════════════════
with tab4:
    st.markdown(f"### 🔬 Criteria Breakdown — Top {min(50, len(df))} stocks")
    for _, row in df.head(50).iterrows():
        score, checks = multibagger_score(row.to_dict())
        with st.expander(f"{row['Company Name']}  ·  Score: {score}/100  ·  {row.get('Industry','')}"):
            for icon, text in checks:
                st.markdown(f"{icon} &nbsp; {text}", unsafe_allow_html=True)
            st.caption(f"Ticker: {row.get('Yahoo Symbol','')} | NSE: {row.get('NSE Symbol','')}")

# ══════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ══════════════════════════════════════════════
with tab5:
    if df.empty:
        st.warning("No data to show analytics.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = go.Figure(go.Histogram(
                x=df_cache["score"], nbinsx=20,
                marker_color="#4f8ef7", opacity=0.8,
            ))
            fig_hist.update_layout(
                title="Score Distribution — All Nifty 500",
                xaxis_title="Multibagger Score", yaxis_title="# Stocks",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                height=350, showlegend=False,
            )
            fig_hist.add_vline(x=60, line_dash="dash", line_color="#00d4aa", annotation_text="Strong (60+)")
            fig_hist.add_vline(x=40, line_dash="dash", line_color="#fbbf24", annotation_text="Watch (40+)")
            fig_hist.update_xaxes(gridcolor="#2d3452")
            fig_hist.update_yaxes(gridcolor="#2d3452")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            ind_counts = df["Industry"].value_counts().head(15)
            fig_bar = go.Figure(go.Bar(
                x=ind_counts.values, y=ind_counts.index,
                orientation="h", marker_color="#4f8ef7",
            ))
            fig_bar.update_layout(
                title="Top Industries (Filtered Stocks)",
                xaxis_title="Count",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                height=350, yaxis={"autorange": "reversed"},
            )
            fig_bar.update_xaxes(gridcolor="#2d3452")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ROE vs Revenue Growth scatter
        st.subheader("📍 ROE vs Revenue Growth (bubble size = market cap)")
        scatter_df = df[df["roe"].notna() & df["rev_growth"].notna()].copy()

        fig_sc = go.Figure()
        for _, r in scatter_df.iterrows():
            s = int(r["score"])
            c = "#00d4aa" if s >= 60 else ("#fbbf24" if s >= 40 else "#4f8ef7")
            mcap_size = float(r["market_cap"]) / 1e9 if pd.notna(r.get("market_cap")) else 1
            fig_sc.add_trace(go.Scatter(
                x=[float(r["rev_growth"]) * 100],
                y=[float(r["roe"]) * 100],
                mode="markers",
                marker=dict(
                    size=max(6, min(30, mcap_size * 0.5)),
                    color=c, opacity=0.7,
                    line=dict(color="white", width=0.5)
                ),
                name=r["Company Name"],
                text=f"{r['Company Name']}<br>Score: {s}<br>Rev Gr: {float(r['rev_growth'])*100:.1f}%<br>ROE: {float(r['roe'])*100:.1f}%",
                hoverinfo="text",
                showlegend=False,
            ))

        fig_sc.update_layout(
            xaxis_title="Revenue Growth %", yaxis_title="ROE %",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            height=500,
        )
        fig_sc.add_hline(y=18, line_dash="dash", line_color="#fbbf24", annotation_text="ROE 18%")
        fig_sc.add_vline(x=15, line_dash="dash", line_color="#fbbf24", annotation_text="RevGr 15%")
        fig_sc.update_xaxes(gridcolor="#2d3452")
        fig_sc.update_yaxes(gridcolor="#2d3452")
        st.plotly_chart(fig_sc, use_container_width=True)

        # Top 20 horizontal bar
        st.subheader("🏆 Top 20 by Multibagger Score")
        top20 = df.head(20)
        bar_colors = ["#00d4aa" if s >= 60 else "#fbbf24" for s in top20["score"]]
        fig_top = go.Figure(go.Bar(
            x=top20["score"], y=top20["Company Name"],
            orientation="h", marker_color=bar_colors,
            text=top20["score"], textposition="outside",
        ))
        fig_top.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            height=580, xaxis_title="Score",
            yaxis={"autorange": "reversed"},
            margin=dict(l=220),
        )
        fig_top.update_xaxes(gridcolor="#2d3452")
        st.plotly_chart(fig_top, use_container_width=True)

st.divider()
st.caption("⚠️ For **research and educational purposes only**. Not SEBI-registered investment advice. Always do your own due diligence.")
