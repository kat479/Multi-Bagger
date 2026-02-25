# 🚀 Nifty 500 Multibagger Monitor — Metrics Reference

A plain-English guide to every metric shown in the dashboard, where the data comes from, whether it's annual or quarterly, what a good number looks like, and how it's used in scoring.

---

## 📊 How Data is Sourced

All data is pulled from **Yahoo Finance via the `yfinance` library**. Yahoo Finance aggregates data from financial data providers and company filings.

| Data Type | Source Period |
|-----------|--------------|
| Price data (current, 52W high/low) | Live / end-of-day |
| Valuation ratios (PE, PB, PS) | Trailing Twelve Months (TTM) |
| Growth metrics (revenue, earnings) | Year-over-Year (YoY), TTM vs prior year TTM |
| Profitability (ROE, ROA) | Trailing Twelve Months (TTM) |
| Debt/Equity | Most recent reported quarter |
| Ownership (insider, institutional) | Most recent filing |
| Free Cash Flow | Trailing Twelve Months (TTM) |

> **TTM (Trailing Twelve Months):** The most recent 12-month period, rolling. Not the same as the last financial year. For Indian companies, this may lag the actual latest quarter by 1–2 months depending on when results were filed.

---

## 🏷️ Identity & Classification

### Company Name
The full registered company name as listed on Yahoo Finance.

### NSE Symbol
The stock ticker on the National Stock Exchange of India. Example: `RELIANCE`, `HDFCBANK`.

### Yahoo Symbol
NSE symbol with `.NS` suffix for Yahoo Finance API. Example: `RELIANCE.NS`. Use `.BO` for BSE-listed equivalents.

### Industry
The industry classification as assigned by Yahoo Finance, sourced from the NSE constituent list. Examples: `Pharmaceuticals`, `Information Technology`, `Capital Goods`.

### ISIN
International Securities Identification Number — a unique 12-character code identifying the stock globally. Format: `INE` followed by 9 characters.

---

## 💰 Valuation Metrics

These tell you whether a stock is cheap or expensive relative to its earnings, assets, or sales.

---

### PE — Price to Earnings (Trailing)
**Period: Trailing Twelve Months (TTM / Annual)**

```
PE = Current Share Price ÷ Earnings Per Share (last 12 months)
```

The most widely used valuation metric. Tells you how many years of current earnings you're paying for.

| PE Range | Interpretation |
|----------|---------------|
| < 15 | Cheap (often value traps — investigate why) |
| 15–30 | Reasonable for moderate-growth companies |
| 30–60 | Acceptable only if growth > 20% |
| > 60 | Expensive — requires very high growth to justify |

**Limitation:** Backwards-looking. A company that just had a one-off profit bump will show a misleadingly low PE.

**Dashboard filter:** Sidebar slider lets you set a maximum PE.

---

### Fwd PE — Forward Price to Earnings
**Period: Next Twelve Months (NTM) — Analyst Estimate**

```
Fwd PE = Current Share Price ÷ Estimated EPS (next 12 months)
```

Uses analyst consensus earnings forecasts instead of historical earnings. More useful than trailing PE for fast-growing companies.

- **Fwd PE < Trailing PE** → earnings expected to grow (good)
- **Fwd PE > Trailing PE** → earnings expected to shrink (red flag)
- Often unavailable for smaller Indian stocks (analysts don't cover them)

---

### PEG — Price/Earnings to Growth
**Period: TTM PE ÷ TTM or Forward Earnings Growth Rate (Annual)**

```
PEG = PE Ratio ÷ Annual Earnings Growth Rate (%)
```

The single best quick valuation check for growth stocks. Adjusts PE for how fast the company is growing.

| PEG | Interpretation |
|-----|---------------|
| < 1.0 | Undervalued — paying less than growth justifies ✅ |
| 1.0–1.5 | Fair value 🟡 |
| > 1.5 | Expensive for the growth rate ❌ |
| > 2.5 | Significantly overvalued |

**Example:** Stock with PE 30, growing at 30% → PEG = 1.0 (fair). Stock with PE 60, growing at 15% → PEG = 4.0 (very expensive).

**How the dashboard calculates it:**
Since Yahoo Finance often doesn't return `pegRatio` for Indian stocks, the dashboard uses this fallback logic:
1. Yahoo's `pegRatio` — used if available and between 0–50
2. `trailingPE ÷ earningsGrowth%` — calculated manually
3. `trailingPE ÷ revenueGrowth%` — fallback if earnings growth unavailable

**Scoring weight:** 15 points (PEG < 1.0 = full score)

---

### PB — Price to Book
**Period: Most recent reported quarter**

```
PB = Current Share Price ÷ Book Value Per Share
```

Book value = Total Assets − Total Liabilities. Compares market price to what the company's assets are actually worth on paper.

| PB | Interpretation |
|----|---------------|
| < 1.0 | Trading below asset value (often distressed or cyclical) |
| 1–3 | Reasonable for asset-heavy businesses |
| 3–8 | Typical for quality compounders |
| > 10 | Asset-light businesses (tech, pharma) — use other metrics |

**Best used for:** Banks, NBFCs, manufacturing companies. Less meaningful for software or pharma companies with few tangible assets.

---

### PS — Price to Sales (Trailing)
**Period: Trailing Twelve Months (TTM / Annual)**

```
PS = Market Cap ÷ Total Revenue (last 12 months)
```

Useful when a company has negative earnings (startups, turnarounds). Compares price to how much revenue the business generates.

| PS | Interpretation |
|----|---------------|
| < 1 | Very cheap (check why — margin issues?) |
| 1–3 | Reasonable |
| 3–8 | Expensive; needs strong margins and growth |
| > 10 | Priced for perfection |

---

## 📈 Growth Metrics

These tell you how fast the business is actually expanding.

---

### Rev Gr% — Revenue Growth
**Period: Year-over-Year (YoY), TTM vs prior year TTM**

```
Revenue Growth = (TTM Revenue − Prior Year TTM Revenue) ÷ Prior Year TTM Revenue
```

Shows how fast the company's top line (total sales) is growing. The most honest growth number — harder to manipulate than earnings.

| Growth | Interpretation |
|--------|---------------|
| > 25% | Strong — potential multibagger territory ✅ |
| 15–25% | Good 🟡 |
| 5–15% | Moderate |
| < 5% | Slow / stagnating |
| Negative | Shrinking business ❌ |

**Dashboard display:** Shown as a percentage (e.g. `28.5%`). Green if positive, red if negative.

**Scoring weight:** 20 points (>25% = full score, >15% = partial)

---

### EPS Gr% — Earnings Per Share Growth
**Period: Year-over-Year (YoY), TTM vs prior year TTM**

```
Earnings Growth = (TTM Net Profit − Prior Year TTM Net Profit) ÷ Prior Year TTM Net Profit
```

How fast profits (bottom line) are growing. Should ideally grow faster than revenue — indicates improving margins.

| Growth | Interpretation |
|--------|---------------|
| > 25% | Excellent ✅ |
| 15–25% | Good 🟡 |
| < 15% | Below multibagger threshold ❌ |
| Negative | Earnings declining — avoid unless known one-off |

**Key insight:** If revenue is growing 20% but earnings are growing 40%, margins are expanding — a very bullish sign for multibaggers.

**Scoring weight:** 20 points (>25% = full score, >15% = partial)

---

## 🏆 Quality / Profitability Metrics

These tell you *how well* the business converts capital into profit.

---

### ROE % — Return on Equity
**Period: Trailing Twelve Months (TTM / Annual)**

```
ROE = Net Profit ÷ Shareholders' Equity × 100
```

The single most important quality metric. Measures how efficiently management uses shareholders' money to generate profit. A high, consistent ROE over many years is the hallmark of a compounding machine.

| ROE | Interpretation |
|-----|---------------|
| > 25% | Excellent — economic moat likely ✅ |
| 20–25% | Very good ✅ |
| 15–20% | Good 🟡 |
| 10–15% | Average |
| < 10% | Poor capital efficiency ❌ |

**Caveat:** Very high ROE (>50%) can sometimes be caused by high debt (leverage artificially inflates ROE). Always check alongside D/E ratio.

**Dashboard filter:** Sidebar slider lets you set a minimum ROE.

**Scoring weight:** 15 points (>20% = full score, >15% = partial)

---

### ROA — Return on Assets
**Period: Trailing Twelve Months (TTM / Annual)**

```
ROA = Net Profit ÷ Total Assets × 100
```

Similar to ROE but measures efficiency across *all* assets (including debt-funded ones). Less affected by leverage than ROE.

| ROA | Interpretation |
|-----|---------------|
| > 15% | Excellent |
| 8–15% | Good |
| 3–8% | Average |
| < 3% | Poor |

**Shown in:** Deep Dive tab only. Not part of scoring.

---

### Free Cash Flow (FCF)
**Period: Trailing Twelve Months (TTM / Annual)**

```
FCF = Operating Cash Flow − Capital Expenditure
```

The cash a business actually generates after maintaining/expanding its asset base. Often called the "real earnings" because it's harder to manipulate than reported net profit.

- **Positive FCF** → Business generates real cash, can fund growth or return capital
- **FCF ≈ Net Profit** → High earnings quality (cash matches reported profits)
- **FCF << Net Profit** → Working capital issues or aggressive accounting — investigate
- **Negative FCF** → Company is burning cash (may be fine if investing in growth)

**Display:** Shown in Crores (Cr) for Indian companies. E.g. `₹2,340Cr`.

**Shown in:** Deep Dive tab only. Not part of scoring (due to data gaps from yfinance).

---

## 🔢 Debt & Balance Sheet

---

### D/E — Debt to Equity Ratio
**Period: Most recent reported quarter**

```
D/E = Total Debt ÷ Shareholders' Equity
```

Measures financial risk. How much the company owes relative to what it owns. Lower is safer.

| D/E | Interpretation |
|-----|---------------|
| 0 | Debt-free ✅ |
| 0–0.5 | Very low debt ✅ |
| 0.5–1.0 | Moderate — manageable |
| 1.0–2.0 | High — monitor closely |
| > 2.0 | Very high — significant risk ❌ |

**Important note:** yfinance returns D/E multiplied by 100 for some Indian stocks. The dashboard divides by 100 to normalise. So a displayed value of `0.35` means D/E of 0.35x.

**Exception:** Banks and NBFCs naturally carry high D/E (they lend borrowed money). Use different benchmarks (Capital Adequacy Ratio) for financial companies.

**Dashboard filter:** Sidebar slider — Max Debt/Equity (0 to 5x).

**Scoring weight:** 15 points (D/E < 0.3 = full score, < 0.5 = partial)

---

## 👥 Ownership Metrics

---

### Insider % — Promoter / Insider Holding
**Period: Most recent quarterly regulatory filing**

```
Insider % = Shares held by promoters & insiders ÷ Total shares × 100
```

In Indian stocks, "insiders" = promoters (founding family/group). High promoter holding signals:
- Skin in the game — promoters profit only if stock price rises
- Alignment with minority shareholders
- Less susceptibility to hostile takeovers

| Promoter Holding | Interpretation |
|-----------------|---------------|
| > 60% | High conviction from founders ✅ |
| 50–60% | Good 🟡 |
| 35–50% | Moderate |
| < 35% | Low promoter conviction ❌ |

**Red flag:** If promoter holding is *decreasing* over quarters, investigate why. Promoters selling is a warning sign.

**Separate red flag (not in dashboard):** High promoter pledge % (shares pledged as collateral for loans). Check this manually on Screener.in.

**Scoring weight:** 15 points (>50% = full score, >35% = partial)

---

### Institutional % — Institutional Holding
**Period: Most recent quarterly filing**

```
Institutional % = Shares held by FIIs + DIIs ÷ Total shares × 100
```

Includes Foreign Institutional Investors (FIIs) and Domestic Institutional Investors (DIIs) like mutual funds, insurance companies.

| Interpretation | Detail |
|---------------|--------|
| Low institutional % | Under-researched — discovery hasn't happened yet (good for early entry) |
| Rising institutional % | Smart money is buying — validates thesis |
| Very high institutional % | Already well-known, may be priced in |

**Shown in:** Deep Dive tab only. Not part of scoring.

---

## 📊 Price & Technical Metrics

---

### Current Price
Live price fetched from Yahoo Finance (15-minute delayed for Indian markets during trading hours, real-time after close).

---

### Market Cap
**Period: Live / current**

```
Market Cap = Current Share Price × Total Shares Outstanding
```

Total market value of the company.

| Size | Category | Cap Range |
|------|----------|-----------|
| Large Cap | Nifty 100 | > ₹20,000 Cr |
| Mid Cap | Nifty Midcap | ₹5,000–20,000 Cr |
| Small Cap | Nifty Smallcap | ₹500–5,000 Cr |
| Micro Cap | Under the radar | < ₹500 Cr |

**Multibagger sweet spot:** ₹500–5,000 Cr (small/micro cap). Large caps rarely give 5–10x returns as the base is too high.

**Display:** In Crores (Cr) or Billions (B). E.g. `₹8,240Cr`.

---

### 52W High / 52W Low
**Period: Rolling 52 weeks of daily price data**

The highest and lowest prices the stock has traded at over the past 52 weeks (1 year). Calculated from yfinance daily price history.

---

### ↓ From 52W High %
**Period: Rolling 52 weeks**

```
% From High = (Current Price − 52W High) ÷ 52W High × 100
```

Always a negative number (stock is below its high). Shows how much the stock has corrected from its peak.

| % From High | Interpretation |
|-------------|---------------|
| 0 to -10% | Near highs — expensive entry zone |
| -10 to -25% | Moderate pullback |
| -25 to -40% | Significant correction — potential entry if fundamentals intact |
| < -40% | Deep correction — investigate if business is deteriorating or opportunity |

**Best use:** Combined with score. A stock scoring 70+ that is 35% off its high is potentially a great entry point.

**Display:** Green if near high, red if far below.

---

### From 52W Low %
**Period: Rolling 52 weeks**

```
% From Low = (Current Price − 52W Low) ÷ 52W Low × 100
```

Always positive. How much the stock has recovered from its 52-week bottom.

---

## 🎯 Multibagger Score (0–100)

A composite score calculated by the dashboard to rank stocks on their multibagger potential. **Higher = better.**

| Score | Signal | Colour |
|-------|--------|--------|
| 60–100 | Strong candidate | 🟢 Green |
| 40–59 | Watch list | 🟡 Yellow |
| 0–39 | Doesn't meet criteria | 🔴 Red |

### Scoring Breakdown

| Metric | Max Points | Full Score Threshold | Partial Score |
|--------|-----------|---------------------|---------------|
| Revenue Growth (TTM YoY) | 20 pts | > 25% | > 15% = 12 pts |
| Earnings Growth (TTM YoY) | 20 pts | > 25% | > 15% = 10 pts |
| ROE (TTM) | 15 pts | > 20% | > 15% = 8 pts |
| Debt/Equity (Latest Quarter) | 15 pts | < 0.3x | < 0.5x = 8 pts |
| PEG Ratio (TTM) | 15 pts | < 1.0 | < 1.5 = 8 pts |
| Insider/Promoter Holding | 15 pts | > 50% | > 35% = 8 pts |
| **Total** | **100 pts** | | |

**Important:** A score of N/A for any metric means that data was unavailable from Yahoo Finance — the metric simply doesn't contribute to the score. A stock with many N/A metrics may have an artificially low score even if fundamentally strong.

---

## 📉 Price Charts

### Normalized Performance Chart
All selected stocks are indexed to **100** at the start of the chosen period. Shows relative performance — which stock gave more return over the same time window regardless of absolute price level.

### Candlestick Chart
Standard OHLC (Open, High, Low, Close) chart.
- 🟢 Green candle = price closed higher than it opened
- 🔴 Red candle = price closed lower than it opened
- Volume bars shown on right axis (secondary Y axis)

---

## 🔍 Sidebar Filters

| Filter | What it does |
|--------|-------------|
| Min Multibagger Score | Only show stocks scoring above this threshold |
| Max PE Ratio | Exclude expensive stocks above this PE |
| Min ROE % | Only show stocks with at least this return on equity |
| Max Debt/Equity | Exclude highly leveraged companies |
| Industry / Sector | Filter to specific industries only |
| Price Chart Period | Time window for charts (1M / 3M / 6M / 1Y / 2Y) |
| Top N in Scorecard | How many stocks to show in the top candidates grid |

---

## ⚙️ Data & Caching

| Setting | Value | Where to change |
|---------|-------|----------------|
| Cache TTL | 24 hours | `CACHE_TTL_HOURS` in code |
| Batch size | 50 stocks per API call | `BATCH_SIZE` in code |
| Cache file | `nifty500_cache.csv` | `CACHE_FILE` in code |
| Symbols file | `nifty500_yahoo_symbols.csv` | `SYMBOLS_FILE` in code |

**First run:** Downloads Nifty 500 list from NSE → fetches fundamentals in batches of 50 → saves to `nifty500_cache.csv`. Takes 3–5 minutes.

**Subsequent runs:** Loads from `nifty500_cache.csv` instantly (< 2 seconds).

**Refresh:** Click "🔄 Refresh All Data" in sidebar to delete cache and re-fetch everything.

---

## ⚠️ Limitations & Known Issues

| Issue | Detail |
|-------|--------|
| yfinance data gaps | Yahoo Finance does not always populate all fields for Indian stocks. PEG, Forward PE, and FCF are most commonly missing. |
| PEG calculation | Manually calculated when Yahoo doesn't return it. Uses TTM earnings growth — not forward growth. May differ from analyst PEG estimates. |
| D/E normalisation | yfinance returns D/E in different scales for different stocks. The dashboard divides by 100 to normalise, but some values may still appear inconsistent. |
| TTM vs Annual | yfinance mixes TTM and annual data depending on the metric and stock. Growth metrics are generally TTM YoY. |
| Data freshness | Data is as fresh as when you last refreshed the cache. Cache is valid for 24 hours by default. |
| Indian market timing | Prices are 15-minute delayed during NSE trading hours (9:15 AM – 3:30 PM IST). Real-time after market close. |
| Market cap classification | May not perfectly align with NSE's official large/mid/small cap index classifications which are reviewed semi-annually. |

---

## 📚 Further Research

The dashboard is a **screening tool** — it narrows 500 stocks to a manageable shortlist. Before investing, always do deeper due diligence:

- **Screener.in** — 10-year financial history, promoter pledging, quarterly results
- **Tijori Finance** — Concall transcripts, segment-level data
- **Trendlyne** — Technical analysis + fundamentals combined
- **BSE/NSE filings** — Read the actual annual report (Chairman's letter + MD&A section)
- **NSE India** — Official shareholding patterns (quarterly)

---

*⚠️ This dashboard is for research and educational purposes only. Not SEBI-registered investment advice. Always consult a qualified financial advisor before making investment decisions.*
