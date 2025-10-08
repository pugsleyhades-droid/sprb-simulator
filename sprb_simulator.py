### --- SECTION 1: Imports & Setup ---

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests, pytz, os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas_market_calendars as mcal

# optional advanced modules (guarded imports)
try:
    from pytrends.request import TrendReq
    from sklearn.preprocessing import MinMaxScaler
    from sentence_transformers import SentenceTransformer
except ImportError:
    TrendReq = None
    SentenceTransformer = None

# --- Streamlit page setup ---
st.set_page_config(page_title="SPRB Advanced Forecast Simulator", layout="wide")
st.title("🚀 SPRB Advanced Forecast Simulator")

st.markdown("""
This app models **multi-factor biotech stock behavior** using:
- Monte Carlo price paths  
- Multi-source sentiment  
- Peer correlations  
- (Later) event analog embeddings, fundamentals & risk weighting  
""")

st.success("✅ Section 1 (Imports & Setup) loaded.")

# ---------------------------------------------------------------------
### --- SECTION 2: Data Fetch, Multi-Source Sentiment & Volatility ---

st.header("Section 2 — Data & Sentiment")
ticker = st.text_input("Enter ticker to analyze", value="SPRB").upper().strip()

# Fetch historical market data and info
@st.cache_data(ttl=60*30)
def fetch_market_and_info(ticker_sym):
    try:
        tk = yf.Ticker(ticker_sym)
        hist = tk.history(period="180d", interval="1d")
        info = tk.info if hasattr(tk, "info") else {}
        return hist, info
    except Exception:
        return None, {}

hist, info = fetch_market_and_info(ticker)
if hist is None or hist.empty:
    st.error("Historical price data unavailable for ticker. Please check the ticker symbol.")
    st.stop()
else:
    st.success(f"Fetched historical data: {len(hist)} rows (last close ${hist['Close'].iloc[-1]:.2f})")
    latest_close = float(hist["Close"].iloc[-1])
    st.metric("Latest Close", f"${latest_close:.2f}")

# --- NewsAPI function ---
def fetch_newsapi_texts(query, days=7, page_size=30):
    try:
        api_key = st.secrets["newsapi"]["key"]
    except Exception:
        st.warning("NewsAPI key missing (.streamlit/secrets.toml) → news disabled.")
        return []
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from={(datetime.utcnow() - timedelta(days=days)).date()}"
        "&language=en&sortBy=publishedAt"
        f"&pageSize={page_size}&apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        articles = data.get("articles", [])
        texts = []
        for a in articles:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            texts.append(title + ". " + desc)
        return texts
    except Exception:
        return []

# --- Reddit fetch (simple) ---
def fetch_reddit_titles(keyword, limit=50):
    try:
        url = f"https://api.pushshift.io/reddit/search/submission/?q={keyword}&size={limit}"
        r = requests.get(url, timeout=6)
        data = r.json().get("data", [])
        titles = [d.get("title", "") for d in data]
        return titles
    except Exception:
        return []

# --- Google Trends ---
def google_trends_momentum(keyword, days=7):
    if TrendReq is None:
        return 0.0
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        timeframe = f"now {days}-d"
        pytrends.build_payload([keyword], timeframe=timeframe)
        df = pytrends.interest_over_time()
        if df.empty:
            return 0.0
        series = df[keyword].astype(float)
        return float((series.iloc[-1] - series.iloc[0]) / (series.mean() + 1e-9))
    except Exception:
        return 0.0

# --- Sentiment calc ---
analyzer = SentimentIntensityAnalyzer()
def compute_compound_sentiment(texts):
    if not texts:
        return 0.0
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts if isinstance(t, str)]
    return float(np.mean(scores)) if scores else 0.0

# --- Gather sentiment ---
if st.button("Fetch multi-source sentiment now"):
    with st.spinner("Fetching news, reddit and trends…"):
        news_texts = fetch_newsapi_texts(ticker, days=7, page_size=30)
        reddit_titles = fetch_reddit_titles(ticker, limit=80)
        trends_m = google_trends_momentum(ticker, days=7)

        news_score = compute_compound_sentiment(news_texts)
        reddit_score = compute_compound_sentiment(reddit_titles)
        trend_score = float(trends_m)

        # Weighted composite
        w_news, w_reddit, w_trend = 0.6, 0.25, 0.15
        composite_sentiment = w_news*news_score + w_reddit*reddit_score + w_trend*trend_score

        st.write("**Sentiment breakdown:**")
        st.write(f"- News (avg compound): {news_score:.3f} (n={len(news_texts)})")
        st.write(f"- Reddit (avg compound): {reddit_score:.3f} (n={len(reddit_titles)})")
        st.write(f"- Google Trends momentum: {trend_score:.3f}")
        st.success(f"Composite sentiment score: {composite_sentiment:.3f}")

        if composite_sentiment > 0.05:
            drift_sentiment = 0.02
        elif composite_sentiment < -0.05:
            drift_sentiment = -0.01
        else:
            drift_sentiment = 0.005

        st.info(f"Assigned sentiment drift (daily): {drift_sentiment:.3%}")
else:
    news_texts, reddit_titles = [], []
    news_score = reddit_score = trend_score = composite_sentiment = 0.0
    drift_sentiment = 0.005

# --- Volatility ---
hist_close = hist["Close"].ffill().dropna()
hist_returns = hist_close.pct_change().dropna()
vol_est = float(hist_returns.std()) if len(hist_returns) > 5 else 0.02
vol_est = min(max(vol_est, 0.005), 0.5)
st.info(f"Historical daily volatility estimate: {vol_est*100:.2f}%")

# --- Peer correlation with XBI ---
try:
    xbi = yf.Ticker("XBI").history(period="90d")["Close"].pct_change().dropna()
    corr_with_xbi = float(np.corrcoef(hist_returns.tail(len(xbi)), xbi)[0,1])
    st.write(f"Correlation with XBI (recent): {corr_with_xbi:.3f}")
except Exception:
    corr_with_xbi = 0.0

if news_texts:
    st.markdown("#### Sample recent headlines")
    for t in news_texts[:5]:
        st.write("- " + t)

# --- Share to session ---
st.session_state.setdefault("ticker", ticker)
st.session_state.setdefault("drift_sentiment", drift_sentiment)
st.session_state.setdefault("vol_est", vol_est)
st.session_state.setdefault("composite_sentiment", composite_sentiment)
st.session_state.setdefault("corr_with_xbi", corr_with_xbi)

st.success("✅ Section 2 ready.")

# ---------------------------------------------------------------------
### --- SECTION 3: Monte Carlo Forecast, Probabilities & Best Limit Helper ---

st.header("Section 3 — Monte Carlo Forecast & Target Probabilities")

ticker = st.session_state.get("ticker", "SPRB")
drift_sentiment = float(st.session_state.get("drift_sentiment", 0.005))
vol_est = float(st.session_state.get("vol_est", 0.02))

colA, colB, colC = st.columns(3)
with colA:
    start_price = st.number_input("Start price", min_value=0.01,
                                  value=float(hist["Close"].iloc[-1]), step=0.01)
with colB:
    forecast_days = st.slider("Forecast horizon (trading days)", 1, 10, 2)
with colC:
    simulations = st.slider("Simulations", 1000, 50000, 10000, step=1000)

intraday_steps_per_day = 13
daily_drift = drift_sentiment
st.write(f"Using daily drift = **{daily_drift:.3%}**, vol = **{vol_est:.2%}**, steps/day = **{intraday_steps_per_day}**")

def simulate_paths(start_price, days, sims, daily_vol, daily_mu, steps_per_day=13, seed=42):
    rng = np.random.default_rng(seed)
    total_steps = days * steps_per_day
    mu_step = (daily_mu - 0.5 * daily_vol**2) / steps_per_day
    sigma_step = daily_vol / np.sqrt(steps_per_day)

    df = 3
    t_raw = rng.standard_t(df, size=(sims, total_steps))
    t_std = np.sqrt(df / (df - 2))
    increments = (t_raw / t_std) * sigma_step + mu_step
    increments += rng.normal(0, 0.002, size=(sims, total_steps))

    log_prices = np.cumsum(increments, axis=1)
    prices = start_price * np.exp(log_prices)
    prices = np.concatenate([np.full((sims,1), start_price), prices], axis=1)

    intraday_max, intraday_close = [], []
    for d in range(days):
        sidx, eidx = d*steps_per_day + 1, (d+1)*steps_per_day + 1
        chunk = prices[:, sidx:eidx]
        intraday_max.append(chunk.max(axis=1))
        intraday_close.append(chunk[:, -1])
    return prices, np.vstack(intraday_max).T, np.vstack(intraday_close).T

if st.button("Run 2-Day Forecast Now"):
    with st.spinner("Simulating paths…"):
        paths, intraday_max, intraday_close = simulate_paths(
            start_price, forecast_days, simulations, vol_est, daily_drift, intraday_steps_per_day
        )

    final_close = intraday_close[:, -1]
    mean_price, median_price = np.mean(final_close), np.median(final_close)
    p5, p95 = np.percentile(final_close, [5, 95])

    st.markdown("### Distribution (last day close)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"${mean_price:,.2f}")
    col2.metric("Median", f"${median_price:,.2f}")
    col3.metric("5th %ile", f"${p5:,.2f}")
    col4.metric("95th %ile", f"${p95:,.2f}")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(final_close, bins=120, density=True)
    ax.axvline(mean_price, color="blue", ls="--", label=f"Mean ${mean_price:,.2f}")
    ax.axvline(median_price, color="green", ls="--", label=f"Median ${median_price:,.2f}")
    ax.set_title("Simulated Final-Day Closing Price Distribution")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # Intraday-touch probabilities
    st.markdown("### Intraday-touch probabilities (over entire horizon)")
    max_over_horizon = intraday_max.max(axis=1)
    targets = list(range(200, 235, 5))
    probs = []
    for tval in targets:
        ptouch = np.mean(max_over_horizon >= tval)
        pclose = np.mean(final_close >= tval)
        probs.append({"Target ($)": tval,
                      "P(Touch Intraday)": ptouch,
                      "P(Close ≥ Target)": pclose})
    df_probs = pd.DataFrame(probs)
    df_probs_fmt = df_probs.copy()
    df_probs_fmt["P(Touch Intraday)"] = (df_probs_fmt["P(Touch Intraday)"]*100).map("{:.1f}%".format)
    df_probs_fmt["P(Close ≥ Target)"] = (df_probs_fmt["P(Close ≥ Target)"]*100).map("{:.1f}%".format)
    st.dataframe(df_probs_fmt, use_container_width=True)

    # Best limit heuristic
    st.markdown("### Best Limit between $200–$230 (probability × profit heuristic)")
    shares = st.number_input("Your shares", value=41, min_value=1)
    total_cost = st.number_input("Your total cost ($)", value=5683.0, min_value=0.0, step=1.0)
    breakeven = total_cost / shares

    best_row, best_score = None, -1e9
    rows = []
    for _, r in df_probs.iterrows():
        price = float(r["Target ($)"])
        pfill = float(r["P(Touch Intraday)"])
        profit_if_hit = (price - breakeven) * shares
        score = pfill * profit_if_hit
        rows.append({"Limit ($)": price,
                     "P(Touch Intraday)": pfill,
                     "Profit if filled ($)": profit_if_hit,
                     "Score =P×Profit": score})
        if score > best_score:
            best_score, best_row = score, (price, pfill, profit_if_hit)

    df_best = pd.DataFrame(rows)
    df_best_fmt = df_best.copy()
    df_best_fmt["P(Touch Intraday)"] = (df_best_fmt["P(Touch Intraday)"]*100).map("{:.1f}%".format)
    df_best_fmt["Profit if filled ($)"] = df_best_fmt["Profit if filled ($)"].map("{:,.0f}".format)
    df_best_fmt["Score =P×Profit"] = df_best_fmt["Score =P×Profit"].map("{:,.0f}".format)
    st.dataframe(df_best_fmt, use_container_width=True)

    if best_row:
        limit_price, pfill, prof = best_row
        st.success(f"**Suggested limit:** ${limit_price:.0f}  |  P(touch) ≈ {pfill*100:.1f}%  |  Profit ≈ ${prof:,.0f}")

    # Optional download
    if st.button("Prepare CSV sample"):
        sample_df = pd.DataFrame({
            "final_close": final_close,
            "max_over_horizon": max_over_horizon
        })
        st.download_button("Download CSV",
                           data=sample_df.to_csv(index=False),
                           file_name=f"{ticker}_sim_sample.csv",
                           mime="text/csv")
### --- SECTION 4: Analogs + Fundamentals + Risk Regime + Blended Forecast ---

st.header("Section 4 — Analogs, Fundamentals, Risk Regime & Blended Forecast")

# Reuse objects from previous sections
ticker = st.session_state.get("ticker", ticker)
daily_vol = float(st.session_state.get("vol_est", vol_est))
base_drift = float(st.session_state.get("drift_sentiment", drift_sentiment))
composite_sentiment = float(st.session_state.get("composite_sentiment", 0.0))

# 4A) Fundamentals (simple, transparent heuristic)
def fundamental_score(info_obj, ticker_obj):
    try:
        mktcap = float(info_obj.get("marketCap") or 0.0)
        cash = float(info_obj.get("totalCash") or 0.0)
        debt = float(info_obj.get("totalDebt") or 0.0)
        revenue = float(info_obj.get("totalRevenue") or 0.0)
    except Exception:
        mktcap = cash = debt = revenue = 0.0

    # cash runway proxy from cashflow (may be missing for small caps)
    runway_years = 0.0
    try:
        cf = ticker_obj.cashflow
        # try a few common rows
        ops_cf = 0.0
        for key in [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "OperatingActivitiesNetCashProvidedByUsedIn"
        ]:
            if key in cf.index:
                ops_cf = float(cf.loc[key].iloc[0])
                break
        burn = -ops_cf if ops_cf < 0 else 0.0
        if burn > 0:
            runway_years = cash / (burn + 1e-9)
    except Exception:
        runway_years = 0.0

    # normalize to [-1, +1], favor more cash, less debt, some revenue, longer runway
    s = 0.0
    s += np.tanh((cash - debt) / 1_000_000.0)
    s += np.tanh((revenue - 1_000_000.0) / 1_000_000.0)
    s += np.tanh(runway_years / 2.0)  # >2y runway scores well
    s = float(np.clip(s / 3.0, -1.0, 1.0))
    return s, dict(market_cap=mktcap, cash=cash, debt=debt, revenue=revenue, runway_years=runway_years)

tk_obj = yf.Ticker(ticker)
fund_score, fund_bits = fundamental_score(info, tk_obj)

colF1, colF2, colF3, colF4, colF5 = st.columns(5)
colF1.metric("Market Cap", f"${fund_bits['market_cap']:,.0f}")
colF2.metric("Cash", f"${fund_bits['cash']:,.0f}")
colF3.metric("Debt", f"${fund_bits['debt']:,.0f}")
colF4.metric("Revenue", f"${fund_bits['revenue']:,.0f}")
colF5.metric("Runway (yrs, est)", f"{fund_bits['runway_years']:.2f}")
st.info(f"Fundamental signal (−1..+1): **{fund_score:.3f}**")

# 4B) Market / Sector risk regime (SPY & XBI momentum)
def market_regime():
    try:
        spy = yf.Ticker("SPY").history(period="30d")["Close"].pct_change().dropna()
        xbi = yf.Ticker("XBI").history(period="30d")["Close"].pct_change().dropna()
        spy_mom = float(spy.tail(7).mean())
        xbi_mom = float(xbi.tail(7).mean())
        risk_on = (spy_mom > 0) and (xbi_mom > 0)
        return dict(spy_mom=spy_mom, xbi_mom=xbi_mom, risk_on=risk_on)
    except Exception:
        return dict(spy_mom=0.0, xbi_mom=0.0, risk_on=False)

reg = market_regime()
risk_weight = 1.0 if reg["risk_on"] else 0.7
st.write(f"Market regime → SPY 7-day mom: {reg['spy_mom']:.3%} | XBI 7-day mom: {reg['xbi_mom']:.3%} | Risk-on: {reg['risk_on']}")

# 4C) Event analogs via sentence embeddings (optional)
analogs_df = None
analogs_adj_7 = 0.0
analogs_adj_30 = 0.0

st.markdown("#### Event analogs (optional)")
uploaded = st.file_uploader("Upload event_analogs_db.csv (optional)", type=["csv"], accept_multiple_files=False)
if uploaded is not None:
    try:
        analogs_df = pd.read_csv(uploaded, parse_dates=["date"])
    except Exception:
        st.error("Could not read CSV — expecting columns: date,ticker,title,type,notes,outcome_return_7d,outcome_return_30d")
        analogs_df = None
else:
    # look for local file if user hasn’t uploaded
    if os.path.exists("event_analogs_db.csv"):
        try:
            analogs_df = pd.read_csv("event_analogs_db.csv", parse_dates=["date"])
        except Exception:
            analogs_df = None

def embed_texts_sbert(text_list):
    if SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(text_list, show_progress_bar=False, normalize_embeddings=True)
        return emb.astype("float32")
    except Exception:
        return None

top_news = []
try:
    # reuse news_texts from Section 2 if available
    if isinstance(news_texts, list) and len(news_texts) > 0:
        top_news = news_texts[:6]
except Exception:
    pass

if analogs_df is not None and len(analogs_df) > 0 and len(top_news) > 0:
    st.caption(f"Analog DB loaded: {len(analogs_df)} events. Finding nearest matches to today’s news...")
    # build embeddings (DB + current event bundle)
    db_titles = analogs_df["title"].fillna("").tolist()
    db_emb = embed_texts_sbert(db_titles)
    cur_text = " || ".join(top_news)
    cur_emb = embed_texts_sbert([cur_text])

    if db_emb is not None and cur_emb is not None:
        # cosine similarity since embeddings are normalized: dot product = cosine
        sims = np.dot(db_emb, cur_emb[0])  # shape (N,)
        k = int(min(6, len(sims)))
        top_idx = sims.argsort()[::-1][:k]
        view = analogs_df.iloc[top_idx].copy()
        view["similarity"] = sims[top_idx]
        st.dataframe(view[["date", "ticker", "title", "type", "outcome_return_7d", "outcome_return_30d", "similarity"]])

        # compute mean forward returns (ignore NaNs)
        if "outcome_return_7d" in view.columns:
            analogs_adj_7 = float(pd.to_numeric(view["outcome_return_7d"], errors="coerce").dropna().mean() or 0.0)
        if "outcome_return_30d" in view.columns:
            analogs_adj_30 = float(pd.to_numeric(view["outcome_return_30d"], errors="coerce").dropna().mean() or 0.0)
        st.write(f"Analogs adjustment → 7-day mean: {analogs_adj_7:.3f} | 30-day mean: {analogs_adj_30:.3f}")
    else:
        st.warning("Sentence-Transformers not installed or embedding failed — skipping analogs.")
else:
    st.info("No analog DB or current news available — skipping analog adjustments (optional).")

# 4D) Build a blended daily drift (kept conservative, clamped)
# base_drift from Section 2 sentiment. We add influences from sentiment composite, fundamentals, and analogs (7d).
raw_blend = base_drift + risk_weight * (0.5 * composite_sentiment + 0.3 * fund_score + 0.7 * analogs_adj_7)
blended_drift = float(np.clip(raw_blend, -0.05, 0.05))  # clamp daily drift to ±5% (conservative)
st.subheader("Blended daily drift")
st.write(
    f"- Base (sentiment) drift: {base_drift:.3%}  |  "
    f"Composite sentiment: {composite_sentiment:.3f}  |  "
    f"Fundamental signal: {fund_score:.3f}  |  "
    f"Analogs 7d: {analogs_adj_7:.3f}  |  "
    f"Risk weight: {risk_weight:.2f}"
)
st.success(f"→ **Blended drift (daily): {blended_drift:.3%}**")

# 4E) Run a baseline vs blended forecast and plot combined view
st.markdown("### Combined Confidence Forecast (baseline vs blended)")
colX, colY = st.columns(2)
with colX:
    start_price_blend = st.number_input("Start price for blended run", min_value=0.01, value=float(hist["Close"].iloc[-1]), step=0.01)
with colY:
    days_blend = st.slider("Horizon (days) for blended plot", 1, 20, 5)

if st.button("Run blended forecast"):
    with st.spinner("Simulating baseline and blended scenarios…"):
        # reuse simulate_paths() from Section 3
        paths_base, _, _ = simulate_paths(start_price_blend, days_blend, 6000, daily_vol, base_drift, steps_per_day=13, seed=11)
        paths_blnd, _, _ = simulate_paths(start_price_blend, days_blend, 6000, daily_vol, blended_drift, steps_per_day=13, seed=12)

    mean_base = np.mean(paths_base, axis=0)
    p10_base = np.percentile(paths_base, 10, axis=0)
    p90_base = np.percentile(paths_base, 90, axis=0)

    mean_blnd = np.mean(paths_blnd, axis=0)
    p10_blnd = np.percentile(paths_blnd, 10, axis=0)
    p90_blnd = np.percentile(paths_blnd, 90, axis=0)

    # simple "fair-value" trajectory from fundamentals: linearly drift toward 1+fund_score
    fv_line = np.linspace(start_price_blend, start_price_blend * (1.0 + fund_score), len(mean_blnd))

    steps = np.arange(len(mean_blnd))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    # baseline band
    ax2.fill_between(steps, p10_base, p90_base, color="tab:blue", alpha=0.15, label="Baseline 10–90%")
    ax2.plot(mean_base, color="tab:blue", lw=1.5, label="Baseline mean")
    # blended band
    ax2.fill_between(steps, p10_blnd, p90_blnd, color="tab:orange", alpha=0.18, label="Blended 10–90%")
    ax2.plot(mean_blnd, color="tab:orange", lw=2.0, label="Blended mean")
    # fair value guide
    ax2.plot(fv_line, color="tab:green", ls="--", lw=2, label="Fundamental fair-value guide")
    ax2.set_title(f"{ticker}: Combined Confidence Forecast (baseline vs blended)")
    ax2.set_xlabel("Intraday steps")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    st.pyplot(fig2)

    # Optional: recompute ladder on blended distribution (last point)
    st.markdown("#### Ladder on blended scenario (last-day close distribution)")
    final_blnd = paths_blnd[:, -1]
    max_blnd = paths_blnd.max(axis=1)
    targets = list(range(200, 235, 5))
    rows = []
    for tval in targets:
        ptouch = float(np.mean(max_blnd >= tval))
        pclose = float(np.mean(final_blnd >= tval))
        rows.append({"Target ($)": tval, "P(Touch Intraday)": ptouch, "P(Close ≥ Target)": pclose})
    df_b = pd.DataFrame(rows)
    df_b_fmt = df_b.copy()
    df_b_fmt["P(Touch Intraday)"] = (df_b_fmt["P(Touch Intraday)"]*100).map("{:.1f}%".format)
    df_b_fmt["P(Close ≥ Target)"] = (df_b_fmt["P(Close ≥ Target)"]*100).map("{:.1f}%".format)
    st.dataframe(df_b_fmt, use_container_width=True)

    # quick expected-proceeds ladder (same idea as Section 3)
    shares2 = st.number_input("Shares (blended ladder)", value=41, min_value=1, key="shares_blended")
    cost2 = st.number_input("Total cost ($, blended ladder)", value=5683.0, min_value=0.0, step=1.0, key="cost_blended")
    breakeven2 = cost2 / shares2
    best_row = None
    best_score = -1e18
    rows2 = []
    for _, r in df_b.iterrows():
        price = float(r["Target ($)"])
        pfill = float(r["P(Touch Intraday)"])
        profit_if_hit = (price - breakeven2) * shares2
        score = pfill * profit_if_hit
        rows2.append({"Limit ($)": price, "P(Touch Intraday)": pfill, "Profit if filled ($)": profit_if_hit, "Score=P×Profit": score})
        if score > best_score:
            best_score = score
            best_row = (price, pfill, profit_if_hit)
    df2 = pd.DataFrame(rows2)
    df2_fmt = df2.copy()
    df2_fmt["P(Touch Intraday)"] = (df2_fmt["P(Touch Intraday)"]*100).map("{:.1f}%".format)
    df2_fmt["Profit if filled ($)"] = df2_fmt["Profit if filled ($)"].map("{:,.0f}".format)
    df2_fmt["Score=P×Profit"] = df2_fmt["Score=P×Profit"].map("{:,.0f}".format)
    st.dataframe(df2_fmt, use_container_width=True)
    if best_row:
        st.success(f"Blended ladder suggestion → Limit ${best_row[0]:.0f} | P(touch)≈{best_row[1]*100:.1f}% | Profit≈${best_row[2]:,.0f}")
