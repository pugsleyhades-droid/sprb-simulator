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

# optional advanced modules (you'll install them later)
try:
    from pytrends.request import TrendReq
    from sklearn.preprocessing import MinMaxScaler
    from sentence_transformers import SentenceTransformer
except ImportError:
    TrendReq = None
    SentenceTransformer = None

# --- Streamlit page setup ---
st.set_page_config(page_title="SPRB Advanced Forecast Simulator", layout="wide")
st.title("🚀 SPRB Advanced Forecast Simulator (Section 1 Loaded)")

st.markdown("""
This app models **multi-factor biotech stock behavior** using:
- Monte-Carlo price paths  
- Multi-source sentiment  
- Peer correlations  
- Event analog embeddings  
- Fundamental valuation & risk weighting  

You’re currently running **Section 1 (Imports & Setup)**.  
Once Python is installed and this file runs without errors, we’ll add the next sections.
""")

# --- Simple test so you can confirm the install works ---
st.success("✅ Section 1 loaded successfully. Continue to Section 2 once Streamlit is installed.")
### --- SECTION 2: Data Fetch, Multi-Source Sentiment & Volatility ---

# --- UI: get ticker and basic options ---
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
    except Exception as e:
        return None, {}

hist, info = fetch_market_and_info(ticker)
if hist is None or hist.empty:
    st.error("Historical price data unavailable for ticker. Please check the ticker symbol.")
else:
    st.success(f"Fetched historical data: {len(hist)} rows (last close ${hist['Close'].iloc[-1]:.2f})")
    latest_close = float(hist["Close"].iloc[-1])
    st.metric("Latest Close", f"${latest_close:.2f}")

# --- NewsAPI function (uses st.secrets for the key) ---
def fetch_newsapi_texts(query, days=7, page_size=30):
    # Will read key from .streamlit/secrets.toml; show warning if missing
    try:
        api_key = st.secrets["newsapi"]["key"]
    except Exception:
        st.warning("NewsAPI key missing from .streamlit/secrets.toml — news fetching disabled.")
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
    # Best-effort; may be rate-limited/broken. Use Pushshift or authenticated reddit for reliability.
    try:
        url = f"https://api.pushshift.io/reddit/search/submission/?q={keyword}&size={limit}"
        r = requests.get(url, timeout=6)
        data = r.json().get("data", [])
        titles = [d.get("title","") for d in data]
        return titles
    except Exception:
        return []

# --- Google Trends (if pytrends available) ---
def google_trends_momentum(keyword, days=7):
    if 'TrendReq' not in globals() or TrendReq is None:
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

# --- Sentiment calculation utility ---
analyzer = SentimentIntensityAnalyzer()

def compute_compound_sentiment(texts):
    if not texts:
        return 0.0
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts if isinstance(t, str)]
    if not scores:
        return 0.0
    return float(np.mean(scores))

# --- Gather multi-source sentiment (button to control live calls) ---
if st.button("Fetch multi-source sentiment now"):
    with st.spinner("Fetching news, reddit and trends..."):
        news_texts = fetch_newsapi_texts(ticker, days=7, page_size=30)
        reddit_titles = fetch_reddit_titles(ticker, limit=80)
        trends_m = google_trends_momentum(ticker, days=7)

        news_score = compute_compound_sentiment(news_texts)
        reddit_score = compute_compound_sentiment(reddit_titles)
        trend_score = float(trends_m)

        # Weighted composite (tune weights as you like)
        w_news, w_reddit, w_trend = 0.6, 0.25, 0.15
        composite_sentiment = w_news*news_score + w_reddit*reddit_score + w_trend*trend_score

        st.write("**Sentiment breakdown:**")
        st.write(f"- News (avg compound): {news_score:.3f}  (n={len(news_texts)})")
        st.write(f"- Reddit (avg compound): {reddit_score:.3f}  (n={len(reddit_titles)})")
        st.write(f"- Google Trends momentum: {trend_score:.3f}")
        st.success(f"Composite sentiment score: {composite_sentiment:.3f}")

        # Map to drift
        if composite_sentiment > 0.05:
            drift_sentiment = 0.02
        elif composite_sentiment < -0.05:
            drift_sentiment = -0.01
        else:
            drift_sentiment = 0.005

        st.info(f"Assigned sentiment drift (daily): {drift_sentiment:.3%}")
else:
    # defaults before fetching
    news_texts, reddit_titles = [], []
    news_score = reddit_score = trend_score = composite_sentiment = 0.0
    drift_sentiment = 0.005

# --- Historical volatility estimate (daily) ---
hist_close = hist["Close"].ffill().dropna()
hist_returns = hist_close.pct_change().dropna()
vol_est = float(hist_returns.std()) if len(hist_returns)>5 else 0.02
vol_est = min(max(vol_est, 0.005), 0.5)  # clamp
st.info(f"Historical daily volatility estimate: {vol_est*100:.2f}%")

# --- Peer correlation (simple: compare with biotech ETF XBI) ---
try:
    xbi = yf.Ticker("XBI").history(period="90d")["Close"].pct_change().dropna()
    corr_with_xbi = float(np.corrcoef(hist_returns.tail(len(xbi)), xbi)[0,1])
    st.write(f"Correlation with XBI (recent): {corr_with_xbi:.3f}")
except Exception:
    corr_with_xbi = 0.0

# --- Show a small sample of news if available ---
if news_texts:
    st.markdown("#### Sample recent headlines")
    for t in news_texts[:5]:
        st.write("- " + t)

# --- Expose variables to next sections ---
# (These variables will be used by the Monte Carlo engine in later sections)
st.session_state = getattr(st, "session_state", {})
st.session_state.setdefault("ticker", ticker)
st.session_state.setdefault("drift_sentiment", drift_sentiment)
st.session_state.setdefault("vol_est", vol_est)
st.session_state.setdefault("composite_sentiment", composite_sentiment)
st.session_state.setdefault("corr_with_xbi", corr_with_xbi)

st.success("Section 2 ready. Proceed to Section 3 for Monte Carlo + probability outputs.")
### --- SECTION 3: Monte Carlo Forecast, Probabilities & Best Limit Helper ---

st.header("Section 3 — Monte Carlo Forecast & Target Probabilities")

# Pull shared inputs prepared in Section 2
ticker = st.session_state.get("ticker", "SPRB")
drift_sentiment = float(st.session_state.get("drift_sentiment", 0.005))
vol_est = float(st.session_state.get("vol_est", 0.02))
composite_sentiment = float(st.session_state.get("composite_sentiment", 0.0))

# UI controls for this section
colA, colB, colC = st.columns(3)
with colA:
    start_price = st.number_input("Start price (use live/PM if you want)", min_value=0.01, value=float(hist['Close'].iloc[-1]), step=0.01)
with colB:
    forecast_days = st.slider("Forecast horizon (trading days)", 1, 10, 2)
with colC:
    simulations = st.slider("Simulations", 1000, 50000, 10000, step=1000)

intraday_steps_per_day = 13  # ~30-min bars across 6.5h
daily_drift = drift_sentiment  # from Section 2 sentiment

st.write(f"Using daily drift = **{daily_drift:.3%}**, daily vol = **{vol_est:.2%}**, steps/day = **{intraday_steps_per_day}**")

# Monte Carlo engine (fat-tailed increments via Student-t scaled to sigma)
def simulate_paths(start_price, days, sims, daily_vol, daily_mu, steps_per_day=13, seed=42):
    rng = np.random.default_rng(seed)
    total_steps = days * steps_per_day
    # per-step params for geometric process
    mu_step = (daily_mu - 0.5 * daily_vol**2) / steps_per_day
    sigma_step = daily_vol / np.sqrt(steps_per_day)

    # Student-t increments (df=3) scaled to sigma_step
    df = 3
    t_raw = rng.standard_t(df, size=(sims, total_steps))
    t_std = np.sqrt(df/(df-2))
    increments = (t_raw / t_std) * sigma_step + mu_step

    # small microstructure noise
    increments += rng.normal(0, 0.002, size=(sims, total_steps))

    # build price paths
    log_prices = np.cumsum(increments, axis=1)
    prices = start_price * np.exp(log_prices)
    prices = np.concatenate([np.full((sims,1), start_price), prices], axis=1)

    # intraday stats
    intraday_max = []
    intraday_close = []
    for d in range(days):
        sidx = d*steps_per_day + 1
        eidx = (d+1)*steps_per_day + 1
        chunk = prices[:, sidx:eidx]
        intraday_max.append(chunk.max(axis=1))
        intraday_close.append(chunk[:, -1])
    intraday_max = np.vstack(intraday_max).T   # sims x days
    intraday_close = np.vstack(intraday_close).T
    return prices, intraday_max, intraday_close

# Run sim
if st.button("Run 2-Day Forecast Now"):
    with st.spinner("Simulating paths..."):
        paths, intraday_max, intraday_close = simulate_paths(
            start_price, forecast_days, simulations, vol_est, daily_drift, intraday_steps_per_day
        )

    # Compute overall stats
    final_close = intraday_close[:, -1]
    mean_price = float(np.mean(final_close))
    median_price = float(np.median(final_close))
    p5, p95 = [float(x) for x in np.percentile(final_close, [5, 95])]

    st.markdown("### Distribution (last day close)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"${mean_price:,.2f}")
    col2.metric("Median", f"${median_price:,.2f}")
    col3.metric("5th %ile", f"${p5:,.2f}")
    col4.metric("95th %ile", f"${p95:,.2f}")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(final_close, bins=120, density=True)
    ax.axvline(mean_price, color="blue", linestyle="--", label=f"Mean ${mean_price:,.2f}")
    ax.axvline(median_price, color="green", linestyle="--", label=f"Median ${median_price:,.2f}")
    ax.set_title("Simulated Final-Day Closing Price Distribution")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # Intraday-touch probabilities for 200–230 in $5 steps
    st.markdown("### Intraday-touch probabilities (over entire horizon)")
    max_over_horizon = intraday_max.max(axis=1)
    targets = list(range(200, 235, 5))
    probs = []
    for tval in targets:
        ptouch = float(np.mean(max_over_horizon >= tval))
        pclose = float(np.mean(final_close >= tval))
        probs.append({"Target ($)": tval, "P(Touch Intraday)": ptouch, "P(Close ≥ Target)": pclose})
    df_probs = pd.DataFrame(probs)
    df_probs_fmt = df_probs.copy()
    df_probs_fmt["P(Touch Intraday)"] = df_probs_fmt["P(Touch Intraday)"].map(lambda x: f"{x*100:.1f}%")
    df_probs_fmt["P(Close ≥ Target)"] = df_probs_fmt["P(Close ≥ Target)"].map(lambda x: f"{x*100:.1f}%")
    st.dataframe(df_probs_fmt, use_container_width=True)

    # “Best limit” helper between $200 and $230: maximize probability × profit
    st.markdown("### Best Limit between $200–$230 (probability × profit heuristic)")
    shares = st.number_input("Your shares", value=41, min_value=1)
    total_cost = st.number_input("Your total cost ($)", value=5683.0, min_value=0.0, step=1.0)
    breakeven = total_cost / shares if shares > 0 else 0.0

    best_row = None
    best_score = -1e9
    rows = []
    for _, row in df_probs.iterrows():
        price = float(row["Target ($)"])
        pfill = float(row["P(Touch Intraday)"])
        profit_if_hit = (price - breakeven) * shares
        score = pfill * profit_if_hit
        rows.append({
            "Limit ($)": price,
            "P(Touch Intraday)": pfill,
            "Profit if filled ($)": profit_if_hit,
            "Score = P×Profit": score
        })
        if score > best_score:
            best_score = score
            best_row = (price, pfill, profit_if_hit, score)

    df_best = pd.DataFrame(rows)
    df_best_fmt = df_best.copy()
    df_best_fmt["P(Touch Intraday)"] = df_best_fmt["P(Touch Intraday)"].map(lambda x: f"{x*100:.1f}%")
    df_best_fmt["Profit if filled ($)"] = df_best_fmt["Profit if filled ($)"].map(lambda x: f"{x:,.0f}")
    df_best_fmt["Score = P×Profit"] = df_best_fmt["Score = P×Profit"].map(lambda x: f"{x:,.0f}")
    st.dataframe(df_best_fmt, use_container_width=True)

    if best_row:
        limit_price, pfill, prof, score = best_row
        st.success(f"**Suggested limit**: ${limit_price:.0f}  |  P(touch) ≈ {pfill*100:.1f}%  |  Profit if filled ≈ ${prof:,.0f}")

                "max_over_horizon": max_over_horizon
            })
            st.download_button("Download CSV", sample_df.to_csv(index=False), file_name=f"{ticker}_sim_sample.csv")

# End of app
