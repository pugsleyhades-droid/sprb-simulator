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

# optional advanced modules (you'll install them later; we guard-import here)
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
- Monte-Carlo price paths  
- Multi-source sentiment  
- Peer correlations  
- (Later) event analog embeddings, fundamentals & risk weighting  
""")

st.success("✅ Section 1 (Imports & Setup) loaded.")

# ------------------------------------------------------------------------------------
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

# --- NewsAPI function (uses st.secrets for the key) ---
def fetch_newsapi_texts(query, days=7, page_size=30):
    # Reads key from .streamlit/secrets.toml; shows warning if missing
    try:
        api_key = st.secrets["newsapi"]["key"]
    except Exception:
        st.warning("NewsAPI key missing in .streamlit/secrets.toml → news fetching disabled.")
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

# --- Reddit fetch (simple; best-effort) ---
def fetch_reddit_titles(keyword, limit=50):
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

# --- Sentiment calculation utility ---
analyzer = SentimentIntensityAnalyzer()
def compute_compound_sentiment(texts):
    if not texts:
        return 0.0
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts if isinstance(t, str)]
    return float(np.mean(scores)) if scores else 0.0

# --- Gather multi-source sentiment (button to control live calls) ---
if st.button("Fetch multi-source sentiment now"):
    with st.spinner("Fetching news, reddit and trends..."):
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
        st.write(f"- News (avg compound): {news_score:.3f}  (n={len(news_texts)})")
        st.write(f"- Reddit (avg compound): {reddit_score:.3f}  (n={len(reddit_titles)})")
        st.write(f"- Google Trends
