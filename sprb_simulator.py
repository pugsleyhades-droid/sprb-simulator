import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import pytz
import pandas_market_calendars as mcal

st.set_page_config(page_title="SPRB Advanced Simulator", layout="centered")
st.title("🚀 SPRB Advanced Multi-Day Simulator with Live Sentiment & Market Hours")

# --- 1. LIVE PRICE & HISTORICAL DATA ---
ticker = yf.Ticker("SPRB")
try:
    hist = ticker.history(period="60d")
    live_price = hist["Close"][-1]
    hist_vol = hist["Volume"]
except Exception:
    st.error("Failed to fetch price data.")
    st.stop()

if live_price <= 0:
    st.error("Invalid live price data.")
    st.stop()

st.success(f"**Live Price:** ${live_price:.2f}")

# --- 2. LIVE NEWS SENTIMENT PULL ---
st.markdown("### 📰 Live News Sentiment")

def fetch_news_and_sentiment(query="SPRB", days=3):
    API_KEY = "YOUR_NEWSAPI_KEY"  # Replace with your actual key
    url = f"https://newsapi.org/v2/everything?q={query}&from={(datetime.utcnow() - timedelta(days=days)).date()}&language=en&sortBy=publishedAt&pageSize=20&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        texts = [a['title'] + ". " + (a['description'] or "") for a in articles]
    except Exception:
        texts = []
    return texts

def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    if not texts:
        return 0.0
    scores = [analyzer.polarity_scores(text)['compound'] for text in texts]
    return np.mean(scores)

with st.spinner("Fetching news..."):
    news_texts = fetch_news_and_sentiment()
    sentiment_score = analyze_sentiment(news_texts)

sentiment_label = "Neutral"
if sentiment_score > 0.05:
    sentiment_label = "Bullish"
elif sentiment_score < -0.05:
    sentiment_label = "Bearish"

st.write(f"Sentiment Score: {sentiment_score:.3f} ({sentiment_label})")

sentiment_drift_map = {
    "Bullish": 0.02,
    "Neutral": 0.005,
    "Bearish": -0.01
}
drift_sentiment = sentiment_drift_map[sentiment_label]

# --- 3. HISTORICAL VOLATILITY & VOLUME ---
hist["Returns"] = hist["Close"].pct_change()
vol_estimate = hist["Returns"].std()

if np.isnan(vol_estimate) or vol_estimate <= 0:
    vol_estimate = 0.007
elif vol_estimate > 0.10:
    vol_estimate = 0.10

vol_avg = hist_vol[-10:].mean()
vol_norm = vol_avg / hist_vol.mean()
volatility_adj = vol_estimate * (1 + (vol_norm - 1) * 0.2)

st.info(f"Estimated Daily Volatility (adjusted by volume): {volatility_adj*100:.2f}%")

# --- 4. USER SETTINGS ---
forecast_days = st.slider("Forecast Horizon (days)", 1, 30, 5)
simulations = 1000

# --- 5. MARKET HOURS FILTERING ---
nyse = mcal.get_calendar('NYSE')
today = pd.Timestamp(datetime.now(pytz.UTC)).normalize()
schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=forecast_days * 2))
trading_days = schedule.index[:forecast_days].tolist()

intraday_minutes = 6.5 * 60
intraday_steps_per_day = 13  # Default initial setting

# --- 6. SIMULATION ---
mu = (drift_sentiment - 0.5 * volatility_adj ** 2) * (1 / intraday_steps_per_day)
sigma = volatility_adj * np.sqrt(1 / intraday_steps_per_day)

price_paths = np.zeros((simulations, len(trading_days) * intraday_steps_per_day + 1))
price_paths[:, 0] = live_price
np.random.seed(42)

for t in range(1, len(price_paths[0])):
    shocks = np.random.normal(loc=mu, scale=sigma, size=simulations)
    volume_noise = np.random.normal(loc=0, scale=0.005, size=simulations)
    price_paths[:, t] = price_paths[:, t-1] * np.exp(shocks + volume_noise)
    price_paths[:, t] = np.clip(price_paths[:, t], 0.01, None)

# --- 7. DAILY CLOSES ---
intraday_times = []
for day in trading_days:
    day_start = day + pd.Timedelta(hours=9, minutes=30)
    times = [day_start + pd.Timedelta(minutes=(intraday_minutes / intraday_steps_per_day) * i) for i in range(intraday_steps_per_day)]
    intraday_times.extend(times)

# --- 8. PLOTTING SIMULATED INTRADAY PATHS ---
sample_paths = price_paths[:min(10, simulations), :]

# Calculate average path for all simulations
average_path = np.mean(price_paths, axis=0)

# Convert intraday times to hours
time_hours = [(ts - intraday_times[0]).total_seconds() / 3600 for ts in intraday_times]
time_hours = [0.0] + time_hours

# Plotting the sample intraday paths
fig, ax2 = plt.subplots(figsize=(10, 6))
for i in range(min(10, simulations)):
    ax2.plot(time_hours, sample_paths[i], alpha=0.4, label=f"Path {i+1}")
ax2.plot(time_hours, average_path, label="Average Path", color='blue', linewidth=2)
ax2.set_xlabel("Time (Hours)")
ax2.set_ylabel("Price ($)")
ax2.set_title("Sample Intraday Price Paths & Average Path")
ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1))
st.pyplot(fig)

# --- 9. USER INTERFACE - INTRADAY STEPS ---
intraday_steps_per_day = st.selectbox(
    "Intraday Steps per Day",
    options=[1, 4, 13, 26, 52],
    format_func=lambda x: f"{x} steps/day (~{round(6.5*60/x)} min each)"
)

# --- EXPLANATION BLOCK FOR INTRADAY PRICE PATHS ---
with st.expander("ℹ️ What Are Sample Intraday Price Paths?"):
    st.markdown("""
    These lines represent **simulated price movements of SPRB during market hours** across multiple forecasted trading days.

    Each line shows one **possible intraday scenario** generated using:

    - 📈 **Starting Price**: The current live market price of SPRB
    - 🔄 **Volatility**: Based on historical price swings, adjusted for trading volume
    - 🧠 **Drift (trend)**: Bias derived from live news sentiment (Bullish/Neutral/Bearish)
    - ⏰ **Market hours only**: Simulations occur only between 9:30 AM and 4:00 PM ET
    - 🎲 **Randomness**: Reflects unpredictable market movements and volume-related noise

    ---
    ### Technical Notes:
    - `mu`: Expected return per step (based on sentiment)
    - `sigma`: Volatility per step (from historical data)
    - `dt`: Time step size (e.g., 1/13 for 30-minute intervals)
    - Each step: `Price[t] = Price[t-1] * exp(mu + randomness)`
    - 10 paths are shown out of 1000 total simulations

    ---
    Adjust the number of days and intraday resolution above to explore different outcomes.
    """)

