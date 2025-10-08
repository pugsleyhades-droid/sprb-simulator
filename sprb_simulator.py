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
time_interval = st.selectbox(
    "Select Intraday Time Interval",
    options=["10 minutes", "15 minutes", "30 minutes", "1 hour", "5 hours", "10 hours", "20 hours"]
)

# Time interval to minutes mapping
interval_mapping = {
    "10 minutes": 10,
    "15 minutes": 15,
    "30 minutes": 30,
    "1 hour": 60,
    "5 hours": 5 * 60,
    "10 hours": 10 * 60,
    "20 hours": 20 * 60
}

# --- 5. MARKET HOURS FILTERING ---
nyse = mcal.get_calendar('NYSE')
today = pd.Timestamp(datetime.now(pytz.UTC)).normalize()
schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=forecast_days * 2))
trading_days = schedule.index[:forecast_days].tolist()

intraday_minutes = 6.5 * 60  # Total market hours in minutes
minutes_per_step_adjusted = interval_mapping[time_interval]  # Adjusted minutes per step

intraday_steps_per_day_adjusted = int(intraday_minutes / minutes_per_step_adjusted)  # Steps per day based on interval

intraday_times_adjusted = []
for day in trading_days:
    day_start = day + pd.Timedelta(hours=9, minutes=30)
    times = [day_start + pd.Timedelta(minutes=minutes_per_step_adjusted * i) for i in range(intraday_steps_per_day_adjusted)]
    intraday_times_adjusted.extend(times)

# Total steps across all days
total_steps_adjusted = len(intraday_times_adjusted)

# --- 6. SIMULATION ---
mu = (drift_sentiment - 0.5 * volatility_adj ** 2) * (1 / intraday_steps_per_day_adjusted)
sigma = volatility_adj * np.sqrt(1 / intraday_steps_per_day_adjusted)

price_paths = np.zeros((1000, total_steps_adjusted + 1))  # Adjusted steps count
price_paths[:, 0] = live_price
np.random.seed(42)

for t in range(1, total_steps_adjusted + 1):
    shocks = np.random.normal(loc=mu, scale=sigma, size=1000)
    volume_noise = np.random.normal(loc=0, scale=0.005, size=1000)
    price_paths[:, t] = price_paths[:, t-1] * np.exp(shocks + volume_noise)
    price_paths[:, t] = np.clip(price_paths[:, t], 0.01, None)

# --- 7. DAILY CLOSES ---
day_indices = []
current_day = trading_days[0]
idx_list = []
for i, ts in enumerate(intraday_times_adjusted):
    if ts.normalize() != current_day:
        day_indices.append(idx_list)
        idx_list = []
        current_day = ts.normalize()
    idx_list.append(i + 1)
day_indices.append(idx_list)

daily_closes = np.array([price_paths[:, indices[-1]] for indices in day_indices]).T

# --- 8. DISPLAY METRICS ---
st.markdown("### 📅 Daily Closing Price Percentiles")
percentiles = [5, 50, 95]
percentile_values = {p: np.percentile(daily_closes, p, axis=0) for p in percentiles}

df_metrics = pd.DataFrame({
    f"P{p}": percentile_values[p] for p in percentiles
}, index=[f"Day {i+1}" for i in range(len(day_indices))])

st.dataframe(df_metrics.style.format("${:.2f}"))

fig, ax = plt.subplots(figsize=(8, 4))
days = np.arange(1, len(day_indices) + 1)
ax.plot(days, percentile_values[5], label='5th Percentile', linestyle='--', color='orange')
ax.plot(days, percentile_values[50], label='Median', linestyle='-', color='red')
ax.plot(days, percentile_values[95], label='95th Percentile', linestyle='--', color='green')
ax.set_xlabel("Trading Day")
ax.set_ylabel("Price ($)")
ax.set_title("Simulated Daily Closing Price Percentiles")
ax.legend()
st.pyplot(fig)

# --- 9. INTRADAY PATHS SAMPLE with Average Path ---
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plotting sample paths
for i in range(10):  # Show 10 sample paths
    ax2.plot(intraday_times_adjusted, price_paths[i], alpha=0.4)

# Plot average path
avg_path = price_paths.mean(axis=0)
ax2.plot(intraday_times_adjusted, avg_path, color='black', label='Average Path', lw=2)

ax2.set_xlabel("Time")
ax2.set_ylabel("Price ($)")
ax2.set_title("Simulated Intraday Price Paths (10 samples + Average)")
ax2.legend(loc='upper left')

st.pyplot(fig2)

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
    - `mu` is the drift (expected return) factor.
    - `sigma` is the volatility factor.
    - Simulations use a **log-normal random walk** model with added **volume noise**.
    """)
