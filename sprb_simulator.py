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
    hist = ticker.history(period="60d")  # longer for volume and volatility
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
    # Using NewsAPI.org free tier example (replace with your API key)
    API_KEY = "9011c7b1e87c4c7aa0b63dcda687916a"  # You need to get your own API key from https://newsapi.org/
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
        return 0.0  # Neutral if no data
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

# Map sentiment to daily drift modifier
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
    vol_estimate = 0.10  # cap at 10%

# Use rolling volume average (last 10 days)
vol_avg = hist_vol[-10:].mean()

# Normalize volume for volatility adjustment (just example)
vol_norm = vol_avg / hist_vol.mean()
volatility_adj = vol_estimate * (1 + (vol_norm - 1) * 0.2)  # 20% volatility bump or reduction based on volume

st.info(f"Estimated Daily Volatility (adjusted by volume): {volatility_adj*100:.2f}%")

# --- 4. USER SETTINGS ---
forecast_days = st.slider("Forecast Horizon (days)", 1, 30, 5)
intraday_steps_per_day = st.selectbox(
    "Intraday Steps per Day",
    options=[1, 4, 13, 26, 52],
    format_func=lambda x: f"{x} steps/day (~{round(6.5*60/x)} min each)"
)
simulations = 1000

# --- 5. MARKET HOURS FILTERING ---
nyse = mcal.get_calendar('NYSE')
today = pd.Timestamp(datetime.now(pytz.UTC)).normalize()
schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=forecast_days * 2))
trading_days = schedule.index[:forecast_days].tolist()  # first forecast_days trading days

# Create intraday timestamps per trading day, ignoring non-trading days and hours
intraday_minutes = 6.5 * 60  # 6.5 hours trading day
minutes_per_step = intraday_minutes / intraday_steps_per_day

# Generate datetime index for all intraday steps
intraday_times = []
for day in trading_days:
    day_start = day + pd.Timedelta(hours=9, minutes=30)
    times = [day_start + pd.Timedelta(minutes=minutes_per_step * i) for i in range(intraday_steps_per_day)]
    intraday_times.extend(times)

total_steps = len(intraday_times)
dt = 1 / (intraday_steps_per_day)  # fraction of a day per step

# --- 6. SIMULATION WITH VOLUME IMPACT ---
mu = (drift_sentiment - 0.5 * volatility_adj ** 2) * dt
sigma = volatility_adj * np.sqrt(dt)

price_paths = np.zeros((simulations, total_steps + 1))
price_paths[:, 0] = live_price

np.random.seed(42)

for t in range(1, total_steps + 1):
    # Random shocks
    shocks = np.random.normal(loc=mu, scale=sigma, size=simulations)
    
    # Simulate volume impact as random noise scaled by volume variance (simplified)
    volume_noise = np.random.normal(loc=0, scale=0.005, size=simulations)  # 0.5% noise
    
    price_paths[:, t] = price_paths[:, t-1] * np.exp(shocks + volume_noise)
    price_paths[:, t] = np.clip(price_paths[:, t], 0.01, None)

# --- 7. EXTRACT DAILY CLOSES ---
# Map intraday_times to days
day_indices = []
current_day = trading_days[0]
idx_list = []
for i, ts in enumerate(intraday_times):
    if ts.normalize() != current_day:
        day_indices.append(idx_list)
        idx_list = []
        current_day = ts.normalize()
    idx_list.append(i + 1)  # +1 because price_paths includes initial price at 0
day_indices.append(idx_list)  # last day

# Calculate daily closes as last intraday step per trading day
daily_closes = np.array([price_paths[:, indices[-1]] for indices in day_indices]).T  # shape (simulations, days)

# --- 8. DISPLAY METRICS ---
st.markdown("### 📅 Daily Closing Price Percentiles")
percentiles = [5, 50, 95]
percentile_values = {p: np.percentile(daily_closes, p, axis=0) for p in percentiles}

import pandas as pd
df_metrics = pd.DataFrame({
    f"P{p}": percentile_values[p] for p in percentiles
}, index=[f"Day {i+1}" for i in range(len(day_indices))])

st.dataframe(df_metrics.style.format("${:.2f}"))

# Plot percentiles
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

# --- 9. INTRADAY PATHS SAMPLE ---
st.markdown("### 📈 Sample Intraday Price Paths")

sample_paths = price_paths[:min(10, simulations), :]
time_hours = [(ts - intraday_times[0]).total_seconds() / 3600 for ts in intraday_times]
time_hours = [0.0] + time_hours  # initial price at 0

fig2, ax2 = plt.subplots(figsize=(8, 4))
for i in range(sample_paths.shape[0]):
    ax2.plot(time_hours, sample_paths[i], lw=1)
ax2.set_xlabel("Hours since start")
ax2.set_ylabel("Price ($)")
ax2.set_title("Sample Simulated Intraday Price Paths")
st.pyplot(fig2)

# --- 10. DEBUG INFO ---
with st.expander("🧪 Debug Info"):
    st.write({
        "Live Price": live_price,
        "Sentiment Score": sentiment_score,
        "Sentiment Label": sentiment_label,
        "Drift from Sentiment": drift_sentiment,
        "Volatility Estimate": vol_estimate,
        "Volume Average (last 10d)": vol_avg,
        "Adjusted Volatility": volatility_adj,
        "Intraday Steps": intraday_steps_per_day,
        "Forecast Trading Days": len(day_indices),
        "Mu per step": mu,
        "Sigma per step": sigma,
    })
