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

# --- PAGE SETUP ---
st.set_page_config(page_title="SPRB Advanced Simulator", layout="centered")
st.title("🚀 SPRB Advanced Multi-Day Simulator with Live Sentiment, Market Hours & Target Analysis")

# --- 1. SEARCH STOCK TICKER ---
ticker_input = st.text_input("Enter Stock Ticker (e.g., 'SPRB', 'AAPL')", "SPRB").upper()

# Fetch data for the entered ticker
try:
    ticker = yf.Ticker(ticker_input)
    hist = ticker.history(period="60d")
    live_price = hist["Close"][-1]
    hist_vol = hist["Volume"]
except Exception:
    st.error("Failed to fetch price data. Please check the ticker and try again.")
    st.stop()

if live_price <= 0:
    st.error("Invalid live price data.")
    st.stop()

st.success(f"**Live Price of {ticker_input}:** ${live_price:.2f}")

# --- 2. LIVE NEWS SENTIMENT PULL ---
st.markdown("### 📰 Live News Sentiment")

def fetch_news_and_sentiment(query=ticker_input, days=3):
    try:
        API_KEY = st.secrets["newsapi"]["key"]  # ✅ Securely loaded from secrets.toml
    except Exception:
        st.error("API key not found. Please add it to .streamlit/secrets.toml under [newsapi].")
        st.stop()

    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from={(datetime.utcnow() - timedelta(days=days)).date()}"
        f"&language=en&sortBy=publishedAt&pageSize=20&apiKey={API_KEY}"
    )
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        texts = [a["title"] + ". " + (a.get("description") or "") for a in articles]
    except Exception:
        texts = []
    return texts

def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    if not texts:
        return 0.0
    scores = [analyzer.polarity_scores(text)["compound"] for text in texts]
    return np.mean(scores)

with st.spinner("Fetching latest news..."):
    news_texts = fetch_news_and_sentiment()
    sentiment_score = analyze_sentiment(news_texts)

sentiment_label = "Neutral"
if sentiment_score > 0.05:
    sentiment_label = "Bullish"
elif sentiment_score < -0.05:
    sentiment_label = "Bearish"

st.write(f"**Sentiment Score:** {sentiment_score:.3f} → **{sentiment_label}**")

sentiment_drift_map = {"Bullish": 0.02, "Neutral": 0.005, "Bearish": -0.01}
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
nyse = mcal.get_calendar("NYSE")
today = pd.Timestamp(datetime.now(pytz.UTC)).normalize()
schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=forecast_days * 2))
trading_days = schedule.index[:forecast_days].tolist()

intraday_minutes = 6.5 * 60
intraday_steps_per_day = 13  # Default

# --- 6. SIMULATION ---
mu = (drift_sentiment - 0.5 * volatility_adj**2) * (1 / intraday_steps_per_day)
sigma = volatility_adj * np.sqrt(1 / intraday_steps_per_day)

price_paths = np.zeros((simulations, len(trading_days) * intraday_steps_per_day + 1))
price_paths[:, 0] = live_price
np.random.seed(42)

for t in range(1, len(price_paths[0])):
    shocks = np.random.normal(loc=mu, scale=sigma, size=simulations)
    volume_noise = np.random.normal(loc=0, scale=0.005, size=simulations)
    price_paths[:, t] = price_paths[:, t - 1] * np.exp(shocks + volume_noise)
    price_paths[:, t] = np.clip(price_paths[:, t], 0.01, None)

# --- 7. DAILY CLOSES ---
intraday_times = []
for day in trading_days:
    day_start = day + pd.Timedelta(hours=9, minutes=30)
    times = [
        day_start + pd.Timedelta(minutes=(intraday_minutes / intraday_steps_per_day) * i)
        for i in range(intraday_steps_per_day)
    ]
    intraday_times.extend(times)

# --- 8. PLOTTING SIMULATED PATHS ---
sample_paths = price_paths[:min(10, simulations), :]
average_path = np.mean(price_paths, axis=0)

time_hours = [(ts - intraday_times[0]).total_seconds() / 3600 for ts in intraday_times]
time_hours = [0.0] + time_hours

fig, ax2 = plt.subplots(figsize=(10, 6))
for i in range(min(10, simulations)):
    ax2.plot(time_hours, sample_paths[i], alpha=0.4)
ax2.plot(time_hours, average_path, label="Average Path", color="blue", linewidth=2)
ax2.set_xlabel("Time (Hours)")
ax2.set_ylabel("Price ($)")
ax2.set_title("Sample Intraday Price Paths & Average Path")
ax2.legend(["Paths", "Average"], loc="upper left")
st.pyplot(fig)

# --- 9. PERCENTILES ---
percentiles = [5, 50, 95]
daily_closes = np.array(
    [price_paths[:, i * intraday_steps_per_day + intraday_steps_per_day - 1]
     for i in range(len(trading_days))]).T
percentile_values = {p: np.percentile(daily_closes, p, axis=0) for p in percentiles}

df_metrics = pd.DataFrame({f"P{p}": percentile_values[p] for p in percentiles},
                          index=[f"Day {i+1}" for i in range(len(trading_days))])

st.markdown("### 📅 Daily Closing Price Percentiles")
st.dataframe(df_metrics.style.format("${:.2f}"))

fig1, ax1 = plt.subplots(figsize=(8, 4))
days = np.arange(1, len(trading_days) + 1)
ax1.plot(days, percentile_values[5], label="5th Percentile", linestyle="--", color="orange")
ax1.plot(days, percentile_values[50], label="Median", color="red")
ax1.plot(days, percentile_values[95], label="95th Percentile", linestyle="--", color="green")
ax1.set_xlabel("Trading Day")
ax1.set_ylabel("Price ($)")
ax1.set_title("Simulated Daily Closing Price Percentiles")
ax1.legend()
st.pyplot(fig1)

# --- 10. LIMIT SELL OPTIMIZER ---
st.markdown("## 🎯 Limit-Sell Probability Optimizer ($200–$230)")

targets = np.arange(200, 235, 5)
intraday_max = np.max(price_paths, axis=1)

results = []
for t_price in targets:
    prob_hit = np.mean(intraday_max >= t_price)
    results.append({"Target ($)": t_price, "P(Touch Intraday)": prob_hit})

df_targets = pd.DataFrame(results)
df_targets["P(Touch Intraday)"] = df_targets["P(Touch Intraday)"].apply(lambda x: f"{x*100:.1f}%")
st.dataframe(df_targets)

best_target = df_targets.iloc[np.argmax([float(p.strip('%')) for p in df_targets["P(Touch Intraday)"]])]
st.success(f"💡 Optimal Limit Price (best balance): **${best_target['Target ($)']}** with ~{best_target['P(Touch Intraday)']} chance to hit intraday.")

# --- INFO BLOCK ---
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This simulator combines **historical volatility**, **live sentiment**, and **Monte Carlo path modeling**  
    to project possible short-term stock behavior for **high-volatility tickers** like SPRB.

    **Key components:**
    - Live sentiment drift (Bullish, Neutral, Bearish)
    - Intraday simulation over market hours
    - Probabilities for target prices ($200–$230)
    - 5th–95th percentile forecasts

    ---
    💡 **Tip:** You can securely store your API keys by adding them to `.streamlit/secrets.toml`:
    ```toml
    [newsapi]
    key = "your_actual_api_key"
    ```
    Then access it with `st.secrets["newsapi"]["key"]` instead of hardcoding.
    """)
