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
st.title("🚀 Advanced Multi-Day Simulator with Live Sentiment & Market Hours")

# --- 0. SEARCH FOR TICKER BY NAME OR SYMBOL ---
def search_tickers(query):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        response = requests.get(url)
        results = response.json().get('quotes', [])
        # Filter equities only to avoid crypto etc
        equities = [r for r in results if r.get('quoteType') == 'EQUITY']
        # return list of (symbol, shortname)
        return [(eq['symbol'], eq.get('shortname', eq['symbol'])) for eq in equities]
    except Exception as e:
        st.error(f"Search API error: {e}")
        return []

search_input = st.text_input("Search by Ticker Symbol or Company Name", value="SPRB").strip()

if not search_input:
    st.warning("Please enter a company name or ticker symbol to search.")
    st.stop()

matches = search_tickers(search_input)

if not matches:
    st.error(f"No tickers found for '{search_input}'. Try another search.")
    st.stop()

options = [f"{sym} - {name}" for sym, name in matches]
selected = st.selectbox("Select the correct stock:", options)

ticker_symbol = selected.split(" - ")[0]

# --- 1. LIVE PRICE & HISTORICAL DATA ---
ticker = yf.Ticker(ticker_symbol)
try:
    hist = ticker.history(period="60d")
    if hist.empty:
        st.error(f"No historical data found for {ticker_symbol}. Please check the ticker and try again.")
        st.stop()
    live_price = hist["Close"][-1]
except Exception as e:
    st.error(f"Failed to fetch price data for {ticker_symbol}: {e}")
    st.stop()

if live_price <= 0:
    st.error("Invalid live price data.")
    st.stop()

st.success(f"**Live Price for {ticker_symbol}:** ${live_price:.2f}")

# --- 2. LIVE NEWS SENTIMENT PULL ---
st.markdown("### 📰 Live News Sentiment")

def fetch_news_and_sentiment(query=ticker_symbol, days=3):
    API_KEY = "9011c7b1e87c4c7aa0b63dcda687916a"  # Replace with your actual key
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

vol_avg = hist["Volume"][-10:].mean()
vol_norm = vol_avg / hist["Volume"].mean()
volatility_adj = vol_estimate * (1 + (vol_norm - 1) * 0.2)

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
trading_days = schedule.index[:forecast_days].tolist()

intraday_minutes = 6.5 * 60
minutes_per_step = intraday_minutes / intraday_steps_per_day

intraday_times = []
for day in trading_days:
    day_start = day + pd.Timedelta(hours=9, minutes=30)
    times = [day_start + pd.Timedelta(minutes=minutes_per_step * i) for i in range(intraday_steps_per_day)]
    intraday_times.extend(times)

total_steps = len(intraday_times)
dt = 1 / intraday_steps_per_day

# --- 6. SIMULATION ---
mu = (drift_sentiment - 0.5 * volatility_adj ** 2) * dt
sigma = volatility_adj * np.sqrt(dt)

price_paths = np.zeros((simulations, total_steps + 1))
price_paths[:, 0] = live_price
np.random.seed(42)

for t in range(1, total_steps + 1):
    shocks = np.random.normal(loc=mu, scale=sigma, size=simulations)
    volume_noise = np.random.normal(loc=0, scale=0.005, size=simulations)
    price_paths[:, t] = price_paths[:, t-1] * np.exp(shocks + volume_noise)
    price_paths[:, t] = np.clip(price_paths[:, t], 0.01, None)

# --- 7. DAILY CLOSES ---
day_indices = []
current_day = trading_days[0]
idx_list = []
for i, ts in enumerate(intraday_times):
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
days = range(1, len(day_indices) + 1)
ax.plot(days, percentile_values[5], label='5th Percentile', linestyle='--', color='orange')
ax.plot(days, percentile_values[50], label='Median', linestyle='-', color='red')
ax.plot(days, percentile_values[95], label='95th Percentile', linestyle='--', color='green')
ax.set_xlabel("Trading Day")
ax.set_ylabel("Price ($)")
ax.set_title(f"Simulated Daily Closing Price Percentiles for {ticker_symbol}")
ax.legend()
st.pyplot(fig)

# --- EXPLANATION BLOCK ---
with st.expander("ℹ️ What Are Sample Intraday Price Paths?"):
    st.markdown("""
    These lines represent **simulated price movements during market hours** across forecasted trading days.

    Each line shows one **possible intraday scenario** generated using:

    - 📈 **Starting Price**: The current live market price
    - 🔄 **Volatility**: Based on historical price swings and volume
    - 🧠 **Drift (trend)**: From live news sentiment (Bullish/Neutral/Bearish)
    - ⏰ **Market hours only**: 9:30 AM - 4:00 PM ET
    - 🎲 **Randomness**: Reflects unpredictable market movements and volume noise

    ---

    ### Technical Notes:
    - `mu`: Expected return per time step (based on sentiment)
    - `sigma`: Volatility per step (from historical data)
    - `dt`: Time step size (e.g., 1/13 for 30-minute intervals)
    - Step formula: `Price[t] = Price[t-1] * exp(mu + randomness)`
    - 10 paths shown out of 1000 total simulations

    Adjust days and intraday steps above to explore different outcomes.
    """)

# --- 9. INTRADAY PATHS SAMPLE + AVERAGE ---

st.markdown("### 📈 Sample Intraday Price Paths + Average")

sample_paths = price_paths[:min(10, simulations), :]
average_path = np.mean(price_paths, axis=0)

time_hours = [(ts - intraday_times[0]).total_seconds() / 3600 for ts in intraday_times]
time_hours = [0.0] + time_hours  # Include starting point at 0

fig2, ax2 = plt.subplots(figsize=(8, 4))

for i in range(sample_paths.shape[0]):
    ax2.plot(time_hours, sample_paths[i], lw=1, alpha=0.525, label=f'Path {i+1}')

ax2.plot(time_hours, average_path, lw=2.5, color='black', linestyle='-', label='Average Path')

ax2.set_xlabel("Hours since simulation start")
ax2.set_ylabel("Simulated Price ($)")
ax2.set_title(f"Sample Intraday Price Paths + Average for {ticker_symbol}")
ax2.legend(loc='upper left', fontsize='x-small', ncol=2)
ax2.grid(True)

st.pyplot(fig2)

# --- 10. DEBUG INFO ---
with st.expander("🧪 Debug Info"):
    st.write({
        "Ticker": ticker_symbol,
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
