import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPRB Simulator", layout="centered")
st.title("📈 SPRB Real-Time Price Close Estimator")

# Fetch live price
ticker = yf.Ticker("SPRB")
try:
    hist = ticker.history(period="10d")
    live_price = hist["Close"][-1]
except Exception as e:
    live_price = None

if live_price is not None:
    st.success(f"**Live Price**: ${live_price:.2f}")
else:
    st.error("Could not fetch live price. Please check internet / ticker symbol.")

st.markdown("### 🧭 Mode Selection")
mode = st.radio("Choose Mode", ["Auto (Live Data)", "Manual"], index=0)

if mode == "Manual":
    # Manual mode: user picks drift & volatility
    drift = st.slider("Expected Drift (%)", 0.0, 1.0, 0.3, step=0.01) / 100
    volatility = st.slider("Expected Volatility (%)", 0.1, 2.0, 0.7, step=0.01)
    volatility /= 100  # convert to decimal
else:
    # Auto mode: estimate volatility from recent data
    hist["Returns"] = hist["Close"].pct_change()
    vol_estimate = hist["Returns"].std()
    if np.isnan(vol_estimate) or vol_estimate <= 0:
        # fallback
        vol_estimate = 0.7 / 100  # e.g. 0.7%
    st.info(f"📊 Estimated Volatility (10-day): {vol_estimate * 100:.2f}%")
    volatility = vol_estimate

    # Sentiment-based drift
    sentiment = st.radio("📣 News Sentiment", ["Bullish", "Neutral", "Bearish"], index=0)
    drift_values = {
        "Bullish": 0.005,    # +0.5% per day
        "Neutral": 0.001,    # +0.1% per day
        "Bearish": -0.002    # –0.2% per day
    }
    drift = drift_values[sentiment]
    st.info(f"📈 Drift set to {drift * 100:.2f}% based on sentiment: {sentiment}")

# If live_price is None, stop further simulation
if live_price is None:
    st.stop()

# Monte Carlo simulation
simulations = 10000
shocks = np.random.normal(
    loc=(drift - 0.5 * volatility ** 2),
    scale=volatility,
    size=simulations
)
simulated_prices = live_price * np.exp(shocks)

# Display metrics
st.markdown("### 📊 Simulated Closing Price Distribution")
st.metric("5th Percentile", f"${np.percentile(simulated_prices, 5):.2f}")
st.metric("Median", f"${np.median(simulated_prices):.2f}")
st.metric("95th Percentile", f"${np.percentile(simulated_prices, 95):.2f}")

# Histogram with confidence lines
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(simulated_prices, bins=30, color='skyblue', edgecolor='black')
ax.axvline(np.percentile(simulated_prices, 5), color='orange', linestyle='--', label='5th %ile')
ax.axvline(np.median(simulated_prices), color='red', linestyle='-', label='Median')
ax.axvline(np.percentile(simulated_prices, 95), color='green', linestyle='--', label='95th %ile')
ax.set_title("Simulated Close Price Distribution")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)
