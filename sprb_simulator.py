import streamlit as st
import yfinance as yf
import numpy as np

st.set_page_config(page_title="SPRB Close Price Simulator", layout="centered")

st.title("📈 SPRB Real-Time Price Close Estimator")

# Get live price from Yahoo Finance
try:
    ticker = yf.Ticker("SPRB")
    live_price = ticker.history(period="1d")["Close"][-1]
except Exception:
    live_price = None

if live_price:
    st.success(f"**Live Price**: ${live_price:.2f}")
else:
    st.error("Could not fetch live price. Please check your internet connection or try again later.")

st.markdown("### ➕ Simulation Settings")
col1, col2 = st.columns(2)

with col1:
    drift = st.slider("📈 Expected Upward Drift (%)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
with col2:
    volatility = st.slider("📊 Expected Volatility (%)", min_value=0.1, max_value=2.0, value=0.7, step=0.05)

# Convert percentages to decimal
drift /= 100
volatility /= 100

# Run Monte Carlo simulation
simulations = 10000
shocks = np.random.normal(loc=(drift - 0.5 * volatility**2), scale=volatility, size=simulations)
simulated_prices = live_price * np.exp(shocks)

# Display metrics
st.markdown("### 📊 Simulated Closing Price Distribution (10,000 trials)")
st.metric("🔻 5th Percentile", f"${np.percentile(simulated_prices, 5):.2f}")
st.metric("🔸 Median", f"${np.median(simulated_prices):.2f}")
st.metric("🔺 95th Percentile", f"${np.percentile(simulated_prices, 95):.2f}")

# Optional histogram
st.markdown("### 📉 Distribution Chart")
st.bar_chart(np.histogram(simulated_prices, bins=30)[0])
