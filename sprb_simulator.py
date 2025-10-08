import streamlit as st
import yfinance as yf
import numpy as np

st.set_page_config(page_title="SPRB Simulator", layout="centered")
st.title("📈 SPRB Real-Time Price Close Estimator")

# Fetch live price
try:
    ticker = yf.Ticker("SPRB")
    live_price = ticker.history(period="1d")["Close"][-1]
except Exception as e:
    live_price = None

if live_price is not None:
    st.success(f"**Live Price**: ${live_price:.2f}")
else:
    st.error("Could not fetch live price. Check internet / ticker symbol.")

st.markdown("### 🔧 Simulation Settings")
col1, col2 = st.columns(2)
with col1:
    drift = st.slider("Expected Drift (%)", 0.0, 1.0, 0.3, step=0.01)
with col2:
    volatility = st.slider("Expected Volatility (%)", 0.1, 2.0, 0.7, step=0.01)

# Convert slider values from % to decimals
drift /= 100
volatility /= 100

# Monte Carlo simulation
simulations = 10000
shocks = np.random.normal(
    loc=(drift - 0.5 * volatility ** 2),
    scale=volatility,
    size=simulations
)
simulated_prices = live_price * np.exp(shocks)

st.markdown("### 📊 Simulated Closing Price Distribution")
st.metric("5th Percentile", f"${np.percentile(simulated_prices, 5):.2f}")
st.metric("Median", f"${np.median(simulated_prices):.2f}")
st.metric("95th Percentile", f"${np.percentile(simulated_prices, 95):.2f}")

# Optional: histogram
hist_vals, bin_edges = np.histogram(simulated_prices, bins=30)
st.bar_chart(hist_vals)
