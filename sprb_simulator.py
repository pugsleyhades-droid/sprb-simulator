import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPRB Debug Simulator", layout="centered")
st.title("🔍 SPRB Debug Simulator")

# Debug step 1
st.write("DEBUG: Starting app")

# Fetch live price
try:
    ticker = yf.Ticker("SPRB")
    hist = ticker.history(period="10d")
    st.write("DEBUG: History data", hist.tail())
    live_price = hist["Close"][-1]
    st.write("DEBUG: live_price:", live_price)
except Exception as e:
    st.error(f"Error fetching price or hist: {e}")
    st.stop()

if live_price is None or live_price <= 0:
    st.error("Invalid live price fetched; stopping.")
    st.stop()

# Mode selector
mode = st.radio("Choose Mode", ["Auto (Live Data)", "Manual"], index=0)

# Drift & Volatility determination
try:
    if mode == "Manual":
        drift = st.slider("Expected Drift (%)", -1.0, 1.0, 0.3, step=0.01) / 100
        volatility = st.slider("Expected Volatility (%)", 0.1, 3.0, 0.7, step=0.01) / 100
    else:
        hist["Returns"] = hist["Close"].pct_change()
        st.write("DEBUG: Returns", hist["Returns"].tail())
        vol_estimate = hist["Returns"].std()
        st.write("DEBUG: vol_estimate (std)", vol_estimate)

        if np.isnan(vol_estimate) or vol_estimate <= 0:
            vol_estimate = 0.7 / 100  # fallback
            st.write("DEBUG: vol_estimate fallback used")

        st.info(f"Estimated Volatility: {vol_estimate * 100:.2f}%")
        volatility = vol_estimate

        sentiment = st.radio("News Sentiment", ["Bullish", "Neutral", "Bearish"], index=0)
        drift_map = {
            "Bullish": 0.02,
            "Neutral": 0.005,
            "Bearish": -0.01
        }
        drift = drift_map[sentiment]
        st.info(f"Drift = {drift * 100:.2f}% for sentiment: {sentiment}")
except Exception as e:
    st.error(f"Error in drift/vol logic: {e}")
    st.stop()

# Simulation block
try:
    T = 1
    simulations = 10000
    mu = (drift - 0.5 * volatility ** 2) * T
    sigma = volatility * np.sqrt(T)
    st.write("DEBUG: mu, sigma:", mu, sigma)
    shocks = np.random.normal(loc=mu, scale=sigma, size=simulations)
    st.write("DEBUG: shocks sample", shocks[:5])
    simulated_prices = live_price * np.exp(shocks)
    simulated_prices = np.clip(simulated_prices, 0.01, None)
    st.write("DEBUG: sim_prices sample", simulated_prices[:5])
except Exception as e:
    st.error(f"Error in simulation: {e}")
    st.stop()

# Display metrics
p5 = np.percentile(simulated_prices, 5)
p50 = np.median(simulated_prices)
p95 = np.percentile(simulated_prices, 95)

st.metric("5th Percentile", f"${p5:.2f}")
st.metric("Median", f"${p50:.2f}")
st.metric("95th Percentile", f"${p95:.2f}")

# Histogram
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(simulated_prices, bins=30, color='skyblue', edgecolor='black')
ax.axvline(p5, color='orange', linestyle='--', label='5th %ile')
ax.axvline(p50, color='red', linestyle='-', label='Median')
ax.axvline(p95, color='green', linestyle='--', label='95th %ile')
ax.legend()
st.pyplot(fig)
