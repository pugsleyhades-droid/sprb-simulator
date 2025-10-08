import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SPRB Simulator", layout="centered")
st.title("📈 SPRB Real-Time Price Close Estimator")

# ---------- FETCH LIVE PRICE ----------
ticker = yf.Ticker("SPRB")
try:
    hist = ticker.history(period="10d")
    live_price = hist["Close"][-1]
except Exception:
    st.error("Could not fetch live price. Please check your internet or ticker symbol.")
    st.stop()

if live_price is None or live_price <= 0:
    st.error("Invalid live price.")
    st.stop()

st.success(f"**Live Price**: ${live_price:.2f}")

# ---------- MODE SELECTOR ----------
st.markdown("### 🧭 Mode Selection")
mode = st.radio("Choose Mode", ["Auto (Live Data)", "Manual"], index=0)

# ---------- DRIFT & VOLATILITY ----------
if mode == "Manual":
    drift = st.slider("Expected Drift (%)", -1.0, 1.0, 0.3, step=0.01) / 100
    volatility = st.slider("Expected Volatility (%)", 0.1, 3.0, 0.7, step=0.01) / 100
else:
    hist["Returns"] = hist["Close"].pct_change()
    vol_estimate = hist["Returns"].std()

    # ✅ Fix: Cap or fallback on invalid or extreme vol estimates
    if np.isnan(vol_estimate) or vol_estimate <= 0:
        vol_estimate = 0.7 / 100  # fallback = 0.7%
        st.warning("Volatility estimation failed. Using fallback 0.7%")
    elif vol_estimate > 0.10:
        st.warning(f"⚠️ Raw volatility was {vol_estimate*100:.2f}%, capping to 10%.")
        vol_estimate = 0.10  # cap at 10%

    volatility = vol_estimate
    st.info(f"📊 Using Estimated Volatility: {volatility * 100:.2f}%")

    # Sentiment selection
    sentiment = st.radio("📣 News Sentiment", ["Bullish", "Neutral", "Bearish"], index=0)
    drift_map = {
        "Bullish": 0.02,    # +2% daily drift
        "Neutral": 0.005,   # +0.5%
        "Bearish": -0.01    # -1%
    }
    drift = drift_map[sentiment]
    st.info(f"📈 Drift set to {drift * 100:.2f}% based on sentiment: {sentiment}")

# ---------- MONTE CARLO SIMULATION ----------
T = 1  # 1 day
simulations = 10000

mu = (drift - 0.5 * volatility ** 2) * T
sigma = volatility * np.sqrt(T)

# Generate price paths
shocks = np.random.normal(loc=mu, scale=sigma, size=simulations)
simulated_prices = live_price * np.exp(shocks)

# Clamp minimum to avoid display artifacts
simulated_prices = np.clip(simulated_prices, 0.01, None)

# ---------- DISPLAY METRICS ----------
p5 = np.percentile(simulated_prices, 5)
p50 = np.median(simulated_prices)
p95 = np.percentile(simulated_prices, 95)

st.markdown("### 📊 Simulated Closing Price Distribution")
st.metric("5th Percentile", f"${p5:.2f}")
st.metric("Median", f"${p50:.2f}")
st.metric("95th Percentile", f"${p95:.2f}")

# ---------- HISTOGRAM ----------
st.markdown("### 📉 Price Distribution Histogram")
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(simulated_prices, bins=30, color='skyblue', edgecolor='black')
ax.axvline(p5, color='orange', linestyle='--', label='5th %ile')
ax.axvline(p50, color='red', linestyle='-', label='Median')
ax.axvline(p95, color='green', linestyle='--', label='95th %ile')
ax.set_title("Simulated Closing Price Distribution")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# ---------- DEBUG INFO ----------
with st.expander("🧪 Debug Info"):
    st.write({
        "Live Price": live_price,
        "Drift": drift,
        "Volatility": volatility,
        "Mu (drift term)": mu,
        "Sigma (vol term)": sigma,
        "Shocks (sample)": shocks[:5],
        "Simulated Prices (sample)": simulated_prices[:5]
    })
