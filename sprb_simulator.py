```python
# SPRB Advanced Simulator (updated)
# - Fixes ordering bug (intraday steps chosen before simulation)
# - Adds fat-tailed Student-t increments option
# - Adds start-price override (pre-market)
# - Computes intraday-touch probabilities for target grid (200-230 step 5)
# - Computes expected ladder fills/proceeds
# - Keep NewsAPI (optional) and VADER sentiment

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

st.set_page_config(page_title="SPRB Advanced Simulator", layout="wide")
st.title("🚀 SPRB Advanced Simulator — Updated")

# -------------------------
# 0. Sidebar controls
# -------------------------
st.sidebar.header("Simulation Controls")
ticker_input = st.sidebar.text_input("Ticker", "SPRB").upper()
forecast_days = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=30, value=2)
simulations = st.sidebar.number_input("Simulations", min_value=100, max_value=200000, value=50000, step=100)
intraday_steps_per_day = st.sidebar.selectbox(
    "Intraday steps/day",
    options=[1, 4, 13, 26, 52],
    index=2,
    format_func=lambda x: f"{x} steps/day (~{round(6.5*60/x)} min each)"
)
use_fat_tails = st.sidebar.checkbox("Use fat-tailed Student-t increments", value=True)
student_df = st.sidebar.slider("Student-t df (if fat tails)", min_value=2, max_value=10, value=3)

# optional override for start price (useful pre-market)
override_price = st.sidebar.number_input("Override start price (0 = use live)", min_value=0.0, value=0.0, step=0.01)

# news API key (optional)
news_api_key = st.sidebar.text_input("NewsAPI key (optional)", value="", type="password")

# target grid for intraday touch probabilities
targets = st.sidebar.multiselect("Targets to evaluate (intraday touch)", [200,205,210,215,220,225,230], default=[200,205,210,215,220,225,230])

# limit ladder example weights (fractions summing to 1)
st.sidebar.markdown("### Limit ladder example (fractions sum to 1)")
l1 = st.sidebar.number_input("Frac @ first limit", value=0.4, min_value=0.0, max_value=1.0, step=0.05)
l2 = st.sidebar.number_input("Frac @ second limit", value=0.3, min_value=0.0, max_value=1.0, step=0.05)
l3 = st.sidebar.number_input("Frac @ third limit", value=0.2, min_value=0.0, max_value=1.0, step=0.05)
l4 = st.sidebar.number_input("Frac @ fourth limit", value=0.1, min_value=0.0, max_value=1.0, step=0.05)
ladder_fracs = [l1, l2, l3, l4]
# normalize if needed
s = sum(ladder_fracs)
if s == 0:
    ladder_fracs = [0.4,0.3,0.2,0.1]
else:
    ladder_fracs = [f/s for f in ladder_fracs]

# -------------------------
# 1. Fetch price and history
# -------------------------
@st.cache_data(ttl=60)
def get_hist(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="90d", interval="1d")
    return hist

try:
    hist = get_hist(ticker_input)
    if hist is None or hist.empty:
        st.error("No historical data found for that ticker.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching ticker data: {e}")
    st.stop()

live_price = float(hist["Close"].iloc[-1])
start_price = override_price if override_price > 0 else live_price

st.metric("Live (or override) price", f"${start_price:.2f}")

# -------------------------
# 2. Sentiment (optional)
# -------------------------
def fetch_news_texts(query, days=3, key="9011c7b1e87c4c7aa0b63dcda687916a"):
    if not key:
        return []
    url = f"https://newsapi.org/v2/everything?q={query}&from={(datetime.utcnow() - timedelta(days=days)).date()}&language=en&sortBy=publishedAt&pageSize=50&apiKey={key}"
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        articles = data.get("articles", [])
        texts = [a.get("title","") + ". " + (a.get("description") or "") for a in articles]
    except Exception:
        texts = []
    return texts

def sentiment_score_from_texts(texts):
    if not texts:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts]
    return float(np.mean(scores))

texts = fetch_news_texts(ticker_input, days=3, key=news_api_key)
sent_score = sentiment_score_from_texts(texts)
if sent_score > 0.05:
    sentiment_label = "Bullish"
elif sent_score < -0.05:
    sentiment_label = "Bearish"
else:
    sentiment_label = "Neutral"

st.write(f"Sentiment ({sentiment_label}): {sent_score:.3f}")

# map sentiment to drift
sentiment_drift_map = {"Bullish": 0.02, "Neutral": 0.005, "Bearish": -0.01}
drift_sentiment = sentiment_drift_map.get(sentiment_label, 0.005)

# -------------------------
# 3. Volatility estimate
# -------------------------
hist['Returns'] = hist['Close'].pct_change()
vol_est = hist['Returns'].std()
if np.isnan(vol_est) or vol_est <= 0:
    vol_est = 0.02
vol_est = min(max(vol_est, 0.005), 0.50)  # clamp to reasonable biotech extremes

# small volume adjustment (optional)
vol_adj = vol_est

st.info(f"Estimated daily volatility: {vol_adj*100:.2f}% ; Using drift {drift_sentiment*100:.2f}% daily")

# -------------------------
# 4. Market calendar -> trading days
# -------------------------
nyse = mcal.get_calendar('NYSE')
today = pd.Timestamp(datetime.now(pytz.UTC)).normalize()
schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=forecast_days*2))
trading_days = schedule.index[:forecast_days].tolist()
if len(trading_days) == 0:
    st.error("No upcoming trading days found (market closed?)")
    st.stop()

# -------------------------
# 5. Simulation (vectorized)
# -------------------------
total_steps = len(trading_days) * intraday_steps_per_day
st.write(f"Simulating {simulations:,} paths × {total_steps} steps ({intraday_steps_per_day} steps/day × {len(trading_days)} days) ...")

# set seeds for reproducibility
np.random.seed(42)

# per-step params (geometric Brownian approx)
mu_step = (drift_sentiment - 0.5 * vol_adj**2) / intraday_steps_per_day
sigma_step = vol_adj / np.sqrt(intraday_steps_per_day)

# generate shocks
if use_fat_tails:
    # Student-t scaled to have approx std = 1, then scaled by sigma_step
    t_raw = np.random.standard_t(df=student_df, size=(simulations, total_steps))
    t_std = np.sqrt(student_df / (student_df - 2)) if student_df > 2 else 1.0
    shocks = (t_raw / t_std) * sigma_step + mu_step
else:
    shocks = np.random.normal(loc=mu_step, scale=sigma_step, size=(simulations, total_steps))

# small volume noise to add microstructure
volume_noise = np.random.normal(loc=0.0, scale=0.002, size=(simulations, total_steps))
log_increments = shocks + volume_noise

# cumulative product
log_prices = np.cumsum(log_increments, axis=1)
price_paths = start_price * np.exp(log_prices)
# prepend start price column for plotting convenience
price_paths = np.concatenate([np.full((simulations,1), start_price), price_paths], axis=1)

# compute intraday maxima for each simulated day
# for convenience, break into day-chunks
intraday_max = []
intraday_close = []
for d in range(len(trading_days)):
    start_idx = d*intraday_steps_per_day + 1
    end_idx = (d+1)*intraday_steps_per_day + 1
    chunk = price_paths[:, start_idx:end_idx]
    intraday_max.append(chunk.max(axis=1))
    intraday_close.append(chunk[:, -1])
intraday_max = np.vstack(intraday_max).T  # shape sims x days
intraday_close = np.vstack(intraday_close).T

# For 2-day intraday-touch probability, check max across both days:
max_over_horizon = intraday_max.max(axis=1)

# -------------------------
# 6. Compute target touch probabilities (over whole horizon)
# -------------------------
targets_sorted = sorted(targets)
touch_probs = {t: float(np.mean(max_over_horizon >= t)) for t in targets_sorted}
close_probs = {t: float(np.mean(intraday_close[:, -1] >= t)) for t in targets_sorted}  # last day close

st.markdown("### 🔎 Intraday touch probabilities (over horizon)")
df_touch = pd.DataFrame({
    "Target ($)": targets_sorted,
    "P(touch intraday across horizon)": [touch_probs[t] for t in targets_sorted],
    "P(close ≥ target on last day)": [close_probs[t] for t in targets_sorted]
})
df_touch["P(touch intraday across horizon)"] = df_touch["P(touch intraday across horizon)"].map("{:.2%}".format)
df_touch["P(close ≥ target on last day)"] = df_touch["P(close ≥ target on last day)"].map("{:.2%}".format)
st.table(df_touch)

# -------------------------
# 7. Limit-ladder expected fills
# -------------------------
# compute fill probability per limit price as P(horizon max >= limit)
limits = np.array([200,205,210,215,220,225,230])
fill_probs = {L: float(np.mean(max_over_horizon >= L)) for L in limits}

# user ladder - choose four price levels in the 200-230 range automatically
ladder_prices = [205, 210, 220, 230]
shares_input = st.number_input("Shares owned", value=41)
cost_input = st.number_input("Total cost ($)", value=5683.0)
shares = shares_input

ladder_df = []
for p, frac in zip(ladder_prices, ladder_fracs):
    p_fill = fill_probs.get(p, 0.0)
    expected_shares_filled = shares * frac * p_fill
    expected_proceeds = expected_shares_filled * p
    ladder_df.append({
        "limit": p,
        "fraction_of_position": frac,
        "fill_prob": p_fill,
        "expected_shares_filled": expected_shares_filled,
        "expected_proceeds_$": expected_proceeds
    })
ladder_df = pd.DataFrame(ladder_df)
ladder_df["fill_prob"] = ladder_df["fill_prob"].map("{:.2%}".format)
ladder_df["expected_proceeds_$"] = ladder_df["expected_proceeds_$"].map("${:,.2f}".format)
st.markdown("### Ladder expected fills / proceeds (over horizon)")
st.table(ladder_df)

# Ladder summary
total_expected_proceeds = sum([float(r.replace("$","").replace(",","")) for r in ladder_df["expected_proceeds_$"]])
expected_profit = total_expected_proceeds - cost_input
st.metric("Expected proceeds from ladder (over horizon)", f"${total_expected_proceeds:,.2f}", delta=f"${expected_profit:,.2f}")

# -------------------------
# 8. Percentiles / plots for closing prices by day
# -------------------------
daily_closes = np.array([intraday_close[:, i] for i in range(len(trading_days))]).T
percentiles = [5,50,95]
percentile_values = {p: np.percentile(daily_closes, p, axis=0) for p in percentiles}

df_metrics = pd.DataFrame({f"P{p}": percentile_values[p] for p in percentiles}, index=[f"Day {i+1}" for i in range(len(trading_days))])
st.markdown("### 📅 Daily closing percentiles")
st.dataframe(df_metrics.style.format("${:.2f}"))

# plot sample paths (small number for viz)
sample_n = min(50, int(st.sidebar.number_input("Sample paths to plot", value=10, min_value=1, max_value=100)))
fig, ax = plt.subplots(figsize=(10,5))
for i in range(sample_n):
    ax.plot(price_paths[i,:], alpha=0.6)
ax.plot(np.mean(price_paths, axis=0), color='black', linewidth=2, label='Mean path')
ax.set_title("Sample simulated intraday paths (first paths)")
ax.set_xlabel("Step")
ax.set_ylabel("Price ($)")
ax.legend()
st.pyplot(fig)

# histogram of final day closing
final_closes = daily_closes[:, -1]
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.hist(final_closes, bins=120, density=True)
ax2.axvline(cost_input / shares if shares>0 else 0, color='red', linestyle='--', label='Your per-share cost')
ax2.set_title("Histogram of closing prices (last day of horizon)")
ax2.set_xlabel("Price")
ax2.set_ylabel("Density")
ax2.legend()
st.pyplot(fig2)

# -------------------------
# 9. Notes / warnings
# -------------------------
st.info("""
**Notes & cautions**
- This is a Monte-Carlo simulator (stochastic). Results are probabilistic, not deterministic.
- Fat-tailed Student-t shocks allow big spikes but are still a model — real market microstructure, halts, and block trades can produce different outcomes.
- Do not paste private API keys into shared code. Use environment variables for production.
""")
```

    - Parameters: mu = average return based on sentiment, volatility adjusted by market volume

    Adjust the number of days and intraday resolution above to explore different outcomes.
    """)
