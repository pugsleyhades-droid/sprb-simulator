# sprb_advanced_full.py
# Advanced SPRB simulator: multi-source sentiment, event-analogs, transformer/time-series forecasting,
# fundamental signals, risk-adjusted weighting, and combined confidence forecast.
#
# Requirements (see README in this chat): sentence-transformers, pytrends, torch, transformers, xgboost, lightgbm, faiss-cpu
# Put your NewsAPI key into .streamlit/secrets.toml as:
# [newsapi]
# key = "YOUR_KEY"

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pytz
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pytrends.request import TrendReq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import faiss
import torch
import torch.nn as nn
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas_market_calendars as mcal
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="SPRB — AI + Analogs Forecast", layout="wide")
st.title("SPRB — Full AI Forecast Suite (Sentiment, Analogs, Transformer, Fundamentals)")

# ---------------------------
# --- Helper / config
# ---------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast SBERT model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL, device=DEVICE)

embed_model = load_embedding_model()

def load_newsapi_texts(query, days=7, page_size=50):
    """Fetch news headlines & descriptions via NewsAPI (requires secret key)."""
    try:
        key = st.secrets["newsapi"]["key"]
    except Exception:
        st.warning("NewsAPI key not found in secrets. Live news disabled.")
        return []
    from_date = (datetime.utcnow() - timedelta(days=days)).date()
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from={from_date}&language=en&sortBy=publishedAt&pageSize={page_size}&apiKey={key}"
    )
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        articles = data.get("articles", [])
        texts = []
        for a in articles:
            t = (a.get("title") or "") + ". " + (a.get("description") or "")
            texts.append(t)
        return texts
    except Exception as e:
        st.error(f"News fetch failed: {e}")
        return []

def fetch_reddit_titles(keyword, limit=50):
    """Lightweight Reddit scraping via Pushshift or redditsearch.io fallback.
       This is a simple approach—rate limits apply. Use credentials for production."""
    try:
        url = f"https://api.pushshift.io/reddit/search/submission/?q={keyword}&size={limit}"
        r = requests.get(url, timeout=6).json()
        titles = [item.get("title","") for item in r.get("data",[])]
        return titles
    except Exception:
        # fallback scraping of redditsearch.io preview
        try:
            r2 = requests.get(f"https://www.redditsearch.io/search?q={keyword}", timeout=6)
            soup = BeautifulSoup(r2.text, "html.parser")
            posts = [p.text for p in soup.select(".post-title")]
            return posts[:limit]
        except Exception:
            return []

def google_trends_momentum(keyword, days=7):
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        timeframe = f"now {days}-d"
        pytrends.build_payload([keyword], timeframe=timeframe)
        df = pytrends.interest_over_time()
        if df.empty:
            return 0.0
        series = df[keyword].astype(float)
        # simple momentum: (last - first) / mean
        return float((series.iloc[-1] - series.iloc[0]) / (series.mean() + 1e-9))
    except Exception:
        return 0.0

def compute_sentiment_score(texts):
    analyzer = SentimentIntensityAnalyzer()
    if not texts:
        return 0.0
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    return float(np.mean(scores))

# ---------------------------
# --- User inputs
# ---------------------------
ticker = st.text_input("Ticker", value="SPRB").upper().strip()
start_price_override = st.number_input("Override start price (0 = use live)", min_value=0.0, step=0.01, value=0.0)
forecast_days = st.slider("Forecast horizon (days)", 1, 20, 5)
simulations = st.number_input("Monte Carlo simulations", min_value=500, max_value=20000, value=2000, step=500)
use_transformer_forecast = st.checkbox("Use Transformer time-series model (slow; experimental)", value=False)
show_diagnostics = st.checkbox("Show diagnostics & intermediate outputs", value=False)
run_btn = st.button("Run full forecast")

# ---------------------------
# --- Fetch market & fundamentals
# ---------------------------
@st.cache_data(ttl=60*30)
def get_hist_and_financials(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="180d", interval="1d")
    info = tk.info
    # financial statements (may be empty for smallcaps)
    fin = {}
    try:
        fin['financials'] = tk.financials
        fin['balance_sheet'] = tk.balance_sheet
        fin['cashflow'] = tk.cashflow
    except Exception:
        fin = {}
    return hist, info, fin

hist, info, fin = get_hist_and_financials(ticker)
if hist is None or hist.empty:
    st.error("No historical data — check ticker.")
    st.stop()

live_price = hist["Close"].iloc[-1]
start_price = start_price_override if start_price_override > 0 else live_price
st.metric("Start price (used)", f"${start_price:.2f}")

# simple fundamentals / fair value heuristic
def compute_fundamental_score(info, fin):
    """
    Returns a simple normalized fundamental score in [-1, +1].
    Uses cash on hand, market cap, revenue (if available), and cashflow.
    This is a heuristic — you can replace with more advanced DCF later.
    """
    try:
        market_cap = info.get("marketCap") or 0.0
        cash = info.get("totalCash") or 0.0
        total_debt = info.get("totalDebt") or 0.0
        revenue = info.get("totalRevenue") or 0.0
        # cash runway proxy
        if cash > 0 and revenue > 0:
            cash_runway = cash / max(1, revenue/4)  # quarters heuristic
        else:
            cash_runway = 0.0
        # free-cashflow proxy
        fcf = 0.0
        if "cashflow" in fin and not fin["cashflow"].empty:
            cf = fin["cashflow"].loc["Total Cash From Operating Activities"].iloc[0] if "Total Cash From Operating Activities" in fin["cashflow"].index else 0
            fcf = cf
        # normalize
        score = 0.0
        score += np.tanh((cash - total_debt) / (1e6))  # scaled
        score += np.tanh((revenue - 1e6) / (1e6))
        score += np.tanh(cash_runway / 4)  # years
        # clamp to [-1,1]
        return float(np.clip(score/3.0, -1, 1))
    except Exception:
        return 0.0

fund_score = compute_fundamental_score(info, fin)
st.write(f"Fundamental signal (heuristic): {fund_score:.3f}")

# ---------------------------
# --- Multi-source sentiment & event texts
# ---------------------------
with st.spinner("Gathering multi-source sentiment..."):
    news_texts = load_newsapi_texts(ticker, days=7)
    reddit_texts = fetch_reddit_titles(ticker, limit=80)
    trends_m = google_trends_momentum(ticker, days=7)
    news_score = compute_sentiment_score(news_texts)
    reddit_score = compute_sentiment_score(reddit_texts)
    # combine with simple weights
    weights = np.array([0.6, 0.25, 0.15])
    composite_sentiment = np.dot(weights, np.array([news_score, reddit_score, trends_m]))
    st.write(f"Composite sentiment: news={news_score:.3f}, reddit={reddit_score:.3f}, trends={trends_m:.3f} -> composite={composite_sentiment:.3f}")

# ---------------------------
# --- Event embeddings & analog matching
# ---------------------------
# Local "historical events" DB: a CSV with columns: date, ticker, title, type, notes, outcome_return_7d, outcome_return_30d
# We'll attempt to load it if present; otherwise we create an empty frame and show how to populate.
EVENTS_CSV = "event_analogs_db.csv"

@st.cache_data(ttl=60*60)
def load_event_db(path=EVENTS_CSV):
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    except Exception:
        # return empty template
        cols = ["date","ticker","title","type","notes","outcome_return_7d","outcome_return_30d"]
        return pd.DataFrame(columns=cols)

event_db = load_event_db(EVENTS_CSV)
st.write(f"Loaded {len(event_db)} historical events for analogue matching (file: {EVENTS_CSV})")

# embed current news headlines (concatenate top N)
current_event_text = ""
if news_texts:
    current_event_text = " || ".join(news_texts[:6])
else:
    current_event_text = f"{ticker} news {datetime.utcnow().date()}"

# embed database titles for nearest neighbor search
def build_event_index(df, embed_model):
    texts = df["title"].fillna("").tolist()
    if not texts:
        return None, None
    embeddings = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    # use FAISS index for speed
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.array(embeddings).astype("float32"))
    return index, np.array(embeddings, dtype="float32")

if not event_db.empty:
    event_index, event_embeddings = build_event_index(event_db, embed_model)
else:
    event_index, event_embeddings = None, None

# compute current embedding
cur_emb = embed_model.encode([current_event_text], normalize_embeddings=True)[0].astype("float32")

analogs = []
if event_index is not None:
    # search nearest 5 analogs
    D, I = event_index.search(np.expand_dims(cur_emb, axis=0), k=min(6, len(event_db)))
    inds = I[0].tolist()
    for idx in inds:
        row = event_db.iloc[idx].to_dict()
        row["similarity"] = float(D[0][inds.index(idx)])
        analogs.append(row)

if analogs:
    st.markdown("### Top analog events (closest historical matches)")
    st.dataframe(pd.DataFrame(analogs)[["date","ticker","title","type","outcome_return_7d","outcome_return_30d","similarity"]])

# aggregate analog outcome expectations (if present)
if analogs:
    vals7 = [a.get("outcome_return_7d", np.nan) for a in analogs if not pd.isna(a.get("outcome_return_7d"))]
    vals30 = [a.get("outcome_return_30d", np.nan) for a in analogs if not pd.isna(a.get("outcome_return_30d"))]
    analog_adj_7 = float(np.nanmean(vals7)) if vals7 else 0.0
    analog_adj_30 = float(np.nanmean(vals30)) if vals30 else 0.0
else:
    analog_adj_7 = analog_adj_30 = 0.0

st.write(f"Analogs adjustment: 7-day mean return = {analog_adj_7:.3f}, 30-day mean = {analog_adj_30:.3f}")

# ---------------------------
# --- Risk-adjusted weighting (market regime)
# ---------------------------
@st.cache_data(ttl=60*5)
def get_sector_market_signals():
    # SPY and XBI as proxies
    spy = yf.Ticker("SPY").history(period="30d")["Close"].pct_change().dropna()
    xbi = yf.Ticker("XBI").history(period="30d")["Close"].pct_change().dropna()
    # compute last-week momentum & volatility
    spy_mom = spy.tail(7).mean()
    xbi_mom = xbi.tail(7).mean()
    # risk_on if both positive
    risk_on = (spy_mom > 0) and (xbi_mom > 0)
    return {"spy_mom": float(spy_mom), "xbi_mom": float(xbi_mom), "risk_on": risk_on}

market_signals = get_sector_market_signals()
st.write("Market regime:", market_signals)

# risk weight adjusts how much sentiment/fundamentals/analogs influence forecast
if market_signals["risk_on"]:
    risk_weight = 1.0
else:
    risk_weight = 0.7

# ---------------------------
# --- Modeling: prepare features for time-series / transformer
# ---------------------------
# Build short returns series dataset for supervised learning
close = hist["Close"].fillna(method="ffill")
returns = close.pct_change().dropna()
window = 10  # lookback window for sequence models

X_seq = []
y_next = []
for i in range(window, len(returns)-1):
    X_seq.append(returns.iloc[i-window:i].values)
    y_next.append(returns.iloc[i+1])
X_seq = np.array(X_seq)  # shape (n_samples, window)
y_next = np.array(y_next)

# If not enough data, fallback to small training set
if len(X_seq) < 20:
    st.warning("Not enough historical return data for transformer training; model will use tree fallback.")
    use_transformer_forecast = False

# ---------------------------
# --- Simple Transformer (proof-of-concept) model (tiny)
# ---------------------------
class TinyTransformer(nn.Module):
    def __init__(self, seq_len, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * seq_len, 1)
    def forward(self, x):
        # x: (batch, seq_len)
        b, s = x.shape
        x = x.unsqueeze(-1)  # (b, s, 1)
        x = self.input_proj(x)  # (b, s, d)
        x = x.permute(1,0,2)  # seq first for transformer: (s,b,d)
        x = self.encoder(x)   # (s,b,d)
        x = x.permute(1,0,2).contiguous().view(b, -1)  # (b, s*d)
        out = self.head(x).squeeze(-1)
        return out

# Train small models if requested
model_forecast = None
model_tree = None
scaler = StandardScaler()
if run_btn:
    with st.spinner("Training time-series models (transformer or tree)..."):
        try:
            # prepare training tensors
            X_train = X_seq
            y_train = y_next
            # scale features for tree
            X_flat = X_train.reshape(X_train.shape[0], -1)
            X_flat_s = scaler.fit_transform(X_flat)
            # tree model (fast)
            model_tree = xgb.XGBRegressor(n_estimators=200, max_depth=3, verbosity=0)
            model_tree.fit(X_flat_s, y_train)
            st.success("Trained XGBoost baseline model.")
            if use_transformer_forecast:
                device = DEVICE
                tiny = TinyTransformer(seq_len=window, d_model=32, nhead=4, num_layers=2).to(device)
                opt = torch.optim.Adam(tiny.parameters(), lr=1e-3)
                loss_fn = nn.MSELoss()
                # tiny training loop (short)
                X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
                y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
                for epoch in range(30):
                    tiny.train()
                    opt.zero_grad()
                    pred = tiny(X_t)
                    loss = loss_fn(pred, y_t)
                    loss.backward()
                    opt.step()
                model_forecast = tiny
                st.success("Trained Tiny Transformer model (proof-of-concept).")
        except Exception as e:
            st.error(f"Model training failed: {e}")
            use_transformer_forecast = False

# ---------------------------
# --- Monte Carlo simulation (price paths), with multiple adjustments
# ---------------------------
def monte_carlo_paths(start_price, days, sims, daily_vol, base_drift, analog_adj_7, fund_score, composite_sentiment, risk_weight):
    """
    Produces price paths (intraday steps) and returns:
      - paths: (sims, steps+1)
      - final_closes: last close prices for each sim
      - intraday_max: max across intraday steps (per sim)
    The base_drift is daily; we optionally adjust it by sentiment/analogs/fundamentals.
    """
    steps_per_day = 13
    total_steps = days * steps_per_day
    dt = 1.0 / steps_per_day
    # Adjust drift components:
    # baseline drift = base_drift
    # add composite sentiment scaled by risk_weight and fundamentals and analog effect
    # clamp extremes
    adj = base_drift + (0.5 * composite_sentiment + 0.3 * fund_score + 0.7 * analog_adj_7) * risk_weight
    adj = float(np.clip(adj, -0.05, 0.05))
    # per-step mu and sigma
    mu_step = (adj - 0.5 * (daily_vol**2)) / steps_per_day
    sigma_step = daily_vol / np.sqrt(steps_per_day)
    # generate log returns
    rng = np.random.default_rng(42)
    # use Student-t scaled to sigma for fat tails
    df_t = 4
    # draw t variates and scale them to have unit std then multiply by sigma_step
    t_raw = rng.standard_t(df_t, size=(sims, total_steps))
    t_std = np.sqrt(df_t / (df_t - 2))
    increments = (t_raw / t_std) * sigma_step + mu_step
    # optional small microstructure noise
    micro = rng.normal(0, 0.002, size=(sims, total_steps))
    increments += micro
    # build paths multiplicatively
    log_prices = np.cumsum(increments, axis=1)
    prices = start_price * np.exp(log_prices)
    prices = np.concatenate([np.full((sims,1), start_price), prices], axis=1)
    # compute daily closes & intraday maxima
    intraday_max = []
    intraday_close = []
    for d in range(days):
        sidx = d*steps_per_day + 1
        eidx = (d+1)*steps_per_day + 1
        chunk = prices[:, sidx:eidx]
        intraday_max.append(chunk.max(axis=1))
        intraday_close.append(chunk[:, -1])
    intraday_max = np.vstack(intraday_max).T
    intraday_close = np.vstack(intraday_close).T
    return prices, intraday_max, intraday_close

# Run simulation when user clicks Run
if run_btn:
    with st.spinner("Running Monte Carlo with combined signals..."):
        base_drift = 0.005  # baseline daily drift (neutral small positive)
        paths, intraday_max_arr, intraday_close_arr = monte_carlo_paths(
            start_price=start_price,
            days=forecast_days,
            sims=int(simulations),
            daily_vol=float(volatility_adj),
            base_drift=base_drift,
            analog_adj_7=analog_adj_7,
            fund_score=fund_score,
            composite_sentiment=composite_sentiment,
            risk_weight=risk_weight
        )
        # aggregate horizon metrics (max across days)
        max_over_horizon = intraday_max_arr.max(axis=1)
        final_close = intraday_close_arr[:, -1]
        st.success("Simulation complete.")

        # compute target touch probabilities for grid 200-230 step 5
        targets = list(range(200, 235, 5))
        touch_probs = {t: np.mean(max_over_horizon >= t) for t in targets}
        close_probs = {t: np.mean(final_close >= t) for t in targets}
        df_targets = pd.DataFrame({
            "target": targets,
            "P_touch_intraday": [f"{touch_probs[t]*100:.1f}%" for t in targets],
            "P_close_last_day": [f"{close_probs[t]*100:.1f}%" for t in targets]
        })
        st.markdown("### Target probabilities (intraday touch over horizon vs close on last day)")
        st.dataframe(df_targets)

        # compute combined confidence forecast (mean, 10-90% bands)
        mean_path = np.mean(paths, axis=0)
        p10 = np.percentile(paths, 10, axis=0)
        p90 = np.percentile(paths, 90, axis=0)

        # also compute an "analog-informed expected path" by shifting mean path by analog_adj_7
        analog_shift = 1.0 + analog_adj_7
        analog_mean_path = mean_path * analog_shift

        # Fundamental fair value line (crudely): start_price * (1 + fund_score)
        fv_line = np.linspace(start_price, start_price*(1+fund_score), len(mean_path))

        # Plot combined chart
        fig, ax = plt.subplots(figsize=(10,5))
        steps = np.arange(len(mean_path))
        ax.plot(mean_path, label="MonteCarlo Mean", color="blue")
        ax.fill_between(steps, p10, p90, color="blue", alpha=0.12, label="10–90% MC band")
        ax.plot(analog_mean_path, label="Analog-adjusted mean", color="purple", linestyle="--")
        ax.plot(fv_line, label="Fundamental fair-value (heuristic)", color="darkgreen", linestyle=":")
        ax.set_title(f"Combined Confidence Forecast — {ticker} (start ${start_price:.2f})")
        ax.set_xlabel("Intraday step")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Show expected portfolio for user-provided shares/cost (simple)
        shares = st.number_input("Shares you hold (for expected proceeds calc)", value=41)
        total_cost = st.number_input("Total cost / investment ($)", value=5683.0)
        expected_final = np.mean(final_close) * shares
        median_final = np.median(final_close) * shares
        p5_val, p95_val = np.percentile(final_close * shares, [5,95])
        st.metric("Expected portfolio value (mean)", f"${expected_final:,.0f}")
        st.metric("Median portfolio value", f"${median_final:,.0f}")
        st.write(f"5th–95th percentile range: ${p5_val:,.0f} – ${p95_val:,.0f}")
        st.write(f"Expected profit/loss (mean): ${expected_final - total_cost:,.0f}")

        # Provide downloadable CSV of paths (small sample to avoid huge file)
        if st.button("Download sample of simulated final prices (CSV)"):
            sample_df = pd.DataFrame({
                "final_close": final_close,
                "max_over_horizon": max_over_horizon
            })
            st.download_button("Download CSV", sample_df.to_csv(index=False), file_name=f"{ticker}_sim_sample.csv")

# End of app

    key = "your_actual_api_key"
    ```
    Then access it with `st.secrets["newsapi"]["key"]` instead of hardcoding.
    """)
