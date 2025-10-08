### --- SECTION 3: Monte Carlo Forecast, Probabilities & Best Limit Helper ---

st.header("Section 3 — Monte Carlo Forecast & Target Probabilities")

# Pull shared inputs prepared in Section 2
ticker = st.session_state.get("ticker", "SPRB")
drift_sentiment = float(st.session_state.get("drift_sentiment", 0.005))
vol_est = float(st.session_state.get("vol_est", 0.02))
composite_sentiment = float(st.session_state.get("composite_sentiment", 0.0))

# UI controls for this section
colA, colB, colC = st.columns(3)
with colA:
    start_price = st.number_input(
        "Start price (use live/PM if you want)",
        min_value=0.01,
        value=float(hist["Close"].iloc[-1]),
        step=0.01
    )
with colB:
    forecast_days = st.slider("Forecast horizon (trading days)", 1, 10, 2)
with colC:
    simulations = st.slider("Simulations", 1000, 50000, 10000, step=1000)

intraday_steps_per_day = 13  # ~30-min bars across 6.5h
daily_drift = drift_sentiment  # from Section 2 sentiment

st.write(
    f"Using daily drift = **{daily_drift:.3%}**, "
    f"daily vol = **{vol_est:.2%}**, "
    f"steps/day = **{intraday_steps_per_day}**"
)

# Monte Carlo engine (fat-tailed increments via Student-t scaled to sigma)
def simulate_paths(start_price, days, sims, daily_vol, daily_mu, steps_per_day=13, seed=42):
    rng = np.random.default_rng(seed)
    total_steps = days * steps_per_day
    # per-step params for geometric process
    mu_step = (daily_mu - 0.5 * daily_vol**2) / steps_per_day
    sigma_step = daily_vol / np.sqrt(steps_per_day)

    # Student-t increments (df=3) scaled to sigma_step
    df = 3
    t_raw = rng.standard_t(df, size=(sims, total_steps))
    t_std = np.sqrt(df / (df - 2))
    increments = (t_raw / t_std) * sigma_step + mu_step

    # small microstructure noise
    increments += rng.normal(0, 0.002, size=(sims, total_steps))

    # build price paths
    log_prices = np.cumsum(increments, axis=1)
    prices = start_price * np.exp(log_prices)
    prices = np.concatenate([np.full((sims, 1), start_price), prices], axis=1)

    # intraday stats
    intraday_max = []
    intraday_close = []
    for d in range(days):
        sidx = d * steps_per_day + 1
        eidx = (d + 1) * steps_per_day + 1
        chunk = prices[:, sidx:eidx]
        intraday_max.append(chunk.max(axis=1))
        intraday_close.append(chunk[:, -1])
    intraday_max = np.vstack(intraday_max).T   # sims x days
    intraday_close = np.vstack(intraday_close).T
    return prices, intraday_max, intraday_close

# Run sim
if st.button("Run 2-Day Forecast Now"):
    with st.spinner("Simulating paths..."):
        paths, intraday_max, intraday_close = simulate_paths(
            start_price, forecast_days, simulations, vol_est, daily_drift, intraday_steps_per_day
        )

    # Compute overall stats
    final_close = intraday_close[:, -1]
    mean_price = float(np.mean(final_close))
    median_price = float(np.median(final_close))
    p5, p95 = [float(x) for x in np.percentile(final_close, [5, 95])]

    st.markdown("### Distribution (last day close)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"${mean_price:,.2f}")
    col2.metric("Median", f"${median_price:,.2f}")
    col3.metric("5th %ile", f"${p5:,.2f}")
    col4.metric("95th %ile", f"${p95:,.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(final_close, bins=120, density=True)
    ax.axvline(mean_price, color="blue", linestyle="--", label=f"Mean ${mean_price:,.2f}")
    ax.axvline(median_price, color="green", linestyle="--", label=f"Median ${median_price:,.2f}")
    ax.set_title("Simulated Final-Day Closing Price Distribution")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # Intraday-touch probabilities for 200–230 in $5 steps
    st.markdown("### Intraday-touch probabilities (over entire horizon)")
    max_over_horizon = intraday_max.max(axis=1)
    targets = list(range(200, 235, 5))
    probs = []
    for tval in targets:
        ptouch = float(np.mean(max_over_horizon >= tval))
        pclose = float(np.mean(final_close >= tval))
        probs.append({"Target ($)": tval, "P(Touch Intraday)": ptouch, "P(Close ≥ Target)": pclose})
    df_probs = pd.DataFrame(probs)
    df_probs_fmt = df_probs.copy()
    df_probs_fmt["P(Touch Intraday)"] = df_probs_fmt["P(Touch Intraday)"].map(lambda x: f"{x*100:.1f}%")
    df_probs_fmt["P(Close ≥ Target)"] = df_probs_fmt["P(Close ≥ Target)"].map(lambda x: f"{x*100:.1f}%")
    st.dataframe(df_probs_fmt, use_container_width=True)

    # “Best limit” helper between $200 and $230: maximize probability × profit
    st.markdown("### Best Limit between $200–$230 (probability × profit heuristic)")
    shares = st.number_input("Your shares", value=41, min_value=1)
    total_cost = st.number_input("Your total cost ($)", value=5683.0, min_value=0.0, step=1.0)
    breakeven = total_cost / shares if shares > 0 else 0.0

    best_row = None
    best_score = -1e9
    rows = []
    for _, row in df_probs.iterrows():
        price = float(row["Target ($)"])
        pfill = float(row["P(Touch Intraday)"])
        profit_if_hit = (price - breakeven) * shares
        score = pfill * profit_if_hit
        rows.append({
            "Limit ($)": price,
            "P(Touch Intraday)": pfill,
            "Profit if filled ($)": profit_if_hit,
            "Score = P×Profit": score
        })
        if score > best_score:
            best_score = score
            best_row = (price, pfill, profit_if_hit, score)

    df_best = pd.DataFrame(rows)
    df_best_fmt = df_best.copy()
    df_best_fmt["P(Touch Intraday)"] = df_best_fmt["P(Touch Intraday)"].map(lambda x: f"{x*100:.1f}%")
    df_best_fmt["Profit if filled ($)"] = df_best_fmt["Profit if filled ($)"].map(lambda x: f"{x:,.0f}")
    df_best_fmt["Score = P×Profit"] = df_best_fmt["Score = P×Profit"].map(lambda x: f"{x:,.0f}")
    st.dataframe(df_best_fmt, use_container_width=True)

    if best_row:
        limit_price, pfill, prof, score = best_row
        st.success(
            f"**Suggested limit**: ${limit_price:.0f}  |  "
            f"P(touch) ≈ {pfill*100:.1f}%  |  "
            f"Profit if filled ≈ ${prof:,.0f}"
        )

    # Optional: Download a small CSV sample (final_close + max_over_horizon)
    st.markdown("#### Download sample of simulated results")
    if st.button("Prepare CSV sample"):
        sample_df = pd.DataFrame({
            "final_close": final_close,
            "max_over_horizon": max_over_horizon
        })
        st.download_button(
            "Download CSV",
            data=sample_df.to_csv(index=False),
            file_name=f"{ticker}_sim_sample.csv",
            mime="text/csv"
        )

# End of app
