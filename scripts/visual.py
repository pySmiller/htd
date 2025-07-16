import streamlit as st
import pandas as pd

def simulate(df, model_col, bm_col, actual_col, odds, stake, n_bets):
    df = df.copy()
    df["edge"] = df[model_col] - df[bm_col]
    df["result"] = df[actual_col] - df[bm_col]

    def outcome(r):
        if r.edge > 0 and r.result > 0:   return "win"
        if r.edge < 0 and r.result < 0:   return "win"
        if r.edge == 0 or r.result == 0:  return "push"
        return "loss"

    df["outcome"] = df.apply(outcome, axis=1)
    df = df[df.outcome != "push"]
    df = df.reindex(df.edge.abs().sort_values(ascending=False).index)
    bets = df.iloc[:n_bets]

    wins = (bets.outcome == "win").sum()
    losses = (bets.outcome == "loss").sum()
    profit = wins * (stake * (odds - 1)) - losses * stake
    roi = profit / (n_bets * stake) * 100

    return wins, losses, profit, roi

st.title("🏷️ Profitability Calculator")

uploaded = st.file_uploader("Upload bookmaker_comparison.csv", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    odds = st.number_input("Decimal odds per win", value=1.83, step=0.01)
    stake = st.number_input("Stake per bet (USD)", value=5.0, step=0.5)
    n_bets = st.number_input("Number of bets", value=100, step=1)
    market = st.selectbox("Market to simulate", ["spread", "total", "both"])

    if st.button("Run Simulation"):
        runs = []
        if market in ("spread", "both"):
            runs.append(("Spread", simulate(
                df,
                model_col="model_pred_spread",
                bm_col="bm_pred_spread",
                actual_col="actual_spread",
                odds=odds,
                stake=stake,
                n_bets=n_bets
            )))
        if market in ("total", "both"):
            runs.append(("Total", simulate(
                df,
                model_col="model_pred_total",
                bm_col="bm_pred_total",
                actual_col="actual_total",
                odds=odds,
                stake=stake,
                n_bets=n_bets
            )))

        for name, (wins, losses, profit, roi) in runs:
            st.subheader(f"{name} Market Results")
            st.write(f"- Bets placed: **{n_bets}**")
            st.write(f"- Wins: **{wins}**")
            st.write(f"- Losses: **{losses}**")
            st.write(f"- Profit: **${profit:.2f}**")
            st.write(f"- ROI: **{roi:.2f}%**")
