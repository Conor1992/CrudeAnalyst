import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crude Oil Analyst Dashboard",
    layout="wide"
)

st.title("🛢️ Crude Oil Analyst Dashboard")
st.caption("Market monitoring and analytics using Yahoo Finance data only.")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Controls")

benchmark_map = {
    "WTI (CL=F)": "CL=F",
    "Brent (BZ=F)": "BZ=F"
}

benchmark_label = st.sidebar.selectbox(
    "Select Benchmark",
    list(benchmark_map.keys())
)

benchmark = benchmark_map[benchmark_label]

# Date pickers
start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2023, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

# ---------------------------------------------------------
# MARKET DATA SECTION
# ---------------------------------------------------------
st.subheader("📈 Benchmark Price Overview")

ticker = yf.Ticker(benchmark)

try:
    hist = ticker.history(start=start_date, end=end_date, interval="1d")

    if hist.empty:
        st.error("No data returned for the selected date range.")
    else:
        fig = px.line(
            hist,
            x=hist.index,
            y="Close",
            title=f"{benchmark_label} Price Trend",
            labels={"Close": "Price (USD)", "index": "Date"}
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Price", f"${hist['Close'][-1]:.2f}")
        col2.metric("Daily Change", f"{hist['Close'][-1] - hist['Close'][-2]:.2f}")
        pct = (hist['Close'][-1] / hist['Close'][0] - 1) * 100
        col3.metric("Return Over Period", f"{pct:.2f}%")

except Exception as e:
    st.error(f"Error fetching market data: {e}")

# ---------------------------------------------------------
# SUPPLY & DEMAND INPUTS
# ---------------------------------------------------------
st.subheader("⚖️ Supply & Demand Indicators")

colA, colB = st.columns(2)

with colA:
    opec_output = st.number_input("OPEC Output (mbpd)", value=28.0)
    us_production = st.number_input("US Production (mbpd)", value=13.2)
    inventory_level = st.number_input("OECD Inventories (mb)", value=2700)

with colB:
    refinery_runs = st.number_input("Global Refinery Runs (mbpd)", value=82.0)
    demand_growth = st.number_input("Demand Growth (mbpd)", value=1.5)
    spare_capacity = st.number_input("OPEC Spare Capacity (mbpd)", value=3.5)

st.success("Supply & demand inputs updated.")

# ---------------------------------------------------------
# CRACK SPREAD CALCULATOR
# ---------------------------------------------------------
st.subheader("🧮 3-2-1 Crack Spread Calculator")

colC, colD, colE = st.columns(3)

with colC:
    crude_price = st.number_input("Crude Price (USD/bbl)", value=80.0)
with colD:
    gasoline_price = st.number_input("Gasoline Price (USD/bbl)", value=95.0)
with colE:
    diesel_price = st.number_input("Diesel Price (USD/bbl)", value=100.0)

crack_spread = (0.5 * gasoline_price + 0.5 * diesel_price) - crude_price

st.metric("3-2-1 Crack Spread", f"${crack_spread:.2f}")
