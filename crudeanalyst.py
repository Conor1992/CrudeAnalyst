import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crude Oil Analyst Dashboard",
    layout="wide"
)

st.title("🛢️ Crude Oil Analyst Dashboard")
st.caption("Curves, spreads, indicators, correlations, comparison, and exports using Yahoo Finance only.")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Controls")

benchmark_map = {
    "WTI (CL=F)": "CL=F",
    "Brent (BZ=F)": "BZ=F"
}
benchmark_label = st.sidebar.selectbox("Primary benchmark", list(benchmark_map.keys()))
benchmark = benchmark_map[benchmark_label]

comparison_map = {
    "None": None,
    "Brent (BZ=F)": "BZ=F",
    "WTI (CL=F)": "CL=F",
    "US Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
    "S&P 500 (^GSPC)": "^GSPC",
    "Euro Stoxx 50 (^STOXX50E)": "^STOXX50E",
    "Gold (GC=F)": "GC=F",
    "NatGas (NG=F)": "NG=F"
}
comparison_label = st.sidebar.selectbox("Comparison asset (normalized)", list(comparison_map.keys()))
comparison_ticker = comparison_map[comparison_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Correlation matrix assets")
corr_assets_map = {
    "WTI (CL=F)": "CL=F",
    "Brent (BZ=F)": "BZ=F",
    "US Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
    "S&P 500 (^GSPC)": "^GSPC",
    "Euro Stoxx 50 (^STOXX50E)": "^STOXX50E",
    "Gold (GC=F)": "GC=F",
    "NatGas (NG=F)": "NG=F"
}
corr_default = ["WTI (CL=F)", "Brent (BZ=F)", "US Dollar Index (DX-Y.NYB)", "S&P 500 (^GSPC)"]
corr_selected_labels = st.sidebar.multiselect(
    "Select assets for correlation",
    list(corr_assets_map.keys()),
    default=corr_default
)

start_date = st.sidebar.date_input("Start date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

# ---------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    return t.history(start=start, end=end, interval="1d")

try:
    data = get_history(benchmark, start_date, end_date)
    comp_data = get_history(comparison_ticker, start_date, end_date) if comparison_ticker else None
    brent_data = get_history("BZ=F", start_date, end_date)
    wti_data = get_history("CL=F", start_date, end_date)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if data is None or data.empty:
    st.error("No data returned for the selected date range.")
    st.stop()

# ---------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_mid"] = ma20
    df["BB_upper"] = ma20 + 2 * std20
    df["BB_lower"] = ma20 - 2 * std20

    # Rolling volatility (20-day, annualized)
    df["Vol_20d_annualized"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    return df

data = add_indicators(data)

# ---------------------------------------------------------
# PRICE & VOLUME
# ---------------------------------------------------------
st.subheader("📈 Price & Volume")

fig_price = px.line(
    data,
    x=data.index,
    y="Close",
    title=f"{benchmark_label} close price",
    labels={"Close": "Price (USD)", "index": "Date"}
)
st.plotly_chart(fig_price, use_container_width=True)

fig_vol = px.bar(
    data,
    x=data.index,
    y="Volume",
    title=f"{benchmark_label} volume",
    labels={"Volume": "Volume", "index": "Date"}
)
st.plotly_chart(fig_vol, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Latest close", f"${data['Close'][-1]:.2f}")
col2.metric("Daily change", f"{data['Close'][-1] - data['Close'][-2]:.2f}")
pct = (data['Close'][-1] / data['Close'][0] - 1) * 100
col3.metric("Return over period", f"{pct:.2f}%")

# ---------------------------------------------------------
# TECHNICAL INDICATORS VIEW
# ---------------------------------------------------------
st.subheader("📊 Technical indicators")

tab_price, tab_rsi, tab_macd, tab_bb, tab_vol = st.tabs(
    ["Price", "RSI", "MACD", "Bollinger Bands", "Volatility"]
)

with tab_price:
    st.line_chart(data["Close"])

with tab_rsi:
    st.line_chart(data["RSI_14"])

with tab_macd:
    st.line_chart(data[["MACD", "MACD_signal"]])

with tab_bb:
    bb_df = data[["Close", "BB_upper", "BB_mid", "BB_lower"]].dropna()
    fig_bb = px.line(
        bb_df,
        x=bb_df.index,
        y=["Close", "BB_upper", "BB_mid", "BB_lower"],
        labels={"value": "Price (USD)", "index": "Date", "variable": "Series"},
        title="Bollinger Bands"
    )
    st.plotly_chart(fig_bb, use_container_width=True)

with tab_vol:
    st.line_chart(data["Vol_20d_annualized"])

# ---------------------------------------------------------
# BRENT–WTI SPREAD
# ---------------------------------------------------------
st.subheader("🔀 Brent–WTI spread")

if not brent_data.empty and not wti_data.empty:
    spread_df = pd.DataFrame({
        "Brent": brent_data["Close"],
        "WTI": wti_data["Close"]
    }).dropna()
    spread_df["Brent_WTI_Spread"] = spread_df["Brent"] - spread_df["WTI"]

    fig_spread = px.line(
        spread_df,
        x=spread_df.index,
        y="Brent_WTI_Spread",
        title="Brent–WTI spread (USD/bbl)",
        labels={"Brent_WTI_Spread": "Spread (USD/bbl)", "index": "Date"}
    )
    st.plotly_chart(fig_spread, use_container_width=True)

    st.metric("Latest Brent–WTI spread", f"{spread_df['Brent_WTI_Spread'][-1]:.2f} USD/bbl")
else:
    st.info("Insufficient data to compute Brent–WTI spread for the selected dates.")

# ---------------------------------------------------------
# NORMALIZED PERFORMANCE COMPARISON
# ---------------------------------------------------------
st.subheader("📐 Normalized performance comparison")

if comparison_ticker and comp_data is not None and not comp_data.empty:
    comp_df = pd.DataFrame({
        benchmark_label: data["Close"],
        comparison_label: comp_data["Close"]
    }).dropna()

    norm_df = comp_df / comp_df.iloc[0] * 100

    fig_comp = px.line(
        norm_df,
        x=norm_df.index,
        y=norm_df.columns,
        title="Normalized performance (start = 100)",
        labels={"value": "Index (start=100)", "index": "Date", "variable": "Asset"}
    )
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Select a comparison asset in the sidebar to see normalized performance.")

# ---------------------------------------------------------
# FUTURES CURVE (MULTI-CONTRACT)
# ---------------------------------------------------------
st.subheader("🧵 Futures curve (multi-contract)")

curve_underlying = st.selectbox("Curve underlying", ["WTI", "Brent"])

# NOTE: These tickers are examples; adjust to the exact contracts you care about.
if curve_underlying == "WTI":
    futures_contracts = {
        "Front": "CL=F",          # front month
        "Jun 2025": "CLM25.NYM",
        "Dec 2025": "CLZ25.NYM",
        "Jun 2026": "CLM26.NYM",
        "Dec 2026": "CLZ26.NYM"
    }
else:
    futures_contracts = {
        "Front": "BZ=F",          # front month
        "Jun 2025": "BZM25.NYM",
        "Dec 2025": "BZZ25.NYM",
        "Jun 2026": "BZM26.NYM",
        "Dec 2026": "BZZ26.NYM"
    }

curve_prices = {}
for label, ticker in futures_contracts.items():
    try:
        df_curve = get_history(ticker, date.today() - timedelta(days=10), date.today())
        if df_curve is not None and not df_curve.empty:
            curve_prices[label] = df_curve["Close"][-1]
    except Exception:
        continue

if curve_prices:
    curve_df = pd.DataFrame(
        {"Contract": list(curve_prices.keys()), "Price": list(curve_prices.values())}
    )
    fig_curve = px.line(
        curve_df,
        x="Contract",
        y="Price",
        markers=True,
        title=f"{curve_underlying} futures curve",
        labels={"Price": "Price (USD/bbl)", "Contract": "Contract"}
    )
    st.plotly_chart(fig_curve, use_container_width=True)
else:
    st.info("No futures curve data available with current tickers. Adjust contract tickers in the code if needed.")

# ---------------------------------------------------------
# CORRELATION MATRIX (DAILY RETURNS)
# ---------------------------------------------------------
st.subheader("🔗 Correlation matrix (daily returns)")

corr_tickers = {label: corr_assets_map[label] for label in corr_selected_labels}
corr_closes = {}

for label, ticker in corr_tickers.items():
    try:
        df_corr = get_history(ticker, start_date, end_date)
        if df_corr is None or df_corr.empty:
            st.warning(f"No data for {label} ({ticker}). Skipping.")
            continue
        if "Close" not in df_corr.columns:
            st.warning(f"{label} ({ticker}) has no 'Close' column. Skipping.")
            continue
        corr_closes[label] = df_corr["Close"]
    except Exception as e:
        st.warning(f"Error loading {label} ({ticker}): {e}")

if len(corr_closes) < 2:
    st.info("Not enough valid assets to compute a correlation matrix.")
else:
    corr_df = pd.DataFrame(corr_closes).dropna(how="any")
    if corr_df.shape[0] < 2:
        st.info("No overlapping data across selected assets.")
    else:
        returns = corr_df.pct_change().dropna(how="any")
        if returns.empty:
            st.info("No overlapping return data across selected assets.")
        else:
            corr_matrix = returns.corr()
            st.write("Correlation matrix (daily returns):")
            st.dataframe(corr_matrix)

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation heatmap"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------
# SIMPLE FORWARD PROJECTION
# ---------------------------------------------------------
st.subheader("🔮 Simple forward projection")

horizon_days = st.slider("Forecast horizon (days)", 5, 60, 30)

last_date = data.index[-1]
last_price = data["Close"][-1]
recent_ret = data["Close"].pct_change().dropna().tail(60)
avg_daily_ret = recent_ret.mean()

future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
future_prices = [last_price * ((1 + avg_daily_ret) ** i) for i in range(1, horizon_days + 1)]

future_df = pd.DataFrame({"Close": future_prices}, index=future_dates)
proj_df = pd.concat([data[["Close"]].tail(60), future_df])

fig_forecast = px.line(
    proj_df,
    x=proj_df.index,
    y="Close",
    title=f"{benchmark_label} – historical & simple projection",
    labels={"Close": "Price (USD)", "index": "Date"}
)
st.plotly_chart(fig_forecast, use_container_width=True)
st.caption("Projection uses average daily return over the last 60 trading days (naive, not a full econometric model).")

# ---------------------------------------------------------
# EXPORT DATA
# ---------------------------------------------------------
st.subheader("📤 Export data")

export_df = data.copy()
export_df.index.name = "Date"
csv = export_df.to_csv().encode("utf-8")

st.download_button(
    label="Download benchmark data with indicators (CSV)",
    data=csv,
    file_name=f"{benchmark.replace('=','')}_with_indicators.csv",
    mime="text/csv"
)
