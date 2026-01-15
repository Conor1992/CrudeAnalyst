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
st.caption("Price, technicals, spreads, curves, correlations, volatility, and simulations using Yahoo Finance only.")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Global Controls")

benchmark_map = {
    "WTI (CL=F)": "CL=F",
    "Brent (BZ=F)": "BZ=F"
}
benchmark_label = st.sidebar.selectbox("Primary benchmark", list(benchmark_map.keys()))
benchmark = benchmark_map[benchmark_label]

start_date = st.sidebar.date_input("Start date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Feature Toggles")
show_spreads = st.sidebar.checkbox("Show Spread Analysis", True)
show_curve = st.sidebar.checkbox("Show Futures Curve Analysis", True)
show_corr_vol = st.sidebar.checkbox("Show Correlation & Volatility", True)
show_forecasting = st.sidebar.checkbox("Show Forecasts & Simulations", True)

# ---------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    if ticker is None:
        return pd.DataFrame()
    t = yf.Ticker(ticker)
    return t.history(start=start, end=end, interval="1d")

try:
    data = get_history(benchmark, start_date, end_date)
    wti_data = get_history("CL=F", start_date, end_date)
    brent_data = get_history("BZ=F", start_date, end_date)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if data is None or data.empty:
    st.error("No data returned for the selected date range.")
    st.stop()

returns = data["Close"].pct_change().dropna()

# ---------------------------------------------------------
# SECTION 1: PRICE & VOLUME
# ---------------------------------------------------------
st.header("Section 1 – Price & Volume")

col_p1, col_p2 = st.columns([2, 1])

with col_p1:
    fig_price = px.line(
        data,
        x=data.index,
        y="Close",
        title=f"{benchmark_label} close price",
        labels={"Close": "Price (USD)", "index": "Date"}
    )
    st.plotly_chart(fig_price, use_container_width=True)

with col_p2:
    if "Volume" in data.columns and not data["Volume"].isna().all():
        fig_vol = px.bar(
            data,
            x=data.index,
            y="Volume",
            title=f"{benchmark_label} volume",
            labels={"Volume": "Volume", "index": "Date"}
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("No volume data available for this benchmark.")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Latest close", f"${data['Close'].iloc[-1]:.2f}")
col_m2.metric("Daily change", f"{data['Close'].iloc[-1] - data['Close'].iloc[-2]:.2f}")
pct_period = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
col_m3.metric("Return over period", f"{pct_period:.2f}%")

# ---------------------------------------------------------
# SECTION 2: TECHNICAL INDICATORS
# ---------------------------------------------------------
st.header("Section 2 – Technical Indicators")

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
# SECTION 3: SPREAD ANALYSIS
# ---------------------------------------------------------
st.header("Section 3 – Spread Analysis")

if show_spreads:
    st.subheader("Brent–WTI Spread")

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

        st.metric("Latest Brent–WTI spread", f"{spread_df['Brent_WTI_Spread'].iloc[-1]:.2f} USD/bbl")
    else:
        st.info("Insufficient data to compute Brent–WTI spread.")

    st.subheader("Crack Spreads (RBOB & Heating Oil)")

    rb = get_history("RB=F", start_date, end_date)   # RBOB gasoline
    ho = get_history("HO=F", start_date, end_date)   # Heating oil

    if not rb.empty and not ho.empty and not wti_data.empty:
        crack_df = pd.DataFrame({
            "WTI": wti_data["Close"],
            "RBOB": rb["Close"],
            "HO": ho["Close"]
        }).dropna()

        crack_df["321_crack"] = ((2 * crack_df["RBOB"] + crack_df["HO"]) / 3) - crack_df["WTI"]
        crack_df["532_crack"] = ((3 * crack_df["RBOB"] + 2 * crack_df["HO"]) / 5) - crack_df["WTI"]
        crack_df["gasoline_crack"] = crack_df["RBOB"] - crack_df["WTI"]
        crack_df["diesel_crack"] = crack_df["HO"] - crack_df["WTI"]

        fig_crack = px.line(
            crack_df,
            x=crack_df.index,
            y=["321_crack", "532_crack", "gasoline_crack", "diesel_crack"],
            title="Crack spreads (USD/bbl)",
            labels={"value": "Spread (USD/bbl)", "index": "Date", "variable": "Crack"}
        )
        st.plotly_chart(fig_crack, use_container_width=True)

        st.metric("Latest 3-2-1 crack", f"{crack_df['321_crack'].iloc[-1]:.2f} USD/bbl")
    else:
        st.info("Insufficient data to compute crack spreads.")

# ---------------------------------------------------------
# SECTION 4: FUTURES CURVE ANALYSIS
# ---------------------------------------------------------
st.header("Section 4 – Futures Curve Analysis")

if show_curve:
    st.subheader("Futures Curve (Multi-Contract + Analytics)")

    curve_underlying = st.selectbox("Curve underlying", ["WTI", "Brent"])

    if curve_underlying == "WTI":
        futures_contracts = {
            "CL1": "CL=F",
            "CL2": "CLM25.NYM",
            "CL3": "CLZ25.NYM",
            "CL4": "CLM26.NYM",
            "CL5": "CLZ26.NYM"
        }
    else:
        futures_contracts = {
            "BZ1": "BZ=F",
            "BZ2": "BZM25.NYM",
            "BZ3": "BZZ25.NYM",
            "BZ4": "BZM26.NYM",
            "BZ5": "BZZ26.NYM"
        }

    curve_prices = {}
    for label, ticker in futures_contracts.items():
        df_curve = get_history(ticker, date.today() - timedelta(days=10), date.today())
        if df_curve is not None and not df_curve.empty:
            curve_prices[label] = df_curve["Close"].iloc[-1]

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

        curve_df["Carry_%"] = curve_df["Price"].pct_change() * 100
        st.write("Adjacent contract carry (roll yield, %):")
        st.dataframe(curve_df)

        front = curve_df["Price"].iloc[0]
        back = curve_df["Price"].iloc[-1]
        shape = "Contango" if back > front else "Backwardation"
        st.metric("Curve shape", shape, f"{(back/front - 1)*100:.2f}% front→back")

        if len(curve_df) >= 2:
            cal_spread = curve_df["Price"].iloc[0] - curve_df["Price"].iloc[1]
            st.metric("Front–second calendar spread", f"{cal_spread:.2f} USD/bbl")

        if len(curve_df) >= 5:
            bf = (
                curve_df["Price"].iloc[0]
                - 2 * curve_df["Price"].iloc[2]
                + curve_df["Price"].iloc[4]
            )
            st.metric("Simple butterfly (1–3–5)", f"{bf:.2f} USD/bbl")
    else:
        st.info("No futures curve data available with current tickers.")

# ---------------------------------------------------------
# SECTION 5: CORRELATION & VOLATILITY ANALYSIS
# ---------------------------------------------------------
st.header("Section 5 – Correlation & Volatility Analysis")

if show_corr_vol:
    st.subheader("Correlation Matrix (Daily Returns)")

    corr_assets = {
        "WTI": "CL=F",
        "Brent": "BZ=F",
        "USD": "DX-Y.NYB",
        "S&P 500": "^GSPC",
        "Gold": "GC=F",
        "NatGas": "NG=F"
    }

    corr_data = {}
    for name, ticker in corr_assets.items():
        df = get_history(ticker, start_date, end_date)
        if not df.empty:
            corr_data[name] = df["Close"]

    corr_df = pd.DataFrame(corr_data).pct_change().dropna()

    if not corr_df.empty:
        corr_matrix = corr_df.corr()
        st.dataframe(corr_matrix)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough overlapping data to compute correlations.")

    st.subheader("Volatility & Regime Analysis")

    vol_20 = data["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    vol_60 = data["Close"].pct_change().rolling(60).std() * np.sqrt(252)

    vol_df = pd.DataFrame({"Vol_20d": vol_20, "Vol_60d": vol_60}).dropna()
    fig_vol = px.line(
        vol_df,
        x=vol_df.index,
        y=["Vol_20d", "Vol_60d"],
        title="Realized volatility (annualized)",
        labels={"value": "Volatility", "index": "Date", "variable": "Horizon"}
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    ret_20 = data["Close"].pct_change().rolling(20).mean()
    regime_df = pd.DataFrame({"ret_20": ret_20, "vol_20": vol_20}).dropna()

    if not regime_df.empty:
        vol_median = regime_df["vol_20"].median()

        def classify_regime(row):
            if row["ret_20"] > 0 and row["vol_20"] < vol_median:
                return "Trending up"
            elif row["ret_20"] < 0 and row["vol_20"] < vol_median:
                return "Trending down"
            else:
                return "Mean-reverting / choppy"

        regime_df["Regime"] = regime_df.apply(classify_regime, axis=1)
        st.bar_chart(regime_df["Regime"].value_counts())
        st.metric("Current volatility regime", regime_df["Regime"].iloc[-1])

# ---------------------------------------------------------
# SECTION 6: FORECASTS & SIMULATIONS
# ---------------------------------------------------------
st.header("Section 6 – Forecasts & Simulations")

if show_forecasting:
    if returns.empty:
        st.info("Not enough return data for forecasting.")
    else:
        horizon_days = st.slider("Forecast horizon (days)", 5, 60, 30)

        last_date = data.index[-1]
        last_price = data["Close"].iloc[-1]

        # Exponential smoothing
        alpha = st.slider("Exponential smoothing alpha", 0.01, 0.5, 0.2)
        level = data["Close"].iloc[0]
        for p in data["Close"]:
            level = alpha * p + (1 - alpha) * level
        exp_forecast = [level] * horizon_days

        # AR(1) on returns
        window_ar = 60
        if len(returns) > window_ar + 1:
            r_window = returns.tail(window_ar)
            r_lag = r_window.shift(1).dropna()
            r_curr = r_window.iloc[1:]
            phi = (r_lag * r_curr).sum() / (r_lag ** 2).sum()
            last_r = returns.iloc[-1]
            ar_prices = []
            p = last_price
            for i in range(horizon_days):
                p = p * (1 + phi * last_r)
                ar_prices.append(p)
        else:
            ar_prices = [np.nan] * horizon_days

        # Monte Carlo
        mc_paths = st.slider("Monte Carlo paths", 50, 500, 200)
        mu = returns.mean()
        sigma = returns.std()

        mc_matrix = np.zeros((horizon_days, mc_paths))
        for j in range(mc_paths):
            prices = [last_price]
            for i in range(1, horizon_days):
                shock = np.random.normal(mu, sigma)
                prices.append(prices[-1] * (1 + shock))
            mc_matrix[:, j] = prices

        mc_df = pd.DataFrame(mc_matrix)
        mc_median = mc_df.median(axis=1)
        mc_p10 = mc_df.quantile(0.1, axis=1)
        mc_p90 = mc_df.quantile(0.9, axis=1)

        forecast_index = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
        forecast_df = pd.DataFrame({
            "ExpSmooth": exp_forecast,
            "AR1": ar_prices,
            "MC_median": mc_median.values,
            "MC_p10": mc_p10.values,
            "MC_p90": mc_p90.values
        }, index=forecast_index)

        hist_tail = data["Close"].tail(60)
        combined = pd.concat(
            [hist_tail.rename("Historical"), forecast_df],
            axis=0
        )

        fig_forecast = px.line(
            combined,
            x=combined.index,
            y=["Historical", "ExpSmooth", "AR1", "MC_median", "MC_p10", "MC_p90"],
            title=f"{benchmark_label} – forecasts & Monte Carlo bands",
            labels={"value": "Price (USD)", "index": "Date", "variable": "Series"}
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.caption("ExpSmooth = exponential smoothing level; AR1 = rolling AR(1) on returns; MC bands = 10th/90th percentiles of simulated paths.")
