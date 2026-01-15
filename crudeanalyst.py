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
st.caption("Curves, spreads, indicators, volatility, correlations, forecasts, and scenarios using Yahoo Finance only.")

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
    if ticker is None:
        return pd.DataFrame()
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

# Daily returns
returns = data["Close"].pct_change().dropna()

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
        corr_returns = corr_df.pct_change().dropna(how="any")
        if corr_returns.empty:
            st.info("No overlapping return data across selected assets.")
        else:
            corr_matrix = corr_returns.corr()
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
# VOLATILITY ANALYTICS
# ---------------------------------------------------------
st.subheader("🌪 Volatility analytics")

# GARCH-style EWMA volatility
def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    var_list = []
    prev_var = returns.var()
    for r in returns:
        v = lam * prev_var + (1 - lam) * (r ** 2)
        var_list.append(v)
        prev_var = v
    ewma = np.sqrt(np.array(var_list)) * np.sqrt(252)
    return pd.Series(ewma, index=returns.index)

ewma_series = ewma_vol(returns)
data["EWMA_vol_annualized"] = ewma_series.reindex(data.index)

# Realized volatility cones
horizons = [10, 20, 60, 120]
cone_df = pd.DataFrame(index=data.index)

for h in horizons:
    cone_df[f"Vol_{h}d"] = data["Close"].pct_change().rolling(h).std() * np.sqrt(252)

fig_cone = px.line(
    cone_df,
    x=cone_df.index,
    y=[c for c in cone_df.columns],
    title="Realized volatility cones (annualized)",
    labels={"value": "Volatility", "index": "Date", "variable": "Horizon"}
)
st.plotly_chart(fig_cone, use_container_width=True)

# Volatility regime detection (based on 20d vol)
vol_series = data["Vol_20d_annualized"].dropna()
if not vol_series.empty:
    low_thr = vol_series.quantile(0.33)
    high_thr = vol_series.quantile(0.66)

    def classify_regime(v):
        if v <= low_thr:
            return "Low"
        elif v >= high_thr:
            return "High"
        else:
            return "Medium"

    regime = vol_series.apply(classify_regime)
    regime_counts = regime.value_counts()

    st.write("Volatility regimes (based on 20d realized vol):")
    st.bar_chart(regime_counts)

    current_vol = vol_series.iloc[-1]
    current_regime = classify_regime(current_vol)
    st.metric("Current volatility regime", current_regime, f"{current_vol:.2%}")
else:
    st.info("Not enough data to classify volatility regimes.")

# ---------------------------------------------------------
# FORECASTING: EXP SMOOTHING, ROLLING AR, MONTE CARLO
# ---------------------------------------------------------
st.subheader("🔮 Forecasting & simulations")

horizon_days = st.slider("Forecast horizon (days)", 5, 60, 30)

last_date = data.index[-1]
last_price = data["Close"][-1]

# Exponential smoothing forecast (on price)
alpha = st.slider("Exponential smoothing alpha", 0.01, 0.5, 0.2)
level = data["Close"].iloc[0]
for p in data["Close"]:
    level = alpha * p + (1 - alpha) * level
exp_forecast_prices = [level for _ in range(horizon_days)]
exp_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
exp_df = pd.DataFrame({"ExpSmooth": exp_forecast_prices}, index=exp_dates)

# Rolling AR(1) forecast on returns
window_ar = 60
if len(returns) > window_ar + 1:
    r_window = returns.tail(window_ar)
    r_lag = r_window.shift(1).dropna()
    r_curr = r_window.iloc[1:]
    phi = (r_lag * r_curr).sum() / (r_lag ** 2).sum()
    last_r = returns.iloc[-1]
    ar_returns = [phi ** i * last_r for i in range(1, horizon_days + 1)]
    ar_prices = []
    p = last_price
    for r in ar_returns:
        p = p * (1 + r)
        ar_prices.append(p)
    ar_df = pd.DataFrame({"AR1": ar_prices}, index=exp_dates)
else:
    ar_df = pd.DataFrame(index=exp_dates)

# Monte Carlo simulation
mc_paths = st.slider("Monte Carlo paths", 50, 500, 200)
mu = returns.mean()
sigma = returns.std()
dt = 1.0

mc_matrix = np.zeros((horizon_days, mc_paths))
for j in range(mc_paths):
    prices = [last_price]
    for i in range(1, horizon_days):
        shock = np.random.normal(mu * dt, sigma * np.sqrt(dt))
        prices.append(prices[-1] * (1 + shock))
    mc_matrix[:, j] = prices

mc_index = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
mc_df = pd.DataFrame(mc_matrix, index=mc_index)
mc_median = mc_df.median(axis=1)
mc_p10 = mc_df.quantile(0.1, axis=1)
mc_p90 = mc_df.quantile(0.9, axis=1)

forecast_df = pd.DataFrame({"Historical": data["Close"].tail(60)})
forecast_df = forecast_df.append(
    pd.DataFrame(
        {
            "ExpSmooth": exp_df["ExpSmooth"],
            "AR1": ar_df["AR1"] if "AR1" in ar_df.columns else np.nan,
            "MC_median": mc_median,
            "MC_p10": mc_p10,
            "MC_p90": mc_p90,
        }
    )
)

fig_forecast = px.line(
    forecast_df,
    x=forecast_df.index,
    y=["Historical", "ExpSmooth", "AR1", "MC_median", "MC_p10", "MC_p90"],
    title=f"{benchmark_label} – forecasts & Monte Carlo bands",
    labels={"value": "Price (USD)", "index": "Date", "variable": "Series"}
)
st.plotly_chart(fig_forecast, use_container_width=True)

st.caption("ExpSmooth = exponential smoothing level; AR1 = rolling AR(1) on returns; MC bands = 10th/90th percentiles of simulated paths.")

# ---------------------------------------------------------
# SCENARIO ANALYSIS
# ---------------------------------------------------------
st.subheader("🧮 Scenario analysis")

st.write("Simple what-if on crude vs USD and OPEC supply (illustrative elasticities).")

usd_change = st.slider("USD index change (%)", -10.0, 10.0, 0.0, step=0.5)
opec_change = st.slider("OPEC supply change (mbpd)", -3.0, 3.0, 0.0, step=0.1)

# Very rough illustrative elasticities (you can tune these):
beta_usd = -0.3   # crude % change per 1% USD move
beta_opec = -0.02 # crude % change per 1 mbpd OPEC change

price_impact_pct = beta_usd * usd_change + beta_opec * opec_change
scenario_price = last_price * (1 + price_impact_pct / 100.0)

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Current price", f"${last_price:.2f}")
col_s2.metric("Scenario price", f"${scenario_price:.2f}")
col_s3.metric("Price impact", f"{price_impact_pct:.2f}%")

st.caption("Elasticities are placeholders; adjust beta_usd and beta_opec in code to match your own views or regressions.")

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
