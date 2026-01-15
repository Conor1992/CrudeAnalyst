import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta

# Optional PDF generation (pip install fpdf)
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crude Oil Analytics Terminal",
    layout="wide"
)

st.title("🛢️ Crude Oil Analytics Terminal")

# ---------------------------------------------------------
# SIDEBAR – GLOBAL CONTROLS & PAGE NAV
# ---------------------------------------------------------
st.sidebar.header("Global Controls")

benchmark_map = {
    "WTI (CL=F)": "CL=F",
    "Brent (BZ=F)": "BZ=F"
}
benchmark_label = st.sidebar.selectbox("Primary benchmark", list(benchmark_map.keys()))
benchmark = benchmark_map[benchmark_label]

start_date = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select page",
    [
        "1. Price & Technicals",
        "2. Spreads & Curves",
        "3. Correlation, Vol & GARCH",
        "4. Cointegration & Spread Trading",
        "5. Factor Risk & Advanced Forecasts",
        "6. PDF Report"
    ]
)

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
# HELPER FUNCTIONS
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


def garch_11(returns: pd.Series, omega=0.000001, alpha=0.05, beta=0.9):
    """Simple GARCH(1,1) volatility (daily, then annualized)."""
    r = returns.dropna().values
    n = len(r)
    if n == 0:
        return pd.Series(dtype=float)
    var = np.zeros(n)
    var[0] = np.var(r)
    for t in range(1, n):
        var[t] = omega + alpha * (r[t-1] ** 2) + beta * var[t-1]
    vol = np.sqrt(var) * np.sqrt(252)
    return pd.Series(vol, index=returns.dropna().index)


def simple_arima_111(series: pd.Series, horizon: int = 30):
    """Very simple ARIMA(1,1,1)-like: difference, AR(1) on diff, MA(1) via residual mean."""
    s = series.dropna()
    if len(s) < 50:
        return pd.Series([np.nan] * horizon,
                         index=[s.index[-1] + timedelta(days=i) for i in range(1, horizon+1)])
    diff = s.diff().dropna()
    # AR(1) on diff
    x = diff.shift(1).dropna()
    y = diff.iloc[1:]
    phi = (x * y).sum() / (x ** 2).sum()
    # MA(1) approx: mean residual
    resid = y - phi * x
    theta = resid.mean()
    last = s.iloc[-1]
    forecasts = []
    for i in range(horizon):
        d = phi * diff.iloc[-1] + theta
        last = last + d
        forecasts.append(last)
    idx = [s.index[-1] + timedelta(days=i) for i in range(1, horizon+1)]
    return pd.Series(forecasts, index=idx)


def kalman_trend(series: pd.Series, q=0.0001, r=0.001):
    """Simple 1D Kalman filter for trend estimation."""
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    x_est = s.iloc[0]
    p_est = 1.0
    trend = []
    for z in s:
        # Predict
        x_pred = x_est
        p_pred = p_est + q
        # Update
        k = p_pred / (p_pred + r)
        x_est = x_pred + k * (z - x_pred)
        p_est = (1 - k) * p_pred
        trend.append(x_est)
    return pd.Series(trend, index=s.index)


def engle_granger_cointegration(x: pd.Series, y: pd.Series):
    """Simple Engle-Granger: regress y on x, return residuals & beta."""
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 50:
        return None, None
    X = np.column_stack([df.iloc[:, 0], np.ones(len(df))])
    Y = df.iloc[:, 1].values
    beta, alpha = np.linalg.lstsq(X, Y, rcond=None)[0]
    fitted = beta * df.iloc[:, 0] + alpha
    resid = df.iloc[:, 1] - fitted
    return resid, beta


def factor_betas(crude: pd.Series, factors: pd.DataFrame):
    df = pd.concat([crude, factors], axis=1).dropna()
    ret = df.pct_change().dropna()
    y = ret.iloc[:, 0].values
    X = np.column_stack([ret.iloc[:, 1:].values, np.ones(len(ret))])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    betas = coeffs[:-1]
    alpha = coeffs[-1]
    return betas, alpha, ret


# ---------------------------------------------------------
# PAGE 1 – PRICE & TECHNICALS
# ---------------------------------------------------------
if page == "1. Price & Technicals":
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

    st.header("Section 2 – Technical Indicators")

    data_ta = add_indicators(data)

    tab_price, tab_rsi, tab_macd, tab_bb, tab_vol = st.tabs(
        ["Price", "RSI", "MACD", "Bollinger Bands", "Volatility"]
    )

    with tab_price:
        st.line_chart(data_ta["Close"])

    with tab_rsi:
        st.line_chart(data_ta["RSI_14"])

    with tab_macd:
        st.line_chart(data_ta[["MACD", "MACD_signal"]])

    with tab_bb:
        bb_df = data_ta[["Close", "BB_upper", "BB_mid", "BB_lower"]].dropna()
        fig_bb = px.line(
            bb_df,
            x=bb_df.index,
            y=["Close", "BB_upper", "BB_mid", "BB_lower"],
            title="Bollinger Bands"
        )
        st.plotly_chart(fig_bb, use_container_width=True)

    with tab_vol:
        st.line_chart(data_ta["Vol_20d_annualized"])

# ---------------------------------------------------------
# PAGE 2 – SPREADS & CURVES
# ---------------------------------------------------------
elif page == "2. Spreads & Curves":
    st.header("Section 3 – Spread Analysis")

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
            title="Brent–WTI spread (USD/bbl)"
        )
        st.plotly_chart(fig_spread, use_container_width=True)
        st.metric("Latest Brent–WTI spread", f"{spread_df['Brent_WTI_Spread'].iloc[-1]:.2f} USD/bbl")
    else:
        st.info("Insufficient data to compute Brent–WTI spread.")

    st.subheader("Crack Spreads (RBOB & Heating Oil)")
    rb = get_history("RB=F", start_date, end_date)
    ho = get_history("HO=F", start_date, end_date)

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
            title="Crack spreads (USD/bbl)"
        )
        st.plotly_chart(fig_crack, use_container_width=True)
    else:
        st.info("Insufficient data to compute crack spreads.")

    st.header("Section 4 – Futures Curve Analysis")

    st.subheader("Futures Curve (Multi-Contract + Dynamics)")
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
            title=f"{curve_underlying} futures curve"
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
# PAGE 3 – CORRELATION, VOL & GARCH
# ---------------------------------------------------------
elif page == "3. Correlation, Vol & GARCH":
    st.header("Section 5 – Correlation & Volatility Analysis")

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

    st.subheader("Realized Volatility & GARCH(1,1)")

    vol_20 = data["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    vol_60 = data["Close"].pct_change().rolling(60).std() * np.sqrt(252)
    garch_vol = garch_11(returns)

    vol_df = pd.DataFrame({
        "Vol_20d": vol_20,
        "Vol_60d": vol_60,
        "GARCH_11": garch_vol
    }).dropna()

    fig_vol = px.line(
        vol_df,
        x=vol_df.index,
        y=["Vol_20d", "Vol_60d", "GARCH_11"],
        title="Realized & GARCH(1,1) volatility (annualized)"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Regime Switching (Return/Vol Regimes)")

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
        st.metric("Current regime", regime_df["Regime"].iloc[-1])
    else:
        st.info("Not enough data to classify regimes.")

# ---------------------------------------------------------
# PAGE 4 – COINTEGRATION & SPREAD TRADING
# ---------------------------------------------------------
elif page == "4. Cointegration & Spread Trading":
    st.header("Section – Cointegration & Spread Trading")

    st.subheader("WTI–Brent Cointegration & Z-Score")

    if not wti_data.empty and not brent_data.empty:
        resid, beta = engle_granger_cointegration(wti_data["Close"], brent_data["Close"])
        if resid is not None:
            z = (resid - resid.mean()) / resid.std()
            spread_sig = pd.DataFrame({"Residual": resid, "Z": z})

            fig_resid = px.line(
                spread_sig,
                x=spread_sig.index,
                y="Residual",
                title=f"Residual (Brent - {beta:.2f} * WTI)"
            )
            st.plotly_chart(fig_resid, use_container_width=True)

            fig_z = px.line(
                spread_sig,
                x=spread_sig.index,
                y="Z",
                title="Z-score of residual (spread signal)"
            )
            st.plotly_chart(fig_z, use_container_width=True)

            st.caption("Simple Engle–Granger: regress Brent on WTI, use residual as spread; Z-score for trading bands.")
        else:
            st.info("Not enough data for cointegration test.")
    else:
        st.info("Need both WTI and Brent data for cointegration.")

    st.subheader("RBOB–WTI Cointegration (Gasoline Crack Style)")

    rb = get_history("RB=F", start_date, end_date)
    if not rb.empty and not wti_data.empty:
        resid_rb, beta_rb = engle_granger_cointegration(wti_data["Close"], rb["Close"])
        if resid_rb is not None:
            z_rb = (resid_rb - resid_rb.mean()) / resid_rb.std()
            spread_rb = pd.DataFrame({"Residual": resid_rb, "Z": z_rb})

            fig_rb = px.line(
                spread_rb,
                x=spread_rb.index,
                y="Z",
                title=f"RBOB–WTI cointegration Z-score (beta={beta_rb:.2f})"
            )
            st.plotly_chart(fig_rb, use_container_width=True)
        else:
            st.info("Not enough data for RBOB–WTI cointegration.")
    else:
        st.info("Need WTI and RBOB data for this analysis.")

# ---------------------------------------------------------
# PAGE 5 – FACTOR RISK & ADVANCED FORECASTS
# ---------------------------------------------------------
elif page == "5. Factor Risk & Advanced Forecasts":
    st.header("Section – Factor Risk Model")

    usd_df = get_history("DX-Y.NYB", start_date, end_date)
    spx_df = get_history("^GSPC", start_date, end_date)
    gold_df = get_history("GC=F", start_date, end_date)

    if not usd_df.empty and not spx_df.empty and not gold_df.empty:
        factors = pd.DataFrame({
            "USD": usd_df["Close"],
            "SPX": spx_df["Close"],
            "Gold": gold_df["Close"]
        })
        betas, alpha, ret = factor_betas(data["Close"], factors)

        beta_usd, beta_spx, beta_gold = betas
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        col_f1.metric("Beta to USD", f"{beta_usd:.3f}")
        col_f2.metric("Beta to S&P 500", f"{beta_spx:.3f}")
        col_f3.metric("Beta to Gold", f"{beta_gold:.3f}")
        col_f4.metric("Alpha (daily)", f"{alpha:.5f}")

        # Factor contribution to variance (very rough)
        cov = ret.cov()
        factor_var = cov.iloc[1:, 1:].values
        w = betas.reshape(-1, 1)
        total_var = float(w.T @ factor_var @ w)
        st.write("Approximate factor variance contribution:", total_var)
    else:
        st.info("Not enough data for full factor model (USD/SPX/Gold).")

    st.header("Section – Advanced Forecasts")

    if returns.empty:
        st.info("Not enough return data for forecasting.")
    else:
        horizon_days = st.slider("Forecast horizon (days)", 5, 60, 30)
        last_date = data.index[-1]
        last_price = data["Close"].iloc[-1]

        # ARIMA-lite
        arima_forecast = simple_arima_111(data["Close"], horizon_days)

        # Kalman trend
        trend_series = kalman_trend(data["Close"])

        # Regime-dependent forecast: if trending up → drift up, if down → drift down, else flat
        vol_20 = data["Close"].pct_change().rolling(20).std() * np.sqrt(252)
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
            current_regime = regime_df["Regime"].iloc[-1]
        else:
            current_regime = "Unknown"

        st.metric("Current regime (for regime-dependent forecast)", current_regime)

        # Simple regime-dependent drift
        if current_regime == "Trending up":
            drift = returns.mean() * 1.5
        elif current_regime == "Trending down":
            drift = returns.mean() * -1.0
        else:
            drift = 0.0

        reg_dep_prices = []
        p = last_price
        for i in range(horizon_days):
            p = p * (1 + drift)
            reg_dep_prices.append(p)

        forecast_index = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
        adv_forecast_df = pd.DataFrame({
            "ARIMA_111": arima_forecast.values,
            "Regime_dep": reg_dep_prices
        }, index=forecast_index)

        hist_tail = data["Close"].tail(60)
        combined = pd.concat(
            [hist_tail.rename("Historical"), adv_forecast_df],
            axis=0
        )

        fig_adv = px.line(
            combined,
            x=combined.index,
            y=["Historical", "ARIMA_111", "Regime_dep"],
            title="Advanced forecasts (ARIMA-lite & regime-dependent)"
        )
        st.plotly_chart(fig_adv, use_container_width=True)

        st.subheader("Kalman Filter Trend")
        if not trend_series.empty:
            trend_df = pd.DataFrame({"Price": data["Close"], "Trend": trend_series}).dropna()
            fig_trend = px.line(
                trend_df,
                x=trend_df.index,
                y=["Price", "Trend"],
                title="Kalman filter trend estimate"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data for Kalman trend.")

# ---------------------------------------------------------
# PAGE 6 – PDF REPORT
# ---------------------------------------------------------
elif page == "6. PDF Report":
    st.header("Section – PDF Report Generator")

    st.write(
        "This will generate a summary report of the analytics in this app. "
        "For real PDF output, the `fpdf` package is used if installed."
    )

    report_text = f"""
Crude Oil Analytics Terminal – Summary Report

Benchmark: {benchmark_label} ({benchmark})
Date range: {start_date} to {end_date}

Sections implemented:
1. Price & Technicals
   - Price, volume, RSI, MACD, Bollinger Bands, realized volatility.

2. Spreads & Curves
   - Brent–WTI spread, crack spreads (3-2-1, 5-3-2, gasoline, diesel).
   - Multi-contract futures curve, carry, curve shape, calendar spreads, butterfly.

3. Correlation, Vol & GARCH
   - Cross-asset correlation matrix and heatmap.
   - Realized volatility (20d, 60d) and GARCH(1,1) volatility.
   - Regime switching based on returns and volatility.

4. Cointegration & Spread Trading
   - Engle–Granger cointegration for WTI–Brent and WTI–RBOB.
   - Residual and Z-score based spread trading signals.

5. Factor Risk & Advanced Forecasts
   - Factor risk model with USD, S&P 500, and Gold.
   - ARIMA(1,1,1)-style forecast, Kalman filter trend model.
   - Regime-dependent forecasting based on current volatility regime.

This report is a high-level summary; detailed charts and metrics are available in the app.
"""

    if HAS_FPDF:
        if st.button("Generate PDF report"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in report_text.split("\n"):
                pdf.multi_cell(0, 8, line)
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                label="Download PDF report",
                data=pdf_bytes,
                file_name="crude_analytics_report.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("fpdf is not installed. Falling back to text download.")
        st.download_button(
            label="Download report as text",
            data=report_text.encode("utf-8"),
            file_name="crude_analytics_report.txt",
            mime="text/plain"
        )
