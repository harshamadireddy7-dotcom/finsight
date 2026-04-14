import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from data_fetcher import fetch_stock_data, add_technical_indicators, TICKERS
from anomaly_detector import detect_anomalies, get_anomaly_summary
from forecaster import forecast_stock, get_forecast_summary
from portfolio import fetch_multi_stock, compute_correlation, compute_performance_stats

st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide")
st.title("📈 FinSight — Live Market Intelligence Platform")
st.caption("Real-time anomaly detection & forecasting | BMW · Allianz · Munich Re · SAP")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="BMW.DE")
    period = st.selectbox("Time Period", ["6mo", "1y", "2y"], index=1)
    page = st.radio("Navigation", ["📊 Live Market Dashboard", "🔴 Anomaly Radar", "🔮 Price Forecast", "🗂️ Portfolio Analyzer"])
    st.markdown("---")
    for t, name in TICKERS.items():
        st.caption(f"`{t}` — {name}")

@st.cache_data(ttl=300)
def load_data(ticker, period):
    df = fetch_stock_data(ticker, period=period)
    return add_technical_indicators(df)

try:
    df = load_data(ticker, period)
except Exception as e:
    st.error(f"Could not load data for {ticker}: {e}")
    st.stop()

if page == "📊 Live Market Dashboard":
    st.subheader(f"📊 {ticker} — Live Market Dashboard")
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"€{latest['Close']:.2f}", f"{change_pct:+.2f}%")
    c2.metric("High", f"€{latest['High']:.2f}")
    c3.metric("Low", f"€{latest['Low']:.2f}")
    c4.metric("Volume", f"{int(latest['Volume']):,}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color="blue", width=1.5)))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, xaxis_title="Date", yaxis_title="Price (€)")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="steelblue", opacity=0.7))
    fig2.add_trace(go.Scatter(x=df.index, y=df["Volume_MA20"], name="20-Day Avg", line=dict(color="red", width=2)))
    fig2.update_layout(height=250, xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🔴 Anomaly Radar":
    st.subheader(f"🔴 {ticker} — Anomaly Radar")
    with st.spinner("Running IsolationForest..."):
        df_anom = detect_anomalies(df)
    normal = df_anom[df_anom["anomaly"] == 1]
    anomalies = df_anom[df_anom["anomaly"] == -1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Days", len(df_anom))
    c2.metric("Anomalies", len(anomalies))
    c3.metric("Rate", f"{len(anomalies)/len(df_anom)*100:.1f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal.index, y=normal["Close"], mode="lines", name="Normal", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["Close"], mode="markers", name="Anomaly", marker=dict(color="red", size=10)))
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (€)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Anomaly Breakdown")
    bd = anomalies["anomaly_type"].value_counts().reset_index()
    bd.columns = ["Type", "Count"]
    st.plotly_chart(px.bar(bd, x="Type", y="Count", color="Type"), use_container_width=True)
    st.dataframe(get_anomaly_summary(df_anom), use_container_width=True)

elif page == "🔮 Price Forecast":
    st.subheader(f"🔮 {ticker} — 30-Day Forecast (Prophet)")
    with st.spinner("Training model..."):
        model, forecast = forecast_stock(df, days=30)
    summary = get_forecast_summary(forecast)
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast End", summary["forecast_end"])
    c2.metric("Predicted Price", f"€{summary['predicted_price']}")
    c3.metric("95% Band", f"€{summary['lower_bound']} – €{summary['upper_bound']}")
    today = pd.Timestamp.today().normalize().tz_localize(None)
    future_fc = forecast[forecast["ds"] > today]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Actual", line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=future_fc["ds"], y=future_fc["yhat"], mode="lines", name="Forecast", line=dict(color="green", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
        y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(0,200,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="95% Band"
    ))
    fig.add_shape(
        type="line",
        x0=str(today.date()), x1=str(today.date()),
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=2, dash="dot")
    )
    fig.add_annotation(
        x=str(today.date()), y=1,
        xref="x", yref="paper",
        text="Today", showarrow=False,
        font=dict(color="red")
    )
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (€)")
    st.plotly_chart(fig, use_container_width=True)
    ft = future_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy().round(2)
    ft.columns = ["Date", "Forecast (€)", "Lower (€)", "Upper (€)"]
    st.dataframe(ft, use_container_width=True)
elif page == "🗂️ Portfolio Analyzer":
    st.subheader("🗂️ Portfolio Analyzer")
    selected = st.multiselect("Select Tickers", list(TICKERS.keys()), default=["BMW.DE", "ALV.DE", "MUV2.DE", "SAP.DE"])
    if len(selected) < 2:
        st.warning("Select at least 2 tickers.")
        st.stop()
    with st.spinner("Fetching portfolio..."):
        prices = fetch_multi_stock(selected, period=period)
    corr = compute_correlation(prices)
    stats = compute_performance_stats(prices)
    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)
    norm = prices / prices.iloc[0] * 100
    fig_p = go.Figure()
    for col in norm.columns:
        fig_p.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
    fig_p.update_layout(title="Indexed Performance (Base=100)", height=400)
    st.plotly_chart(fig_p, use_container_width=True)
    st.dataframe(stats, use_container_width=True)