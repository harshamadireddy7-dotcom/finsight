import yfinance as yf
import pandas as pd

TICKERS = {
    "BMW.DE": "BMW Group",
    "ALV.DE": "Allianz SE",
    "MUV2.DE": "Munich Re",
    "SAP.DE": "SAP SE",
    "DBK.DE": "Deutsche Bank",
}

def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]
    df["Price_Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["MA20_Deviation"] = (df["Close"] - df["MA20"]) / df["MA20"]
    return df