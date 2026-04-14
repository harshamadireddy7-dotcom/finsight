import pandas as pd
from sklearn.ensemble import IsolationForest
from data_fetcher import fetch_stock_data, add_technical_indicators

def detect_anomalies(df, contamination=0.05):
    df = df.copy()
    df.dropna(subset=["Daily_Return", "Volume_Ratio", "Price_Range_Pct", "MA20_Deviation"], inplace=True)
    features = df[["Daily_Return", "Volume_Ratio", "Price_Range_Pct", "MA20_Deviation"]].values
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(features)
    df["anomaly_score"] = model.decision_function(features)
    df["anomaly_type"] = "Normal"
    mask = df["anomaly"] == -1
    df.loc[mask & (df["Daily_Return"] < -0.03), "anomaly_type"] = "Price Spike (Drop)"
    df.loc[mask & (df["Daily_Return"] > 0.03), "anomaly_type"] = "Price Surge"
    df.loc[mask & (df["Volume_Ratio"] > 2.0), "anomaly_type"] = "Volume Surge"
    df.loc[mask & (df["anomaly_type"] == "Normal"), "anomaly_type"] = "MA Divergence"
    return df

def get_anomaly_summary(df):
    anomalies = df[df["anomaly"] == -1].copy()
    anomalies = anomalies[["Close", "Daily_Return", "Volume_Ratio", "anomaly_type", "anomaly_score"]].copy()
    anomalies["Daily_Return"] = (anomalies["Daily_Return"] * 100).round(2)
    anomalies["Volume_Ratio"] = anomalies["Volume_Ratio"].round(2)
    anomalies["anomaly_score"] = anomalies["anomaly_score"].round(4)
    anomalies.columns = ["Close Price", "Daily Return (%)", "Volume vs Avg", "Anomaly Type", "Score"]
    return anomalies.sort_index(ascending=False)

if __name__ == "__main__":
    df = fetch_stock_data("BMW.DE", period="1y")
    df = add_technical_indicators(df)
    df = detect_anomalies(df)
    anomalies = df[df["anomaly"] == -1]
    print(f"Anomalies detected: {len(anomalies)} out of {len(df)} days")
    print(anomalies["anomaly_type"].value_counts())