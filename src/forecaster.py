import pandas as pd
from prophet import Prophet
 
from data_fetcher import fetch_stock_data
 
 
def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert stock DataFrame to Prophet's required format.
    Prophet requires columns named 'ds' (date) and 'y' (value).
    """
    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    return prophet_df
 
 
def forecast_stock(df: pd.DataFrame, days: int = 30) -> tuple:
    """
    Forecast future stock prices using Facebook Prophet.
 
    Args:
        df: Raw stock DataFrame (from fetch_stock_data)
        days: Number of future days to forecast
 
    Returns:
        Tuple of (model, forecast_dataframe)
        forecast_dataframe has columns: ds, yhat, yhat_lower, yhat_upper
    """
    prophet_df = prepare_prophet_data(df)
 
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
 
    model.fit(prophet_df)
 
    future = model.make_future_dataframe(periods=days, freq="B")  # "B" = business days only
    forecast = model.predict(future)
 
    return model, forecast
 
 
def get_forecast_summary(forecast: pd.DataFrame, days: int = 30) -> dict:
    """
    Return a quick summary of the forecast for the next N days.
    """
    future_only = forecast.tail(days)
    return {
        "forecast_start": future_only["ds"].iloc[0].strftime("%Y-%m-%d"),
        "forecast_end": future_only["ds"].iloc[-1].strftime("%Y-%m-%d"),
        "predicted_price": round(future_only["yhat"].iloc[-1], 2),
        "lower_bound": round(future_only["yhat_lower"].iloc[-1], 2),
        "upper_bound": round(future_only["yhat_upper"].iloc[-1], 2),
    }
 
 
if __name__ == "__main__":
    print("Forecasting BMW.DE for next 30 days...")
    df = fetch_stock_data("BMW.DE", period="1y")
    model, forecast = forecast_stock(df, days=30)
 
    summary = get_forecast_summary(forecast)
    print(f"\nForecast period: {summary['forecast_start']} → {summary['forecast_end']}")
    print(f"Predicted price (day 30): €{summary['predicted_price']}")
    print(f"95% confidence band: €{summary['lower_bound']} – €{summary['upper_bound']}")
 