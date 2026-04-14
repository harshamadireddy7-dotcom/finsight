import pandas as pd
import numpy as np

from data_fetcher import fetch_stock_data


def fetch_multi_stock(tickers: list, period: str = "1y") -> pd.DataFrame:
    """
    Fetch closing prices for multiple tickers and align them in one DataFrame.

    Args:
        tickers: List of ticker symbols e.g. ['BMW.DE', 'ALV.DE']
        period: Time period e.g. '1y'

    Returns:
        DataFrame where each column is a ticker's closing price, aligned by date.
    """
    frames = {}
    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, period=period)
            frames[ticker] = df["Close"]
        except Exception as e:
            print(f"Could not fetch {ticker}: {e}")

    combined = pd.DataFrame(frames)
    combined.dropna(inplace=True)
    return combined


def compute_correlation(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix from closing prices.
    """
    returns = prices_df.pct_change().dropna()
    return returns.corr()


def compute_performance_stats(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key performance metrics for each stock.

    Returns:
        DataFrame with columns: Total Return (%), Volatility (Ann.), Sharpe Ratio, Max Drawdown (%)
    """
    returns = prices_df.pct_change().dropna()
    stats = {}

    for ticker in prices_df.columns:
        r = returns[ticker]
        total_return = (prices_df[ticker].iloc[-1] / prices_df[ticker].iloc[0] - 1) * 100
        volatility = r.std() * np.sqrt(252)
        sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0

        rolling_max = prices_df[ticker].cummax()
        drawdown = (prices_df[ticker] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        stats[ticker] = {
            "Total Return (%)": round(total_return, 2),
            "Volatility (Ann.)": round(volatility, 4),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown (%)": round(max_drawdown, 2),
        }

    return pd.DataFrame(stats).T


if __name__ == "__main__":
    tickers = ["BMW.DE", "ALV.DE", "MUV2.DE", "SAP.DE"]
    print(f"Fetching portfolio data for: {tickers}")

    prices = fetch_multi_stock(tickers)
    corr = compute_correlation(prices)
    stats = compute_performance_stats(prices)

    print("\nCorrelation Matrix:")
    print(corr.round(2))

    print("\nPerformance Stats:")
    print(stats)
