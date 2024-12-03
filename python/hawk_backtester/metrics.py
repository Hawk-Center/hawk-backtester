import polars as pl
import numpy as np
from typing import Optional


def calculate_sharpe_ratio(
    returns_df: pl.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    returns_col: str = "returns",
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.

    Args:
        returns_df: DataFrame containing returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)
        returns_col: Name of the column containing returns

    Returns:
        float: Annualized Sharpe ratio
    """
    returns = returns_df[returns_col].to_numpy()

    # Convert annual risk-free rate to per-period rate
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_daily

    # Annualize mean and standard deviation
    mean_annual = np.mean(excess_returns) * periods_per_year
    std_annual = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)

    return mean_annual / std_annual if std_annual > 0 else 0.0


def calculate_metrics(
    returns_df: pl.DataFrame,
    risk_free_rate: float = 0.0,
    returns_col: str = "returns",
    cumulative_returns_col: str = "cumulative_returns",
) -> dict:
    """
    Calculate various performance metrics for a portfolio.

    Args:
        returns_df: DataFrame containing returns
        risk_free_rate: Annual risk-free rate
        returns_col: Name of the column containing returns
        cumulative_returns_col: Name of the column containing cumulative returns

    Returns:
        dict: Dictionary containing calculated metrics
    """
    returns = returns_df[returns_col].to_numpy()
    cum_returns = returns_df[cumulative_returns_col].to_numpy()

    total_return = np.exp(cum_returns[-1]) - 1

    # Calculate drawdown series
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns - running_max
    max_drawdown = np.min(np.exp(drawdown) - 1)

    metrics = {
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (252 / len(returns)) - 1,
        "annualized_volatility": np.std(returns, ddof=1) * np.sqrt(252),
        "sharpe_ratio": calculate_sharpe_ratio(returns_df, risk_free_rate),
        "max_drawdown": max_drawdown,
        "calmar_ratio": (
            ((1 + total_return) ** (252 / len(returns)) - 1) / abs(max_drawdown)
            if max_drawdown != 0
            else np.inf
        ),
    }

    return metrics
