from typing import Optional, Union
import polars as pl
import numpy as np
from hawk_backtester import run_backtest


class PortfolioBacktester:
    """A wrapper class for the Rust-based portfolio backtester.

    This class provides an easy-to-use interface for running portfolio backtests
    and calculating various performance metrics.

    :param prices: DataFrame containing asset prices with a 'date' column and asset prices in subsequent columns
    :type prices: pl.DataFrame
    :param weights: DataFrame containing portfolio weights with an 'insight_date' column and asset weights in subsequent columns
    :type weights: pl.DataFrame
    :param risk_free_rate: Annual risk-free rate as a decimal (e.g., 0.05 for 5%), defaults to 0.0
    :type risk_free_rate: float, optional

    :ivar returns: DataFrame containing daily and cumulative portfolio returns after running backtest
    :type returns: pl.DataFrame
    """

    def __init__(
        self, prices: pl.DataFrame, weights: pl.DataFrame, risk_free_rate: float = 0.0
    ):
        self._prices = prices
        self._weights = weights
        self._risk_free_rate = risk_free_rate
        self._returns = None

    def run(self) -> pl.DataFrame:
        """Run the portfolio backtest.

        :return: DataFrame containing daily and cumulative portfolio returns
        :rtype: pl.DataFrame
        """
        self._returns = run_backtest(self._prices, self._weights, self._risk_free_rate)
        return self._returns

    @property
    def returns(self) -> Optional[pl.DataFrame]:
        """Get the backtest returns.

        :return: DataFrame containing daily and cumulative portfolio returns if backtest has been run
        :rtype: Optional[pl.DataFrame]
        """
        return self._returns

    def calculate_metrics(self) -> dict:
        """Calculate common portfolio performance metrics.

        :return: Dictionary containing various performance metrics:
                - annual_return: Annualized portfolio return
                - annual_volatility: Annualized portfolio volatility
                - sharpe_ratio: Annualized Sharpe ratio
                - max_drawdown: Maximum portfolio drawdown
                - total_return: Total portfolio return over the period
        :rtype: dict
        :raises ValueError: If backtest hasn't been run yet
        """
        if self._returns is None:
            raise ValueError("Must run backtest before calculating metrics")

        # Convert to numpy for calculations
        returns = self._returns.select("returns").to_numpy().flatten()
        cum_returns = self._returns.select("cumulative_returns").to_numpy().flatten()

        # Calculate metrics
        total_return = np.exp(cum_returns[-1]) - 1
        annual_return = np.exp(252 * np.mean(returns)) - 1
        annual_vol = np.sqrt(252) * np.std(returns)
        sharpe_ratio = (
            (annual_return - self._risk_free_rate) / annual_vol
            if annual_vol != 0
            else 0
        )

        # Calculate maximum drawdown
        cum_returns_exp = np.exp(cum_returns)
        rolling_max = np.maximum.accumulate(cum_returns_exp)
        drawdowns = cum_returns_exp / rolling_max - 1
        max_drawdown = np.min(drawdowns)

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
        }
