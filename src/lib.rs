use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::sync::Arc;
use time::Date;

mod backtester;
mod input_handler;
mod metrics;

use backtester::{Backtester, PriceData, WeightEvent};
use input_handler::{parse_price_df, parse_weights_df};
use metrics::BacktestMetrics;

/// Python wrapper for the Rust Backtester
#[pyclass]
struct PyBacktester {
    initial_value: f64,
}

#[pymethods]
impl PyBacktester {
    /// Create a new backtester with a specified initial portfolio value
    #[new]
    fn new(initial_value: f64) -> Self {
        PyBacktester { initial_value }
    }

    /// Run the backtest using price and weight data from Python
    ///
    /// Args:
    ///     prices_df: Polars DataFrame with date column (m/d/y format) and price columns
    ///     weights_df: Polars DataFrame with date column (m/d/y format) and weight columns
    ///
    /// Returns:
    ///     Tuple containing:
    ///     - Polars DataFrame with backtest results
    ///     - Dictionary with performance metrics
    #[pyo3(text_signature = "(self, prices_df, weights_df)")]
    fn run<'py>(
        &self,
        py: Python<'py>,
        prices_df: PyDataFrame,
        weights_df: PyDataFrame,
    ) -> PyResult<(PyDataFrame, Py<PyDict>)> {
        // Extract the underlying Polars DataFrame references
        let prices_df: DataFrame = prices_df.as_ref().clone();
        let weights_df: DataFrame = weights_df.as_ref().clone();

        // Parse the DataFrames into our internal types
        let prices = parse_price_df(&prices_df).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Error parsing price data: {}",
                e
            ))
        })?;

        let weight_events = parse_weights_df(&weights_df).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Error parsing weight data: {}",
                e
            ))
        })?;

        // Create and run the backtester
        let backtester = Backtester {
            prices,
            weight_events,
            initial_value: self.initial_value,
        };

        // Run the backtest and convert the result back to a Polars DataFrame
        let (results_df, metrics) = backtester.run().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error running backtest: {}",
                e
            ))
        })?;

        // Create Python dictionary for metrics
        let metrics_dict = PyDict::new(py);
        metrics_dict.set_item("total_return", metrics.total_return)?;
        metrics_dict.set_item("annualized_return", metrics.annualized_return)?;
        metrics_dict.set_item("annualized_volatility", metrics.annualized_volatility)?;
        metrics_dict.set_item("sharpe_ratio", metrics.sharpe_ratio)?;
        metrics_dict.set_item("sortino_ratio", metrics.sortino_ratio)?;
        metrics_dict.set_item("max_drawdown", metrics.max_drawdown)?;
        metrics_dict.set_item("avg_drawdown", metrics.avg_drawdown)?;
        metrics_dict.set_item("avg_daily_return", metrics.avg_daily_return)?;
        metrics_dict.set_item("win_rate", metrics.win_rate)?;
        metrics_dict.set_item("num_trades", metrics.num_trades)?;

        // Return tuple of DataFrame and metrics dictionary
        Ok((PyDataFrame(results_df), metrics_dict.into()))
    }
}

/// A Python module implemented in Rust using PyO3.
#[pymodule]
fn hawk_backtester(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBacktester>()?;
    Ok(())
}
