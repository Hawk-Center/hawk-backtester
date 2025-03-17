use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use time::Date;

mod backtester;
mod input_handler;
pub mod metrics;

use backtester::Backtester;
use input_handler::{parse_price_df, parse_weights_df};
use metrics::BacktestMetrics;

/// Python wrapper for the Rust Backtester
#[pyclass]
struct HawkBacktester {
    initial_value: Option<f64>,
    start_date: Option<Date>,
}

#[pymethods]
impl HawkBacktester {
    /// Create a new backtester with a specified initial portfolio value
    #[new]
    fn new(initial_value: Option<f64>) -> Self {
        HawkBacktester {
            initial_value,
            start_date: None,
        }
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
    ///     - Dictionary with performance metrics and simulation statistics
    #[pyo3(text_signature = "(self, prices_df, weights_df)")]
    fn run<'py>(
        &self,
        py: Python<'py>,
        prices_df: PyDataFrame,
        weights_df: PyDataFrame,
    ) -> PyResult<Py<PyAny>> {
        let start_time = Instant::now();

        // Extract the underlying Polars DataFrame references
        let prices_df: DataFrame = prices_df.as_ref().clone();
        let weights_df: DataFrame = weights_df.as_ref().clone();

        let parsing_start = Instant::now();
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
        let parsing_time = parsing_start.elapsed();

        // Use default values if not provided
        let initial_value = self.initial_value.unwrap_or(1_000_000.0);
        let start_date = self.start_date.unwrap_or(weight_events[0].timestamp);

        let simulation_start = Instant::now();
        // Create and run the backtester
        let backtester = Backtester {
            prices: &prices,
            weight_events: &weight_events,
            initial_value,
            start_date,
        };

        // Run the backtest and convert the result back to a Polars DataFrame
        let (results_df, metrics) = backtester.run().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error running backtest: {}",
                e
            ))
        })?;
        let simulation_time = simulation_start.elapsed();
        let total_time = start_time.elapsed();

        // Create Python dictionary for metrics and statistics
        let metrics_dict = PyDict::new(py);
        // Performance metrics
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

        // Simulation statistics
        metrics_dict.set_item("num_price_points", prices.len())?;
        metrics_dict.set_item("num_weight_events", weight_events.len())?;
        metrics_dict.set_item("parsing_time_ms", parsing_time.as_millis() as f64)?;
        metrics_dict.set_item("simulation_time_ms", simulation_time.as_millis() as f64)?;
        metrics_dict.set_item("total_time_ms", total_time.as_millis() as f64)?;
        metrics_dict.set_item(
            "simulation_speed_dates_per_sec",
            (prices.len() as f64) / (simulation_time.as_secs_f64()),
        )?;

        // Convert metrics dictionary to DataFrame
        let metrics_df = DataFrame::new(vec![
            Series::new(
                "metric".into(),
                metrics_dict
                    .keys()
                    .iter()
                    .map(|k| k.extract::<String>().unwrap())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "value".into(),
                metrics_dict
                    .values()
                    .iter()
                    .map(|v| v.extract::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            )
            .into(),
        ])
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create metrics DataFrame: {}",
                e
            ))
        })?;

        // print the metrics_df to check the output
        //  println!("Metrics DataFrame: {}", metrics_df);

        // Return the results DataFrame and metrics DataFrame together in a dictionary
        let return_dict = PyDict::new(py);
        return_dict.set_item("backtest_results", PyDataFrame(results_df))?;
        return_dict.set_item("backtest_metrics", PyDataFrame(metrics_df))?;

        // print the return_dict to check the output
        // println!("Return Dictionary: {}", return_dict);

        // Return the results dictionary
        Ok(return_dict.into())
    }
}

/// A Python module implemented in Rust using PyO3.
#[pymodule]
fn hawk_backtester(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HawkBacktester>()?;
    Ok(())
}

#[cfg(test)]
mod tests;
