use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::sync::Arc;
use time::Date;

pub mod backtester;
pub mod input_handler;
use backtester::{Backtester, PriceData, WeightEvent};
use input_handler::{parse_price_df, parse_weights_df};

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
    ///     prices_df: Polars DataFrame with timestamp column (m/d/y format) and price columns
    ///     weights_df: Polars DataFrame with timestamp column (m/d/y format) and weight columns
    ///
    /// Returns:
    ///     Polars DataFrame with backtest results
    fn run(&self, prices_df: PyDataFrame, weights_df: PyDataFrame) -> PyResult<PyDataFrame> {
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
        let results_df = backtester.run().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error running backtest: {}",
                e
            ))
        })?;

        // Return the result as a PyDataFrame
        Ok(PyDataFrame(results_df))
    }
}

/// A Python module implemented in Rust using PyO3.
#[pymodule]
fn hawk_backtester(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBacktester>()?;
    Ok(())
}
