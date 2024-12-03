//! Portfolio backtesting library for financial analysis
//!
//! This library provides tools for calculating portfolio performance metrics including:
//! - Log returns calculation from price time series
//! - Portfolio return calculation with support for cash positions
//! - Risk-free rate adjustments
//!
//! The library is designed to work with Polars DataFrames and provides Python bindings
//! through PyO3 for seamless integration with Python workflows.

use polars::prelude::*;
use std::{collections::HashMap, sync::Arc};
// use polars::frame::DataFrame as CoreDataFrame;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

/// Represents price data for multiple assets at a specific point in time
///
/// # Fields
/// * `prices` - HashMap mapping asset identifiers to their prices
/// * `timestamp` - Point in time for these prices
#[derive(Debug)]
struct PriceData {
    prices: HashMap<Arc<str>, f64>,
    timestamp: time::OffsetDateTime,
}

/// Represents logarithmic returns for multiple assets at a specific point in time
#[derive(Debug)]
struct LogReturnData {
    returns: HashMap<Arc<str>, f64>,
    timestamp: time::OffsetDateTime,
}

/// Represents portfolio weights for multiple assets at a specific point in time
///
/// # Fields
/// * `positions` - HashMap mapping asset identifiers to their weights (0.0 to 1.0)
/// * `timestamp` - Point in time for these weights
///
/// The sum of all weights should be <= 1.0, with any remainder representing cash position
#[derive(Debug)]
struct PortfolioWeightData {
    positions: HashMap<Arc<str>, f64>,
    timestamp: time::OffsetDateTime,
}

/// Represents the portfolio's return at a specific point in time
#[derive(Debug)]
struct PortfolioReturnData {
    returns: f64,
    timestamp: time::OffsetDateTime,
}

/// Trait for types that have an associated timestamp
trait Timestamped {
    fn timestamp(&self) -> time::OffsetDateTime;
}

impl Timestamped for PriceData {
    fn timestamp(&self) -> time::OffsetDateTime {
        self.timestamp
    }
}

impl Timestamped for LogReturnData {
    fn timestamp(&self) -> time::OffsetDateTime {
        self.timestamp
    }
}

impl Timestamped for PortfolioWeightData {
    fn timestamp(&self) -> time::OffsetDateTime {
        self.timestamp
    }
}

impl Timestamped for PortfolioReturnData {
    fn timestamp(&self) -> time::OffsetDateTime {
        self.timestamp
    }
}

/// Calculates logarithmic returns for a series of price data points
///
/// # Arguments
/// * `prices` - Vector of PriceData containing asset prices over time
///
/// # Returns
/// Vector of LogReturnData containing log returns for each asset
fn log_returns(prices: Vec<PriceData>) -> Vec<LogReturnData> {
    let mut log_returns = Vec::new();
    for i in 1..prices.len() {
        let prev = &prices[i - 1];
        let curr = &prices[i];
        let mut returns = HashMap::new();
        for (ticker, price) in curr.prices.iter() {
            if let Some(prev_price) = prev.prices.get(ticker) {
                returns.insert(Arc::clone(ticker), (price / prev_price).ln());
            }
        }
        log_returns.push(LogReturnData {
            returns,
            timestamp: curr.timestamp,
        });
    }
    log_returns
}

/// Calculates portfolio returns based on weights and log returns, accounting for cash positions
///
/// # Arguments
/// * `portfolio_weights` - Vector of portfolio weights over time
/// * `log_returns` - Vector of logarithmic returns for each asset
/// * `risk_free_rate` - Annual risk-free rate as a decimal (e.g., 0.05 for 5% APR)
///
/// # Returns
/// Vector of PortfolioReturnData containing portfolio returns over time
///
/// # Example
/// ```
/// let weights = vec![PortfolioWeightData {
///     positions: HashMap::from([
///         (Arc::from("AAPL"), 0.6),
///         (Arc::from("GOOGL"), 0.3),
///     ]),
///     timestamp: /* timestamp */
/// }];
/// // 10% cash position (1.0 - 0.9 invested)
/// ```
fn portfolio_returns(
    portfolio_weights: Vec<PortfolioWeightData>,
    log_returns: Vec<LogReturnData>,
    risk_free_rate: f64,
) -> Vec<PortfolioReturnData> {
    // Convert annual risk-free rate to daily log return
    // Assuming 252 trading days per year
    let daily_rf_log_return = (1.0 + risk_free_rate).powf(1.0 / 252.0).ln();

    let mut portfolio_returns = Vec::new();
    for i in 0..portfolio_weights.len() {
        let weights = &portfolio_weights[i];
        let returns = &log_returns[i];

        // Calculate sum of weights to determine cash position
        let invested_weight: f64 = weights.positions.values().sum();
        let cash_weight = (1.0 - invested_weight).max(0.0); // Ensure non-negative

        // Calculate return from invested positions
        let invested_return: f64 = returns
            .returns
            .iter()
            .map(|(ticker, ret)| weights.positions.get(ticker).unwrap_or(&0.0) * ret)
            .sum();

        // Add return from cash position
        let total_return = invested_return + (cash_weight * daily_rf_log_return);

        portfolio_returns.push(PortfolioReturnData {
            returns: total_return,
            timestamp: returns.timestamp,
        });
    }
    portfolio_returns
}

/// Converts price data from a Polars DataFrame to a vector of PriceData
///
/// # Arguments
/// * `df` - Polars DataFrame containing price data
///
/// # Returns
/// Vector of PriceData
fn df_to_price_data(df: &DataFrame) -> Result<Vec<PriceData>, PolarsError> {
    let date_series = df.column("date")?;
    let price_columns = df
        .get_columns()
        .iter()
        .filter(|col| col.name() != "date" && col.name() != "insight_date")
        .map(|col| col.name())
        .collect::<Vec<_>>();

    let mut price_data = Vec::new();

    for row_idx in 0..df.height() {
        let mut prices = HashMap::new();
        // Convert Unix timestamp to OffsetDateTime
        let unix_ts = (date_series
            .i64()?
            .get(row_idx)
            .ok_or_else(|| PolarsError::ComputeError("Invalid timestamp".into()))?)
            / 1_000_000;
        let timestamp = time::OffsetDateTime::from_unix_timestamp(unix_ts).map_err(|e| {
            PolarsError::ComputeError(format!("Invalid timestamp conversion: {}", e).into())
        })?;

        for ticker in &price_columns {
            if let Some(price) = df.column(ticker)?.f64()?.get(row_idx) {
                // Correct conversion from PlSmallStr to &str
                prices.insert(Arc::from(ticker.as_str()), price);
            }
        }

        price_data.push(PriceData {
            prices,
            timestamp: timestamp.into(),
        });
    }

    Ok(price_data)
}

/// Converts weight data from a Polars DataFrame to a vector of PortfolioWeightData
///
/// # Arguments
/// * `df` - Polars DataFrame containing weight data
///
/// # Returns
/// Vector of PortfolioWeightData
fn df_to_weight_data(df: &DataFrame) -> Result<Vec<PortfolioWeightData>, PolarsError> {
    let date_series = df.column("insight_date")?;
    let weight_columns = df
        .get_columns()
        .iter()
        .filter(|col| col.name() != "date" && col.name() != "insight_date")
        .map(|col| col.name())
        .collect::<Vec<_>>();

    let mut weight_data = Vec::new();

    for row_idx in 0..df.height() {
        let mut positions = HashMap::new();
        // Convert Unix timestamp to OffsetDateTime
        let unix_ts = (date_series
            .i64()?
            .get(row_idx)
            .ok_or_else(|| PolarsError::ComputeError("Invalid timestamp".into()))?)
            / 1_000_000;
        let timestamp = time::OffsetDateTime::from_unix_timestamp(unix_ts).map_err(|e| {
            PolarsError::ComputeError(format!("Invalid timestamp conversion: {}", e).into())
        })?;

        for ticker in &weight_columns {
            if let Some(weight) = df.column(ticker)?.f64()?.get(row_idx) {
                // Correct conversion from PlSmallStr to &str
                positions.insert(Arc::from(ticker.as_str()), weight);
            }
        }

        weight_data.push(PortfolioWeightData {
            positions,
            timestamp: timestamp.into(),
        });
    }

    Ok(weight_data)
}

/// Converts a Python Polars DataFrame to a Rust Polars DataFrame
///
/// # Arguments
/// * `pydf` - A Python Polars DataFrame
///
/// # Returns
/// A Rust Polars DataFrame
fn convert_py_to_rust_df(pydf: PyDataFrame) -> DataFrame {
    // Convert the PyDataFrame to a Rust DataFrame
    let df: DataFrame = pydf.into();

    // Perform any operations on the DataFrame here
    // For example, let's just clone it for demonstration
    let processed_df = df.clone();

    // Wrap the processed DataFrame back into a PyDataFrame for returning to Python
    processed_df
}

/// Converts a Python Polars DataFrame to a Rust Polars DataFrame and returns it
///
/// # Arguments
/// * `pydf` - A Python Polars DataFrame
///
/// # Returns
/// A Rust Polars DataFrame
#[pyfunction]
fn run_backtest(
    prices: PyDataFrame,
    weights: PyDataFrame,
    risk_free_rate: f64,
) -> PyResult<PyDataFrame> {
    let prices_df = convert_py_to_rust_df(prices);
    let weights_df = convert_py_to_rust_df(weights);

    // Explicitly map PolarsError to PyErr
    let price_data = df_to_price_data(&prices_df)
        .map_err(|e| PyException::new_err(format!("Price data error: {}", e)))?;
    let weight_data = df_to_weight_data(&weights_df)
        .map_err(|e| PyException::new_err(format!("Weight data error: {}", e)))?;

    let log_returns = log_returns(price_data);
    let portfolio_returns = portfolio_returns(weight_data, log_returns, risk_free_rate);

    // Calculate cumulative returns
    let mut cumulative_returns = Vec::with_capacity(portfolio_returns.len());
    let mut running_sum = 0.0;
    for r in portfolio_returns.iter() {
        running_sum += r.returns;
        cumulative_returns.push(running_sum);
    }

    // Convert portfolio returns to a Polars DataFrame with both daily and cumulative returns
    let portfolio_returns_df = DataFrame::new(vec![
        Series::new(
            "date".into(),
            portfolio_returns
                .iter()
                .map(|r| r.timestamp.unix_timestamp())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "returns".into(),
            portfolio_returns
                .iter()
                .map(|r| r.returns)
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new("cumulative_returns".into(), cumulative_returns).into(),
    ])
    .map_err(|e| PyException::new_err(format!("DataFrame creation error: {}", e)))?;

    Ok(PyDataFrame(portfolio_returns_df))
}

#[pymodule]
fn hawk_backtester(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_log_return_backtest() {
        let A: Arc<str> = "A".into();
        let B: Arc<str> = "B".into();
        let C: Arc<str> = "C".into();

        let day_one = time::OffsetDateTime::now_utc();
        let day_two = day_one + time::Duration::days(1);

        let prices = vec![
            PriceData {
                prices: HashMap::from([
                    (Arc::clone(&A), 1.0),
                    (Arc::clone(&B), 2.0),
                    (Arc::clone(&C), 2.0),
                ]),
                timestamp: day_one.clone(),
            },
            PriceData {
                prices: HashMap::from([
                    (Arc::clone(&A), 1.0),
                    (Arc::clone(&B), 1.0),
                    (Arc::clone(&C), 4.0),
                ]),
                timestamp: day_two.clone(),
            },
        ];
        let weights = vec![PortfolioWeightData {
            positions: HashMap::from([
                (Arc::clone(&A), 0.5),
                (Arc::clone(&B), 0.5),
                (Arc::clone(&C), 0.0),
            ]),
            timestamp: day_one.clone(),
        }];

        let log_returns = log_returns(prices);
        println!("LOG RETURNS: {:?}", log_returns);
        let portfolio_returns = portfolio_returns(weights, log_returns, 0.05);
        println!("{:?}", portfolio_returns);
        println!(
            "{:?}",
            portfolio_returns
                .iter()
                .map(|r| r.returns.exp())
                .sum::<f64>()
        );
    }

    #[test]
    fn test_portfolio_returns_with_cash() {
        let A: Arc<str> = "A".into();
        let B: Arc<str> = "B".into();

        let day_one = time::OffsetDateTime::now_utc();
        let day_two = day_one + time::Duration::days(1);

        // Create test data with 60% invested (30% each in A and B)
        let weights = vec![PortfolioWeightData {
            positions: HashMap::from([(Arc::clone(&A), 0.3), (Arc::clone(&B), 0.3)]),
            timestamp: day_one,
        }];

        // Test returns: A up 10%, B down 5%
        let log_returns = vec![LogReturnData {
            returns: HashMap::from([
                (Arc::clone(&A), (0.10_f64).ln()),
                (Arc::clone(&B), (0.95_f64).ln()),
            ]),
            timestamp: day_two,
        }];

        // Test with 5% annual risk-free rate
        let risk_free_rate = 0.05;
        let portfolio_returns = portfolio_returns(weights, log_returns, risk_free_rate);

        // Verify results
        assert!(!portfolio_returns.is_empty());
        let return_data = &portfolio_returns[0];

        // Calculate expected return:
        // 30% * ln(1.10) + 30% * ln(0.95) + 40% * daily_rf_rate
        let daily_rf = (1.0 + risk_free_rate).powf(1.0 / 252.0).ln();
        let expected_return = 0.3 * (0.10_f64).ln() + 0.3 * (0.95_f64).ln() + 0.4 * daily_rf;

        assert!((return_data.returns - expected_return).abs() < 1e-10);
    }

    #[test]
    fn test_price_df_conversion() -> Result<(), PolarsError> {
        // Create sample DataFrame with explicit i64 values
        let dates = Series::new(
            "date".into(),
            &[
                1714521600_i64, // 2024-05-01 00:00:00 UTC
                1714608000_i64, // 2024-05-02 00:00:00 UTC
                1714694400_i64, // 2024-05-03 00:00:00 UTC
            ],
        );

        // Create price series for multiple assets
        let prices_aapl = Series::new("AAPL".into(), &[150.0, 152.5, 151.0]);
        let prices_googl = Series::new("GOOGL".into(), &[2800.0, 2850.0, 2825.0]);
        let prices_msft = Series::new("MSFT".into(), &[280.0, 285.0, 282.5]);

        // Construct DataFrame
        let df = DataFrame::new(vec![
            dates.into(),
            prices_aapl.into(),
            prices_googl.into(),
            prices_msft.into(),
        ])?;

        // Print DataFrame head
        println!("Test DataFrame:\n{}", df.head(Some(5)));

        // Convert to PriceData
        let price_data = df_to_price_data(&df)?;

        // Test basic structure
        assert_eq!(price_data.len(), 3, "Should have 3 time periods");

        // Test first time period
        let first_period = &price_data[0];
        assert_eq!(
            first_period.timestamp,
            time::OffsetDateTime::from_unix_timestamp(1714521600)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
            "First timestamp should match"
        );

        // Test price values for first period
        assert_eq!(
            first_period
                .prices
                .get(&Arc::from("AAPL".to_string()))
                .unwrap(),
            &150.0,
            "AAPL price should match for first period"
        );
        assert_eq!(
            first_period
                .prices
                .get(&Arc::from("GOOGL".to_string()))
                .unwrap(),
            &2800.0,
            "GOOGL price should match for first period"
        );
        assert_eq!(
            first_period
                .prices
                .get(&Arc::from("MSFT".to_string()))
                .unwrap(),
            &280.0,
            "MSFT price should match for first period"
        );

        // Test all tickers are present
        let expected_tickers: Vec<String> = vec!["AAPL", "GOOGL", "MSFT"]
            .into_iter()
            .map(String::from)
            .collect();

        for period in &price_data {
            for ticker in &expected_tickers {
                assert!(
                    period.prices.contains_key(&Arc::from(ticker.clone())),
                    "Missing ticker {} in period {:?}",
                    ticker,
                    period.timestamp
                );
            }
        }

        // Test price continuity
        let second_period = &price_data[1];
        assert_eq!(
            second_period
                .prices
                .get(&Arc::from("AAPL".to_string()))
                .unwrap(),
            &152.5,
            "AAPL price should match for second period"
        );

        // Test timestamp ordering
        for i in 1..price_data.len() {
            assert!(
                price_data[i].timestamp > price_data[i - 1].timestamp,
                "Timestamps should be strictly increasing"
            );
        }

        Ok(())
    }

    #[test]
    fn test_weight_df_conversion() -> Result<(), PolarsError> {
        // Create sample DataFrame with explicit i64 values
        let dates = Series::new(
            "insight_date".into(),
            &[
                1714521600_i64, // 2024-05-01 00:00:00 UTC
                1714608000_i64, // 2024-05-02 00:00:00 UTC
                1714694400_i64, // 2024-05-03 00:00:00 UTC
            ],
        );

        // Create weight series for multiple assets
        // Day 1: 80% invested (30% AAPL, 30% GOOGL, 20% MSFT), 20% cash
        // Day 2: 70% invested (25% AAPL, 25% GOOGL, 20% MSFT), 30% cash
        // Day 3: 90% invested (35% AAPL, 35% GOOGL, 20% MSFT), 10% cash
        let weights_aapl = Series::new("AAPL".into(), &[0.30, 0.25, 0.35]);
        let weights_googl = Series::new("GOOGL".into(), &[0.30, 0.25, 0.35]);
        let weights_msft = Series::new("MSFT".into(), &[0.20, 0.20, 0.20]);

        // Construct DataFrame
        let df = DataFrame::new(vec![
            dates.into(),
            weights_aapl.into(),
            weights_googl.into(),
            weights_msft.into(),
        ])?;

        // Print DataFrame head
        println!("Test Weight DataFrame:\n{}", df.head(Some(5)));

        // Convert to PortfolioWeightData
        let weight_data = df_to_weight_data(&df)?;

        // Test basic structure
        assert_eq!(weight_data.len(), 3, "Should have 3 time periods");

        // Test first time period
        let first_period = &weight_data[0];
        assert_eq!(
            first_period.timestamp,
            time::OffsetDateTime::from_unix_timestamp(1714521600)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
            "First timestamp should match"
        );

        // Test weight values for first period
        assert_eq!(
            first_period
                .positions
                .get(&Arc::from("AAPL".to_string()))
                .unwrap(),
            &0.30,
            "AAPL weight should match for first period"
        );
        assert_eq!(
            first_period
                .positions
                .get(&Arc::from("GOOGL".to_string()))
                .unwrap(),
            &0.30,
            "GOOGL weight should match for first period"
        );
        assert_eq!(
            first_period
                .positions
                .get(&Arc::from("MSFT".to_string()))
                .unwrap(),
            &0.20,
            "MSFT weight should match for first period"
        );

        // Test all tickers are present
        let expected_tickers: Vec<String> = vec!["AAPL", "GOOGL", "MSFT"]
            .into_iter()
            .map(String::from)
            .collect();

        for period in &weight_data {
            // Test that weights sum to less than or equal to 1.0
            let total_weight: f64 = period.positions.values().sum();
            assert!(
                total_weight <= 1.0 + 1e-10,
                "Weights should sum to less than or equal to 1.0, got {}",
                total_weight
            );

            // Test all tickers are present
            for ticker in &expected_tickers {
                assert!(
                    period.positions.contains_key(&Arc::from(ticker.clone())),
                    "Missing ticker {} in period {:?}",
                    ticker,
                    period.timestamp
                );
            }
        }

        // Test specific cash positions
        let periods = [
            (0, 0.80), // First period should be 80% invested
            (1, 0.70), // Second period should be 70% invested
            (2, 0.90), // Third period should be 90% invested
        ];

        for (idx, expected_invested) in periods {
            let total_invested: f64 = weight_data[idx].positions.values().sum();
            assert!(
                (total_invested - expected_invested).abs() < 1e-10,
                "Period {} should have {:.2} invested, got {:.2}",
                idx,
                expected_invested,
                total_invested
            );
        }

        // Test timestamp ordering
        for i in 1..weight_data.len() {
            assert!(
                weight_data[i].timestamp > weight_data[i - 1].timestamp,
                "Timestamps should be strictly increasing"
            );
        }

        Ok(())
    }
}
