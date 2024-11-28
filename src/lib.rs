use chrono::{DateTime, NaiveDateTime};
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

type SharedDataFrame = Arc<Mutex<DataFrame>>;

// Helper function to convert PolarsError to PyErr
fn polars_to_pyerr(e: PolarsError) -> PyErr {
    PyErr::new::<PyValueError, _>(e.to_string())
}

#[derive(Debug, Clone)]
struct Position {
    weight: f64,
    quantity: i32,
    cost_basis: f64,
}

impl Position {
    fn get_position_value(&self) -> f64 {
        self.quantity as f64 * self.cost_basis
    }
}

#[derive(Debug)]
struct PortfolioState {
    date: NaiveDateTime,
    cash: f64,
    positions: Arc<Mutex<HashMap<String, Position>>>,
    total_equity: f64,
}

impl PortfolioState {
    fn new(
        date: NaiveDateTime,
        cash: f64,
        positions: Arc<Mutex<HashMap<String, Position>>>,
    ) -> Self {
        let total_equity = {
            let positions = positions.lock().unwrap();
            cash + positions
                .iter()
                .map(|(_, pos)| pos.get_position_value())
                .sum::<f64>()
        };

        PortfolioState {
            date,
            cash,
            positions,
            total_equity,
        }
    }
}

#[derive(Debug)]
struct Portfolio {
    positions: Arc<Mutex<HashMap<String, Position>>>,
    cash: Mutex<f64>,
    transaction_cost_rate: f64,
}

impl Portfolio {
    fn new(initial_cash: f64) -> Self {
        Portfolio {
            positions: Arc::new(Mutex::new(HashMap::new())),
            cash: Mutex::new(initial_cash),
            transaction_cost_rate: 0.001,
        }
    }

    fn rebalance(&self, target_weights: &DataFrame, prices: &DataFrame) -> PyResult<()> {
        let total_equity = {
            let cash = self.cash.lock().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to lock cash: {}", e))
            })?;
            let positions = self.positions.lock().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to lock positions: {}", e))
            })?;
            *cash
                + positions
                    .iter()
                    .map(|(_, pos)| pos.get_position_value())
                    .sum::<f64>()
        };

        println!("Rebalancing portfolio:");
        println!("Total equity: {}", total_equity);

        {
            let positions = self.positions.lock().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to lock positions: {}", e))
            })?;
            println!("Current positions: {:?}", *positions);
        }

        println!("Target weights: \n{}", target_weights);
        println!("Prices: \n{}", prices);

        // Get current prices for each ticker
        let price_map: HashMap<String, f64> = {
            let ticker_series = prices
                .column("ticker")
                .map_err(polars_to_pyerr)?
                .utf8()
                .map_err(polars_to_pyerr)?;

            let close_series = prices
                .column("close")
                .map_err(polars_to_pyerr)?
                .f64()
                .map_err(polars_to_pyerr)?;

            ticker_series
                .into_iter()
                .zip(close_series.into_iter())
                .filter_map(|(ticker_opt, price_opt)| {
                    if let (Some(ticker), Some(price)) = (ticker_opt, price_opt) {
                        Some((ticker.to_string(), price))
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Update positions based on target weights
        let mut new_positions = HashMap::new();

        let ticker_series = target_weights
            .column("ticker")
            .map_err(polars_to_pyerr)?
            .utf8()
            .map_err(polars_to_pyerr)?;

        let weight_series = target_weights
            .column("weight")
            .map_err(polars_to_pyerr)?
            .f64()
            .map_err(polars_to_pyerr)?;

        for (ticker_opt, weight_opt) in ticker_series.into_iter().zip(weight_series.into_iter()) {
            let ticker = ticker_opt
                .ok_or_else(|| PyErr::new::<PyValueError, _>("Invalid ticker"))?
                .to_string();
            let weight =
                weight_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("Missing weight"))?;

            let price = *price_map
                .get(&ticker)
                .ok_or_else(|| PyErr::new::<PyValueError, _>("Price not found for ticker"))?;

            let target_value = total_equity * weight;
            let quantity = (target_value / price).round() as i32;

            new_positions.insert(
                ticker,
                Position {
                    weight,
                    quantity,
                    cost_basis: price,
                },
            );
        }

        // Calculate transaction costs
        let transaction_cost = self.calculate_transaction_costs(&new_positions, &price_map)?;

        {
            let mut cash = self.cash.lock().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to lock cash: {}", e))
            })?;
            *cash -= transaction_cost;
        }

        // Update positions with Arc and Mutex
        let mut positions = self.positions.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock positions: {}", e))
        })?;
        *positions = new_positions;

        Ok(())
    }

    fn calculate_transaction_costs(
        &self,
        new_positions: &HashMap<String, Position>,
        prices: &HashMap<String, f64>,
    ) -> PyResult<f64> {
        let positions = self.positions.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock positions: {}", e))
        })?;
        let mut total_cost = 0.0;

        for (ticker, new_pos) in new_positions {
            let old_quantity = positions.get(ticker).map_or(0, |p| p.quantity);
            let quantity_change = (new_pos.quantity - old_quantity).abs();

            if let Some(&price) = prices.get(ticker) {
                total_cost += quantity_change as f64 * price * self.transaction_cost_rate;
            }
        }

        Ok(total_cost)
    }
}

/// Initialize the backtester with model state and insights
///
/// Validates:
/// - Required columns exist in model_state
/// - Date ranges are valid between model_state and insights
#[pyfunction]
fn initialize_backtester(model_state: PyDataFrame, model_insights: PyDataFrame) -> PyResult<bool> {
    let state_df: SharedDataFrame = Arc::new(Mutex::new(model_state.0));
    let insights_df: SharedDataFrame = Arc::new(Mutex::new(model_insights.0));

    // Validate model_state schema
    let required_columns = vec![
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ];

    for col in required_columns {
        let df = state_df.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock state_df: {}", e))
        })?;
        if !df.schema().has_key(col) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "model_state missing required column: {}",
                col
            )));
        }
    }

    // Get date ranges using lazy evaluation
    let state_range = {
        let df = state_df.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock state_df: {}", e))
        })?;
        df.lazy()
            .select([
                col("date").min().alias("min_date"),
                col("date").max().alias("max_date"),
            ])
            .collect()
            .map_err(polars_to_pyerr)?
    };

    let insights_range = {
        let df = insights_df.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock insights_df: {}", e))
        })?;
        df.lazy()
            .select([
                col("insight_date").min().alias("min_date"),
                col("insight_date").max().alias("max_date"),
            ])
            .collect()
            .map_err(polars_to_pyerr)?
    };

    println!("Model State Date Range: \n{}", state_range);
    println!("Model Insights Date Range: \n{}", insights_range);

    // Get scalar values for comparison
    let state_min = state_range
        .column("min_date")?
        .datetime()?
        .get(0)
        .ok_or_else(|| PyErr::new::<PyValueError, _>("Empty date range"))?;

    let state_max = state_range
        .column("max_date")?
        .datetime()?
        .get(0)
        .ok_or_else(|| PyErr::new::<PyValueError, _>("Empty date range"))?;

    let insights_min = insights_range
        .column("min_date")?
        .datetime()?
        .get(0)
        .ok_or_else(|| PyErr::new::<PyValueError, _>("Empty date range"))?;

    let insights_max = insights_range
        .column("max_date")?
        .datetime()?
        .get(0)
        .ok_or_else(|| PyErr::new::<PyValueError, _>("Empty date range"))?;

    // Compare date ranges
    if insights_min < state_min || insights_max > state_max {
        return Err(PyErr::new::<PyValueError, _>(
            "Model insights dates must fall within model state date range",
        ));
    }
    // Test that dates are not null
    {
        let df = insights_df.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock insights_df: {}", e))
        })?;
        if df.column("insight_date")?.null_count() > 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Model insights dates must not contain null values",
            ));
        }
    }

    println!("Successfully validated date ranges and schema");

    // Create initial portfolio
    let _portfolio = Portfolio::new(1_000_000.0); // Start with $1M capital

    Ok(true)
}

/// Simulates the portfolio through time
#[pyfunction]
fn simulate_portfolio(
    model_state: PyDataFrame,
    model_insights: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let state_df: SharedDataFrame = Arc::new(Mutex::new(model_state.0));
    let insights_df: SharedDataFrame = Arc::new(Mutex::new(model_insights.0));

    // Debug prints first
    {
        let df = insights_df.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock insights_df: {}", e))
        })?;
        println!(
            "First timestamp value: {:?}",
            df.column("insight_date").map_err(polars_to_pyerr)?.dtype()
        );
    }

    let dates = {
        let df = insights_df.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock insights_df: {}", e))
        })?;
        df.lazy()
            .select([col("insight_date")
                .unique()
                .sort(true)
                .alias("insight_date")])
            .collect()
            .map_err(polars_to_pyerr)?
            .column("insight_date")?
            .datetime()
            .map_err(polars_to_pyerr)?
            .into_iter()
            .filter_map(|opt_dt| opt_dt)
            .collect::<Vec<_>>()
    };

    if dates.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(
            "No valid dates found in insights_df",
        ));
    }

    println!("Number of unique dates: {}", dates.len());

    // Add debug prints
    println!("Parsed dates: {:?}", dates);
    {
        let df = insights_df.lock().map_err(|e| {
            PyErr::new::<PyValueError, _>(format!("Failed to lock insights_df: {}", e))
        })?;
        println!(
            "First timestamp value: {:?}",
            df.column("insight_date")?.datetime()?.get(0)
        );
    }

    // Create portfolio and history
    let portfolio = Arc::new(Portfolio::new(1_000_000.0));
    let mut portfolio_history: Vec<PortfolioState> = Vec::new();

    // For each date
    for date in dates.into_iter() {
        // Get state and insights for this date
        let day_state = {
            let df = state_df.lock().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to lock state_df: {}", e))
            })?;
            df.lazy()
                .filter(col("date").eq(lit(date.into())))
                .collect()
                .map_err(polars_to_pyerr)?
        };

        let day_insights = {
            let df = insights_df.lock().map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to lock insights_df: {}", e))
            })?;
            df.lazy()
                .filter(col("insight_date").eq(lit(date.into())))
                .collect()
                .map_err(polars_to_pyerr)?
        };

        // Rebalance portfolio
        portfolio.rebalance(&day_insights, &day_state)?;

        // Record state
        let cash = *portfolio
            .cash
            .lock()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to lock cash: {}", e)))?;
        let positions = Arc::clone(&portfolio.positions);
        let state = PortfolioState::new(
            DateTime::from_timestamp_micros(date).unwrap().naive_utc(),
            cash,
            positions,
        );
        portfolio_history.push(state);
    }

    // Convert history to DataFrame with log returns
    let dates: Vec<NaiveDateTime> = portfolio_history.iter().map(|s| s.date).collect();
    let equity: Vec<f64> = portfolio_history.iter().map(|s| s.total_equity).collect();
    let mut log_return = vec![0.0];
    for window in equity.windows(2) {
        log_return.push((window[1] / window[0]).ln());
    }

    // Create final DataFrame
    let df = DataFrame::new(vec![
        Series::new("date", dates),
        Series::new("total_equity", equity),
        Series::new("log_return", log_return),
    ])
    .map_err(polars_to_pyerr)?;

    Ok(PyDataFrame(df))
}

#[pymodule]
fn hawk_backtester(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initialize_backtester, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_portfolio, m)?)?;
    Ok(())
}
