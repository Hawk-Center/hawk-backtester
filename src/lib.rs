use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use chrono::{NaiveDate, NaiveDateTime};

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
    date: NaiveDate,
    cash: f64,
    positions: HashMap<String, Position>,  // ticker -> position
    total_equity: f64,
}

#[derive(Debug)]
struct Portfolio {
    positions: HashMap<String, Position>,
    cash: f64,
    transaction_cost_rate: f64,
}

impl Portfolio {
    fn new(initial_cash: f64) -> Self {
        Portfolio {
            positions: HashMap::new(),
            cash: initial_cash,
            transaction_cost_rate: 0.001, // 0.1% transaction cost
        }
    }

    fn rebalance(&mut self, target_weights: &DataFrame, prices: &DataFrame) -> PyResult<()> {
        let total_equity = self.cash + self.positions.iter()
            .map(|(_, pos)| pos.get_position_value())
            .sum::<f64>();

        // Get current prices for each ticker
        let price_map: HashMap<String, f64> = prices.column("ticker")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .iter()
            .zip(
                prices.column("close")
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                    .f64()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                    .iter()
            )
            .filter_map(|(ticker, price)| {
                if let (Some(t), Some(p)) = (ticker.get_str(), price) {
                    Some((t.to_string(), p))
                } else {
                    None
                }
            })
            .collect();

        // Update positions based on target weights
        let mut new_positions = HashMap::new();
        
        // Process each target weight
        for ticker in target_weights.column("ticker")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .iter() {
            let ticker = ticker.get_str()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid ticker"))?;
            
            let weight = target_weights.column("weight")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                .f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                .get(0)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing weight"))?;
                
            let price = *price_map.get(ticker)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Price not found"))?;
            
            let target_value = total_equity * weight;
            let quantity = (target_value / price).round() as i32;
            
            new_positions.insert(ticker.to_string(), Position {
                weight,
                quantity,
                cost_basis: price,
            });
        }
        
        // Calculate transaction costs
        let transaction_cost = self.calculate_transaction_costs(&new_positions, &price_map);
        self.cash -= transaction_cost;
        
        // Update positions
        self.positions = new_positions;
        
        Ok(())
    }
    
    fn calculate_transaction_costs(&self, new_positions: &HashMap<String, Position>, prices: &HashMap<String, f64>) -> f64 {
        let mut total_cost = 0.0;
        
        for (ticker, new_pos) in new_positions {
            let old_quantity = self.positions.get(ticker).map_or(0, |p| p.quantity);
            let quantity_change = (new_pos.quantity - old_quantity).abs();
            
            if let Some(&price) = prices.get(ticker) {
                total_cost += quantity_change as f64 * price * self.transaction_cost_rate;
            }
        }
        
        total_cost
    }
}

impl PortfolioState {
    fn new(date: NaiveDate, cash: f64, positions: HashMap<String, Position>) -> Self {
        let total_equity = cash + positions.iter()
            .map(|(_, pos)| pos.get_position_value())
            .sum::<f64>();
            
        PortfolioState {
            date,
            cash,
            positions,
            total_equity,
        }
    }
}

/// Initialize the backtester with model state and insights
/// 
/// Validates:
/// - Required columns exist in model_state
/// - Date ranges are valid between model_state and insights
#[pyfunction]
fn initialize_backtester(model_state: PyDataFrame, model_insights: PyDataFrame) -> PyResult<bool> {
    let state_df = model_state.0;
    let insights_df = model_insights.0;
    
    // Validate model_state schema
    let required_columns = vec![
        "date", "ticker", "open", "high", "low", "close", "volume", "open_interest"
    ];
    
    for col in required_columns {
        if !state_df.schema().contains(col) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("model_state missing required column: {}", col)
            ));
        }
    }

    // Get date ranges using lazy evaluation
    let state_range = state_df.lazy()
        .select([
            col("date").min().alias("min_date"),
            col("date").max().alias("max_date"),
        ])
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let insights_range = insights_df.lazy()
        .select([
            col("insight_date").min().alias("min_date"),
            col("insight_date").max().alias("max_date"),
        ])
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    println!("Model State Date Range: \n{}", state_range);
    println!("Model Insights Date Range: \n{}", insights_range);

    // Get scalar values for comparison
    let state_min = state_range.column("min_date")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .get(0)
        .map_err(|_e| PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty date range"))?;
    
    let state_max = state_range.column("max_date")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .get(0)
        .map_err(|_e| PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty date range"))?;
    
    let insights_min = insights_range.column("min_date")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .get(0)
        .map_err(|_e| PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty date range"))?;
    
    let insights_max = insights_range.column("max_date")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .get(0)
        .map_err(|_e| PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty date range"))?;

    // Compare date ranges
    if state_min < insights_min || state_max > insights_max {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Date ranges between model_state and insights do not overlap"
        ));
    }
    
    println!("Successfully validated date ranges and schema");
    
    // Create initial portfolio
    let portfolio = Portfolio::new(1_000_000.0); // Start with $1M capital

    Ok(true)
}

/// Simulates the portfolio through time
#[pyfunction]
fn simulate_portfolio(model_state: PyDataFrame, model_insights: PyDataFrame) -> PyResult<PyDataFrame> {
    let state_df = model_state.0.clone();
    let insights_df = model_insights.0.clone();
    
    // Get unique sorted dates from insights
    let dates = insights_df.lazy()
        .select([col("insight_date").unique().alias("insight_date")])
        .collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .column("insight_date")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .datetime()
        .unwrap()
        .into_iter()
        .collect::<Vec<_>>();

    // Create portfolio and history
    let mut portfolio = Portfolio::new(1_000_000.0);
    let mut portfolio_history: Vec<PortfolioState> = Vec::new();
    
    // For each date
    for date in dates.into_iter().flatten() {
        let date = chrono::DateTime::from_timestamp_millis(date).unwrap();
        
        // Get state and insights for this date
        let day_state = state_df.clone().lazy()
            .filter(col("date").eq(lit(date.date_naive())))
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let day_insights = model_insights.0.clone().lazy()
            .filter(col("insight_date").eq(lit(date.naive_utc())))
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            
        // Rebalance portfolio
        portfolio.rebalance(&day_insights, &day_state)?;
        
        // Record state
        let state = PortfolioState::new(
            date.date_naive(),
            portfolio.cash,
            portfolio.positions.clone(),
        );
        portfolio_history.push(state);
    }
    
    // Convert history to DataFrame with log returns
    let dates: Vec<NaiveDate> = portfolio_history.iter().map(|s| s.date).collect();
    let equity: Vec<f64> = portfolio_history.iter().map(|s| s.total_equity).collect();
    let returns: Vec<f64> = equity.windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    
    // Create final DataFrame
    let df = DataFrame::new(vec![
        Series::new("date", dates),
        Series::new("total_equity", equity),
        Series::new("log_return", returns),
    ]).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(PyDataFrame(df))
}

#[pymodule]
fn hawk_backtester(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initialize_backtester, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_portfolio, m)?)?;
    Ok(())
}
