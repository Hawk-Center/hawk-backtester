use polars::prelude::SeriesTrait;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use time::Date;
// Ensure that the `time` crate is compiled with features "formatting" and "parsing".
use time::format_description;

use crate::backtester::{PriceData, WeightEvent};

/// Parses a price DataFrame into a vector of `PriceData`.
///
/// The input DF must include a "timestamp" column (UTF8) in m/d/y format (e.g., "01/15/2023")
/// and one column per security with closing prices.
///
/// # Errors
/// Returns an error if the timestamp column is not present or cannot be parsed.
pub fn parse_price_df(df: &DataFrame) -> Result<Vec<PriceData>, PolarsError> {
    // Get the timestamp column
    let ts_series = df.column("timestamp")?;
    // Extract string values using Series's string representation methods
    let ts_chunked = ts_series.str()?;
    let column_names = df.get_column_names();
    let mut prices_vec = Vec::with_capacity(df.height());

    // Create a format for parsing dates in m/d/y format
    let date_format = format_description::parse("[month]/[day]/[year]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating date format: {:?}", e).into())
    })?;

    for i in 0..df.height() {
        let ts_str = ts_chunked
            .get(i)
            .ok_or_else(|| PolarsError::ComputeError("Missing timestamp value".into()))?;
        let timestamp = Date::parse(ts_str, &date_format).map_err(|e| {
            PolarsError::ComputeError(format!("Error parsing date: {:?}", e).into())
        })?;
        let mut prices = HashMap::new();
        for col_name in &column_names {
            if *col_name == "timestamp" {
                continue;
            }
            let col = df.column(col_name)?;
            // Unwrap the value using `?` because get() now returns a Result.
            let price_val = col.get(i)?;
            let price: f64 = match price_val {
                AnyValue::Float64(p) => p,
                AnyValue::Int64(p) => p as f64,
                _ => price_val.extract().unwrap_or(0.0),
            };
            prices.insert(Arc::from(col_name.to_string()), price);
        }
        prices_vec.push(PriceData { timestamp, prices });
    }
    Ok(prices_vec)
}

/// Parses a weights DataFrame into a vector of `WeightEvent`.
///
/// For each row, the DF must include a "timestamp" column (UTF8) in m/d/y format (e.g., "01/15/2023")
/// and one column per security with weights (the value 0.0 or null may indicate no allocation for that security).
///
/// # Errors
/// Returns an error if the timestamp column is not present or cannot be parsed.
pub fn parse_weights_df(df: &DataFrame) -> Result<Vec<WeightEvent>, PolarsError> {
    let ts_series = df.column("timestamp")?;
    let ts_chunked = ts_series.str()?;
    let column_names = df.get_column_names();
    let mut events = Vec::with_capacity(df.height());

    // Create a format for parsing dates in m/d/y format
    let date_format = format_description::parse("[month]/[day]/[year]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating date format: {:?}", e).into())
    })?;

    for i in 0..df.height() {
        let ts_str = ts_chunked
            .get(i)
            .ok_or_else(|| PolarsError::ComputeError("Missing timestamp value".into()))?;
        let timestamp = Date::parse(ts_str, &date_format).map_err(|e| {
            PolarsError::ComputeError(format!("Error parsing date: {:?}", e).into())
        })?;
        let mut weights = HashMap::new();
        for col_name in &column_names {
            if *col_name == "timestamp" {
                continue;
            }
            let col = df.column(col_name)?;
            let weight_val = col.get(i)?;
            let weight: f64 = match weight_val {
                AnyValue::Float64(w) => w,
                AnyValue::Int64(w) => w as f64,
                _ => weight_val.extract().unwrap_or(0.0),
            };
            weights.insert(Arc::from(col_name.to_string()), weight);
        }
        events.push(WeightEvent { timestamp, weights });
    }
    Ok(events)
}
