use polars::prelude::SeriesTrait;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use time::Date;
// Ensure that the `time` crate is compiled with features "formatting" and "parsing".
use time::format_description;

use crate::backtester::{PriceData, WeightEvent};

/// Validates the structure and content of a price DataFrame.
///
/// Checks:
/// 1. Required columns exist
/// 2. Date column is string type
/// 3. Price columns are numeric
/// 4. No null values in required fields
///
/// # Errors
/// Returns a PolarsError with a descriptive message if validation fails.
fn validate_price_df(df: &DataFrame) -> Result<(), PolarsError> {
    // Check date column exists and is string type
    let date_col = df.column("date")?;
    if !matches!(date_col.dtype(), DataType::String) {
        return Err(PolarsError::ComputeError(
            "Date column must be string type".into(),
        ));
    }

    // Check at least one price column exists
    if df.width() < 2 {
        return Err(PolarsError::ComputeError(
            "DataFrame must have at least one price column besides date".into(),
        ));
    }

    // Check all non-date columns are numeric
    for col_name in df.get_column_names() {
        if col_name != "date" {
            let col = df.column(col_name)?;
            if !matches!(col.dtype(), DataType::Float64 | DataType::Int64) {
                return Err(PolarsError::ComputeError(
                    format!("Column {} must be numeric type", col_name).into(),
                ));
            }
        }
    }

    // Check for null values
    let null_counts = df.null_count();
    if null_counts.iter().any(|col| col.sum().unwrap_or(0) > 0) {
        return Err(PolarsError::ComputeError(
            "DataFrame contains null values".into(),
        ));
    }

    Ok(())
}

/// Validates the structure and content of a weights DataFrame.
///
/// Checks:
/// 1. Required columns exist
/// 2. Date column is string type
/// 3. Weight columns are numeric
/// 4. Weights are between -1 and 1
/// 5. No null values in required fields
///
/// # Errors
/// Returns a PolarsError with a descriptive message if validation fails.
fn validate_weights_df(df: &DataFrame) -> Result<(), PolarsError> {
    // Check date column exists and is string type
    let date_col = df.column("date")?;
    if !matches!(date_col.dtype(), DataType::String) {
        return Err(PolarsError::ComputeError(
            "Date column must be string type".into(),
        ));
    }

    // Check at least one weight column exists
    if df.width() < 2 {
        return Err(PolarsError::ComputeError(
            "DataFrame must have at least one weight column besides date".into(),
        ));
    }

    // Check all non-date columns are numeric and weights are valid
    for col_name in df.get_column_names() {
        if col_name != "date" {
            let col = df.column(col_name)?;
            if !matches!(col.dtype(), DataType::Float64 | DataType::Int64) {
                return Err(PolarsError::ComputeError(
                    format!("Column {} must be numeric type", col_name).into(),
                ));
            }

            // Check weight values are between -1 and 1
            let values: Vec<f64> = col.f64()?.into_iter().map(|v| v.unwrap_or(0.0)).collect();
            if values.iter().any(|&w| w < -1.0 || w > 1.0) {
                return Err(PolarsError::ComputeError(
                    format!("Weights in column {} must be between -1 and 1", col_name).into(),
                ));
            }
        }
    }

    // Check for null values
    let null_counts = df.null_count();
    if null_counts.iter().any(|col| col.sum().unwrap_or(0) > 0) {
        return Err(PolarsError::ComputeError(
            "DataFrame contains null values".into(),
        ));
    }

    Ok(())
}

/// Validates date formats in a DataFrame's date column.
///
/// Checks:
/// 1. All dates are in Y/M/D format (accepts both YYYY/MM/DD and YYYY-MM-DD)
/// 2. All dates are valid calendar dates
///
/// # Errors
/// Returns a PolarsError if any date is invalid or in wrong format.
fn validate_dates(df: &DataFrame) -> Result<(), PolarsError> {
    let date_col = df.column("date")?;
    let date_strs = date_col.str()?;
    let date_format = format_description::parse("[year]/[month]/[day]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating date format: {:?}", e).into())
    })?;
    let iso_format = format_description::parse("[year]-[month]-[day]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating ISO date format: {:?}", e).into())
    })?;

    for i in 0..df.height() {
        let date_str = date_strs.get(i).unwrap();
        // Determine if the date is in ISO format or slash format
        let (parts, separator): (Vec<&str>, &str) = if date_str.contains('-') {
            (date_str.split('-').collect(), "-")
        } else if date_str.contains('/') {
            (date_str.split('/').collect(), "/")
        } else {
            return Err(PolarsError::ComputeError(
                format!("Invalid date format at row {}: {}", i + 1, date_str).into(),
            ));
        };

        if parts.len() != 3 {
            return Err(PolarsError::ComputeError(
                format!("Invalid date format at row {}: {}", i + 1, date_str).into(),
            ));
        }

        let year = parts[0];
        let month = format!(
            "{:02}",
            parts[1].parse::<u8>().map_err(|_| {
                PolarsError::ComputeError(
                    format!("Invalid month at row {}: {}", i + 1, parts[1]).into(),
                )
            })?
        );
        let day = format!(
            "{:02}",
            parts[2].parse::<u8>().map_err(|_| {
                PolarsError::ComputeError(
                    format!("Invalid day at row {}: {}", i + 1, parts[2]).into(),
                )
            })?
        );

        // Try parsing with both formats
        let normalized_date = format!("{}/{}/{}", year, month, day);
        let iso_date = format!("{}-{}-{}", year, month, day);

        if Date::parse(&normalized_date, &date_format).is_err()
            && Date::parse(&iso_date, &iso_format).is_err()
        {
            return Err(PolarsError::ComputeError(
                format!("Invalid date format at row {}: {}", i + 1, date_str).into(),
            ));
        }
    }

    Ok(())
}

/// Validates and sorts price data by date.
/// Returns a new Vec<PriceData> sorted by timestamp.
fn sort_price_data(data: Vec<PriceData>) -> Vec<PriceData> {
    let mut sorted_data = data;
    sorted_data.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    sorted_data
}

/// Parses a price DataFrame into a vector of `PriceData`.
///
/// The input DF must include a "date" column (UTF8) in Y/M/D format (e.g., "2023/01/15" or "2023-01-15")
/// and one column per security with closing prices. Price data is automatically sorted by date in ascending order.
///
/// # Errors
/// Returns an error if validation fails or if data cannot be parsed.
pub fn parse_price_df(df: &DataFrame) -> Result<Vec<PriceData>, PolarsError> {
    // Validate DataFrame structure and content
    validate_price_df(df)?;
    validate_dates(df)?;

    // Get the date column
    let ts_series = df.column("date")?;
    // Extract string values using Series's string representation methods
    let ts_chunked = ts_series.str()?;
    let column_names = df.get_column_names();
    let mut prices_vec = Vec::with_capacity(df.height());

    // Create formats for parsing dates
    let date_format = format_description::parse("[year]/[month]/[day]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating date format: {:?}", e).into())
    })?;
    let iso_format = format_description::parse("[year]-[month]-[day]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating ISO date format: {:?}", e).into())
    })?;

    for i in 0..df.height() {
        let ts_str = ts_chunked
            .get(i)
            .ok_or_else(|| PolarsError::ComputeError("Missing date value".into()))?;

        // Determine format and split accordingly
        let (parts, is_iso): (Vec<&str>, bool) = if ts_str.contains('-') {
            (ts_str.split('-').collect(), true)
        } else {
            (ts_str.split('/').collect(), false)
        };

        let year = parts[0];
        let month = format!("{:02}", parts[1].parse::<u8>().unwrap());
        let day = format!("{:02}", parts[2].parse::<u8>().unwrap());

        // Try parsing with the appropriate format
        let date = if is_iso {
            let iso_date = format!("{}-{}-{}", year, month, day);
            Date::parse(&iso_date, &iso_format)
        } else {
            let normalized_date = format!("{}/{}/{}", year, month, day);
            Date::parse(&normalized_date, &date_format)
        }
        .map_err(|e| PolarsError::ComputeError(format!("Error parsing date: {:?}", e).into()))?;

        let mut prices = HashMap::new();
        for col_name in &column_names {
            if *col_name == "date" {
                continue;
            }
            let col = df.column(col_name)?;
            let price_val = col.get(i)?;
            let price: f64 = match price_val {
                AnyValue::Float64(p) => p,
                AnyValue::Int64(p) => p as f64,
                _ => price_val.extract().unwrap_or(0.0),
            };
            prices.insert(Arc::from(col_name.to_string()), price);
        }
        prices_vec.push(PriceData {
            timestamp: date,
            prices,
        });
    }

    // Sort price data before returning
    Ok(sort_price_data(prices_vec))
}

/// Validates and sorts weight events by date.
/// Returns a new Vec<WeightEvent> sorted by timestamp.
fn sort_weight_events(events: Vec<WeightEvent>) -> Vec<WeightEvent> {
    let mut sorted_events = events;
    sorted_events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    sorted_events
}

/// Parses a weights DataFrame into a vector of `WeightEvent`.
///
/// For each row, the DF must include a "date" column (UTF8) in Y/M/D format (e.g., "2023/01/15" or "2023-01-15")
/// and one column per security with weights (the value 0.0 or null may indicate no allocation for that security).
/// Weight events are automatically sorted by date in ascending order.
///
/// # Errors
/// Returns an error if validation fails or if data cannot be parsed.
pub fn parse_weights_df(df: &DataFrame) -> Result<Vec<WeightEvent>, PolarsError> {
    // Validate DataFrame structure and content
    validate_weights_df(df)?;
    validate_dates(df)?;

    let ts_series = df.column("date")?;
    let ts_chunked = ts_series.str()?;
    let column_names = df.get_column_names();
    let mut events = Vec::with_capacity(df.height());

    // Create formats for parsing dates
    let date_format = format_description::parse("[year]/[month]/[day]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating date format: {:?}", e).into())
    })?;
    let iso_format = format_description::parse("[year]-[month]-[day]").map_err(|e| {
        PolarsError::ComputeError(format!("Error creating ISO date format: {:?}", e).into())
    })?;

    for i in 0..df.height() {
        let ts_str = ts_chunked
            .get(i)
            .ok_or_else(|| PolarsError::ComputeError("Missing date value".into()))?;

        // Determine format and split accordingly
        let (parts, is_iso): (Vec<&str>, bool) = if ts_str.contains('-') {
            (ts_str.split('-').collect(), true)
        } else {
            (ts_str.split('/').collect(), false)
        };

        let year = parts[0];
        let month = format!("{:02}", parts[1].parse::<u8>().unwrap());
        let day = format!("{:02}", parts[2].parse::<u8>().unwrap());

        // Try parsing with the appropriate format
        let date = if is_iso {
            let iso_date = format!("{}-{}-{}", year, month, day);
            Date::parse(&iso_date, &iso_format)
        } else {
            let normalized_date = format!("{}/{}/{}", year, month, day);
            Date::parse(&normalized_date, &date_format)
        }
        .map_err(|e| PolarsError::ComputeError(format!("Error parsing date: {:?}", e).into()))?;

        let mut weights = HashMap::new();
        for col_name in &column_names {
            if *col_name == "date" {
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
        events.push(WeightEvent {
            timestamp: date,
            weights,
        });
    }

    // Sort events by date before returning
    Ok(sort_weight_events(events))
}
