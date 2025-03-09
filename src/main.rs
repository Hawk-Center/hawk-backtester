pub mod backtester;
pub mod input_handler;
pub mod metrics;
pub mod tests;

use backtester::{Backtester, PriceData, WeightEvent};
use input_handler::{parse_price_df, parse_weights_df};
use polars::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // For Testing purposes, use the data/prices.csv and data/weights.csv files.
    // Open CSV files using std::fs::File.
    let price_file = File::open("data/prices.csv")?;
    let price_df = CsvReader::new(price_file).finish()?;

    // For Testing purposes, use the data/weights.csv file.
    let weights_file = File::open("data/weights.csv")?;
    let weights_df = CsvReader::new(weights_file).finish()?;

    // Convert DataFrames into the internal types.
    let prices: Vec<PriceData> = parse_price_df(&price_df)?;
    let weight_events: Vec<WeightEvent> = parse_weights_df(&weights_df)?;

    // Get start date before moving prices
    let start_date = weight_events[0].timestamp;

    // Create the backtester.
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 10_000.0, // For defult testing purposes, use 10_000.0.
        start_date,
    };

    // Run the simulation and output the DataFrame tail.
    let (df, metrics) = backtester.run()?;
    println!("Tail of backtest results:\n{:?}", df.tail(Some(5)));
    println!("Metrics:\n{:?}", metrics);
    Ok(())
}
