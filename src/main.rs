pub mod backtester;
pub mod input_handler;
pub mod metrics;
pub mod tests;

mod lazy;

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

    let lazy_df = lazy::lazy_backtest(&price_df, &weights_df)?;
    println!("{:?}", lazy_df.collect().unwrap());

    Ok(())
}
