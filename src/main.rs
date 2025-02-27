pub mod backtester;
pub mod input_handler;

use backtester::{Backtester, PriceData, WeightEvent};
use input_handler::{parse_price_df, parse_weights_df};
use polars::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open CSV files using std::fs::File.
    let price_file = File::open("data/prices.csv")?;
    let price_df = CsvReader::new(price_file).finish()?;

    let weights_file = File::open("data/weights.csv")?;
    let weights_df = CsvReader::new(weights_file).finish()?;

    // Convert DataFrames into the internal types.
    let prices: Vec<PriceData> = parse_price_df(&price_df)?;
    let weight_events: Vec<WeightEvent> = parse_weights_df(&weights_df)?;

    // Create the backtester.
    let backtester = Backtester {
        prices,
        weight_events,
        initial_value: 10_000.0,
    };

    // Run the simulation and output the DataFrame tail.
    let df = backtester.run()?;
    println!("Tail of backtest results:\n{:?}", df.tail(Some(5)));

    Ok(())
}
