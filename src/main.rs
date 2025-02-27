pub mod backtester;

use backtester::{Backtester, PriceData, WeightEvent};
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use time::{Duration, OffsetDateTime};

fn main() {
    // Create example price data
    let now = OffsetDateTime::now_utc();
    let prices = vec![
        PriceData {
            timestamp: now,
            prices: {
                let mut map = HashMap::new();
                // Example assets with their prices.
                map.insert(Arc::from("AAPL"), 150.0);
                map.insert(Arc::from("GOOG"), 2500.0);
                map
            },
        },
        PriceData {
            timestamp: now + Duration::days(1),
            prices: {
                let mut map = HashMap::new();
                // Prices update for the next day.
                map.insert(Arc::from("AAPL"), 155.0);
                map.insert(Arc::from("GOOG"), 2550.0);
                map
            },
        },
    ];

    // Create example weight event(s) for rebalancing.
    let weight_events = vec![WeightEvent {
        timestamp: now,
        weights: {
            let mut map = HashMap::new();
            // Allocate 70% to AAPL and 20% to GOOG (10% in cash).
            map.insert(Arc::from("AAPL"), 0.7);
            map.insert(Arc::from("GOOG"), 0.2);
            map
        },
    }];

    // Create and run the backtester.
    let backtester = Backtester {
        prices,
        weight_events,
        initial_value: 10_000.0,
    };

    // Run the simulation, which now returns a Polars DataFrame.
    let df = backtester.run().expect("Backtest run failed");

    // Print the tail of the DataFrame for logging purposes.
    println!("Tail of backtest results:\n{:?}", df.tail(Some(5)));
}
