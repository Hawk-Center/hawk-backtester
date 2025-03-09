import polars as pl
import rust_backtester  # This is your built Rust extension


def main():
    # Create example Polars DataFrames

    # Price data example: You may define your columns as needed.
    price_df = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "price": [100.0, 105.0, 102.0],
        }
    )

    # Weight data example: Ensure the schema aligns with what your Rust code expects.
    weights_df = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "weight": [0.3, 0.4, 0.3],
        }
    )

    # Convert DataFrames into Arrow IPC bytes. This conversion is fast and memory-efficient.
    price_ipc = price_df.to_arrow().to_pybytes()
    weight_ipc = weights_df.to_arrow().to_pybytes()

    # Instantiate the backtester from the Rust package.
    bt = rust_backtester.Backtester(10000.0)

    # Load data into the backtester.
    bt.set_prices(price_ipc)
    bt.set_weights(weight_ipc)

    # Run the backtest.
    result_ipc = bt.run_backtest()

    # Convert the returned Arrow IPC bytes back into a Polars DataFrame.
    result_df = pl.read_ipc(result_ipc)

    # Display the result
    print("Backtest Result:")
    print(result_df)


if __name__ == "__main__":
    main()
