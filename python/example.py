"""
Example usage of the Hawk Backtester from Python.
"""

import polars as pl
import hawk_backtester
from hawk_backtester import HawkBacktester



def test_input():
    df = pl.read_csv("data/Updated_Backtester_Input.csv", infer_schema_length=1000)
    prices_df = df.select(
        [
            "date",
            "SLV_adjClose",
            "SPY_adjClose",
            "GLD_adjClose",
            "TLT_adjClose",
            "USO_adjClose",
            "UNG_adjClose",
        ]
    )
    # Rename price columns to match the asset names expected by the backtester
    prices_df = prices_df.rename(
        {
            "SLV_adjClose": "SLV",
            "SPY_adjClose": "SPY",
            "GLD_adjClose": "GLD",
            "TLT_adjClose": "TLT",
            "USO_adjClose": "USO",
            "UNG_adjClose": "UNG",
        }
    )
    prices_df = prices_df.fill_null(strategy="forward")
    prices_df = prices_df.fill_null(strategy="backward")

    weights_df = df.select(
        ["date", "SLV_wgt", "SPY_wgt", "GLD_wgt", "TLT_wgt", "USO_wgt", "UNG_wgt"]
    )
    # Rename weight columns to match the asset names expected by the backtester
    weights_df = weights_df.rename(
        {
            "SLV_wgt": "SLV",
            "SPY_wgt": "SPY",
            "GLD_wgt": "GLD",
            "TLT_wgt": "TLT",
            "USO_wgt": "USO",
            "UNG_wgt": "UNG",
        }
    )
    # Drop Nulls
    # Drop rows with null values in the weights dataframe
    print(weights_df)
    weights_df = weights_df.drop_nulls()
    print(weights_df)

    print("=" * 80)
    print("Running Backtest #1: With Slippage Only (No Commissions)")
    print("=" * 80)
    
    # Example with slippage only
    backtester_no_commissions = HawkBacktester(initial_value=1_000_000.0, slippage_bps=1.0)
    results_no_commissions = backtester_no_commissions.run(prices_df, weights_df)
    
    # Extract metrics
    metrics_df_no_commissions = results_no_commissions["backtest_metrics"]
    
    def get_metric(metrics_df, metric_name):
        return metrics_df.filter(pl.col("metric") == metric_name)["value"][0]
    
    total_return_no_commissions = get_metric(metrics_df_no_commissions, "total_return")
    slippage_cost_no_commissions = get_metric(metrics_df_no_commissions, "cumulative_slippage_cost")
    commission_cost_no_commissions = get_metric(metrics_df_no_commissions, "cumulative_commission_cost")
    num_trades = get_metric(metrics_df_no_commissions, "num_trades")
    
    print(f"Total Return: {total_return_no_commissions:.4%}")
    print(f"Number of Trades: {int(num_trades)}")
    print(f"Cumulative Slippage Cost: ${slippage_cost_no_commissions:,.2f}")
    print(f"Cumulative Commission Cost: ${commission_cost_no_commissions:,.2f}")
    
    print("\n" + "=" * 80)
    print("Running Backtest #2: With Slippage + IBKR Pro Fixed Commissions")
    print("=" * 80)
    
    # Example with IBKR Pro Fixed commission model
    backtester_with_commissions = HawkBacktester(
        initial_value=1_000_000.0, 
        slippage_bps=1.0, 
        fee_model="ibkr_pro_fixed"
    )
    results_with_commissions = backtester_with_commissions.run(prices_df, weights_df)
    
    # Extract metrics
    metrics_df_with_commissions = results_with_commissions["backtest_metrics"]
    
    total_return_with_commissions = get_metric(metrics_df_with_commissions, "total_return")
    slippage_cost_with_commissions = get_metric(metrics_df_with_commissions, "cumulative_slippage_cost")
    commission_cost_with_commissions = get_metric(metrics_df_with_commissions, "cumulative_commission_cost")
    
    print(f"Total Return: {total_return_with_commissions:.4%}")
    print(f"Number of Trades: {int(num_trades)}")
    print(f"Cumulative Slippage Cost: ${slippage_cost_with_commissions:,.2f}")
    print(f"Cumulative Commission Cost: ${commission_cost_with_commissions:,.2f}")
    
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    
    return_difference = total_return_no_commissions - total_return_with_commissions
    total_costs = commission_cost_with_commissions + slippage_cost_with_commissions
    
    print(f"Return Impact from Commissions: {return_difference:.4%}")
    print(f"Total Trading Costs (Slippage + Commission): ${total_costs:,.2f}")
    print(f"Average Commission per Trade: ${commission_cost_with_commissions / num_trades:,.2f}")
    
    # Save results to CSV files
    results_df = results_with_commissions["backtest_results"]
    metrics_df = results_with_commissions["backtest_metrics"]

    results_df.write_csv("backtest_results.csv")
    metrics_df.write_csv("backtest_metrics.csv")

    print(f"\nResults saved to backtest_results.csv and backtest_metrics.csv")



if __name__ == "__main__":

    # Print the version of the Hawk Backtester
    # print(f"Hawk Backtester version: {hawk_backtester.__version__}")
    test_input()

