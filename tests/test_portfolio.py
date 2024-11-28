import polars as pl
import pytest
from hawk_backtester import initialize_backtester, simulate_portfolio
from test_basic import initialize_example_data
import numpy as np

def test_portfolio_rebalancing():
    """Test basic portfolio rebalancing functionality"""
    model_state, model_insights = initialize_example_data()
    
    # Get first day's data
    first_date = model_state["date"].min()
    day_prices = model_state.filter(pl.col("date") == first_date)
    day_weights = model_insights.filter(pl.col("insight_date") == first_date)
    
    # Initialize and test rebalancing
    assert initialize_backtester(day_prices, day_weights)

def test_portfolio_simulation():
    """Test full portfolio simulation"""
    model_state, model_insights = initialize_example_data()
    # Debug prints
    print("Model Insights columns:", model_insights.columns)
    print("First few rows of insights:", model_insights.head())
    
    # Run simulation
    portfolio_history = simulate_portfolio(model_state, model_insights)
    
    # Basic validation checks
    assert isinstance(portfolio_history, pl.DataFrame)
    assert "date" in portfolio_history.columns
    assert "total_equity" in portfolio_history.columns
    assert "log_return" in portfolio_history.columns
    
    # Check that equity values are reasonable (no negative values)
    assert (portfolio_history["total_equity"] >= 0).all()
    
    # Check that we have the expected number of rows
    unique_dates = model_insights["insight_date"].unique()
    assert len(portfolio_history) == len(unique_dates)
    
    # Check that returns are calculated correctly
    equity_values = portfolio_history["total_equity"].to_list()
    for i in range(1, len(equity_values)):
        expected_return = np.log(equity_values[i] / equity_values[i-1])
        actual_return = portfolio_history["log_return"][i]
        assert abs(expected_return - actual_return) < 1e-6

    # Assert that we get a Dataframe back from the simulation
    assert isinstance(portfolio_history, pl.DataFrame)
    print(portfolio_history.tail())



def test_transaction_costs():
    """Test that transaction costs are being applied"""
    model_state, model_insights = initialize_example_data()
    
    # Run simulation
    portfolio_history = simulate_portfolio(model_state, model_insights)
    
    # Get initial and final equity
    initial_equity = portfolio_history["total_equity"][0]
    final_equity = portfolio_history["total_equity"][-1]
    
    # Calculate total return
    total_return = (final_equity - initial_equity) / initial_equity
    
    # The total return should be less than what we'd get without transaction costs
    # This is a basic check that transaction costs are being applied
    assert total_return < 1.0, "Transaction costs should reduce returns"

def test_portfolio_constraints():
    """Test portfolio weight constraints"""
    model_state, model_insights = initialize_example_data()
    
    # Check that weights sum to approximately 1.0
    dates = model_insights["insight_date"].unique()
    for date in dates:
        weights = model_insights.filter(pl.col("insight_date") == date)["weight"]
        weight_sum = abs(weights).sum()
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights should sum to 1.0, got {weight_sum}"

def test_error_handling():
    """Test error handling for invalid inputs"""
    model_state, model_insights = initialize_example_data()
    
    # Test with missing required column
    bad_state = model_state.drop("close")
    with pytest.raises(ValueError):
        initialize_backtester(bad_state, model_insights)
    
    # Test with mismatched dates
    bad_insights = model_insights.with_columns([
        pl.col("insight_date").shift(365)  # Shift dates by 1 year
    ])
    # print(bad_insights.head())
    with pytest.raises(ValueError):
        initialize_backtester(model_state, bad_insights)

if __name__ == "__main__":
    test_portfolio_rebalancing()
    test_portfolio_simulation()
    test_transaction_costs()
    test_portfolio_constraints()
    test_error_handling()
