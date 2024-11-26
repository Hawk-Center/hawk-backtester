import polars as pl
import pytest
from hawk_backtester import initialize_backtester
# from .test_basic import initialize_example_data


def initialize_example_data():
    example_model_state = pl.read_parquet("tests/data/example_model_state.parquet")
    example_model_insights = pl.read_parquet("tests/data/example_insights.parquet")
    
    # Convert datetime[ns, UTC] to Date for model_state
    example_model_state = example_model_state.with_columns([
        pl.col("date").cast(pl.Date).alias("date")
    ])
    
    # Convert string to Date for model_insights
    example_model_insights = example_model_insights.with_columns([
        pl.col("insight_date").str.strptime(pl.Date, format="%Y-%m-%d %H:%M:%S").alias("insight_date")
    ])
    
    # Sort by date
    example_model_state = example_model_state.sort("date")
    example_model_insights = example_model_insights.sort("insight_date")
    
    # Select only the columns we want
    example_model_state = example_model_state.select([
        "date", 
        "ticker", 
        "open", 
        "high", 
        "low", 
        "close", 
        "volume", 
        "open_interest"
    ])
    
    return example_model_state, example_model_insights

def test_portfolio_rebalancing():
    """
    Test basic portfolio rebalancing functionality
    """
    model_state, model_insights = initialize_example_data()
    
    # Get first day's data
    first_date = model_state["date"].min()
    day_prices = model_state.filter(pl.col("date") == first_date)
    day_weights = model_insights.filter(pl.col("insight_date") == first_date)
    
    # Initialize and test rebalancing
    assert initialize_backtester(day_prices, day_weights) 