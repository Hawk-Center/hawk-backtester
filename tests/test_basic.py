import polars as pl
from hawk_backtester import initialize_backtester

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

def test_initialize_backtester():
    model_state, model_insights = initialize_example_data()
    assert initialize_backtester(model_state, model_insights)

if __name__ == "__main__":
    print("Testing backtester initialization...")
    model_state, model_insights = initialize_example_data()
    success = initialize_backtester(model_state, model_insights)
    print(model_state.head())
    print(model_insights.head())
    print(f"Initialization {'successful' if success else 'failed'}")

 