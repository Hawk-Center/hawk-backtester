import polars as pl
import numpy as np
import pytest
from hawk_backtester.utils import (
    prepare_price_data,
    prepare_weight_data,
    DataFormatError,
    DataTypeError,
)
from hawk_backtester.metrics import calculate_sharpe_ratio, calculate_metrics


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    data = {
        "date": [
            "2024-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
        ],
        "ticker": ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL"],
        "close": [100.0, 2500.0, 101.0, 2510.0, 102.0, 2520.0],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_weight_data():
    """Create sample weight data for testing."""
    data = {
        "insight_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "AAPL": [0.4, 0.5, 0.6],
        "GOOGL": [0.3, 0.2, 0.3],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    data = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "returns": [0.01, -0.005, 0.02],
        "cumulative_returns": [0.01, 0.005, 0.025],
    }
    return pl.DataFrame(data)


# def test_convert_to_backtester_format(sample_price_data):
#     """Test the convert_to_backtester_format function."""
#     result = convert_model_state_format(sample_price_data)

#     # Check structure
#     assert "date" in result.columns
#     assert "AAPL" in result.columns
#     assert "GOOGL" in result.columns
#     assert result.height == 3  # Three unique dates

#     # Check values
#     assert result["AAPL"][0] == 100.0
#     assert result["GOOGL"][1] == 2510.0


def test_prepare_price_data(sample_price_data):
    """Test the prepare_price_data function."""
    result = prepare_price_data(sample_price_data)

    # Check structure and data types
    assert result["date"].dtype == pl.Int64
    assert result["AAPL"].dtype == pl.Float64
    assert result["GOOGL"].dtype == pl.Float64

    # Check sorting
    dates = result["date"].to_list()
    assert dates == sorted(dates)


def test_prepare_weight_data(sample_weight_data):
    """Test the prepare_weight_data function."""
    result = prepare_weight_data(sample_weight_data)

    # Check structure and data types
    assert result["insight_date"].dtype == pl.Int64
    assert result["AAPL"].dtype == pl.Float64
    assert result["GOOGL"].dtype == pl.Float64

    # Check weight constraints
    total_weights = result["AAPL"] + result["GOOGL"]
    assert (total_weights <= 1.0 + 1e-10).all()


def test_invalid_price_data():
    """Test error handling for invalid price data."""
    # Empty DataFrame
    with pytest.raises(DataFormatError, match="Input DataFrame is empty"):
        prepare_price_data(pl.DataFrame())

    # Missing required columns
    invalid_data = pl.DataFrame({"wrong_col": [1, 2, 3]})
    with pytest.raises(DataFormatError, match="Missing required columns"):
        prepare_price_data(invalid_data)

    # Invalid date format
    invalid_dates = pl.DataFrame(
        {"date": ["invalid_date"], "ticker": ["AAPL"], "close": [100.0]}
    )
    with pytest.raises(DataTypeError):
        prepare_price_data(invalid_dates)


def test_invalid_weight_data():
    """Test error handling for invalid weight data."""
    # Weights summing to more than 1.0
    invalid_weights = pl.DataFrame(
        {"insight_date": ["2024-01-01"], "AAPL": [0.6], "GOOGL": [0.5]}
    )
    with pytest.raises(ValueError, match="Portfolio weights sum to more than 1.0"):
        prepare_weight_data(invalid_weights)

    # Null values should trigger warning but not error
    null_weights = pl.DataFrame(
        {
            "insight_date": ["2024-01-01", "2024-01-02"],
            "AAPL": [None, 0.5],  # Use None instead of pl.Null
            "GOOGL": [0.5, 0.5],
        }
    )
    with pytest.warns(
        UserWarning,
        match="Weight columns contain null values. These will be filled with zeros.",
    ):
        result = prepare_weight_data(null_weights)

    # Verify nulls were filled with zeros
    assert result["AAPL"][0] == 0.0
    assert result["GOOGL"][0] == 0.5

    # Verify total weights are valid after null filling
    total_weights = result["AAPL"] + result["GOOGL"]
    assert (total_weights <= 1.0).all()
    assert total_weights[0] == 0.5  # First row should sum to 0.5
    assert total_weights[1] == 1.0  # Second row should sum to 1.0


def test_calculate_sharpe_ratio(sample_returns_data):
    """Test Sharpe ratio calculation."""
    sharpe = calculate_sharpe_ratio(sample_returns_data, risk_free_rate=0.02)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

    # Test with zero volatility
    zero_vol_data = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "returns": [0.01, 0.01],  # Constant returns
        }
    )
    sharpe = calculate_sharpe_ratio(zero_vol_data)
    assert sharpe == 0.0


def test_calculate_metrics(sample_returns_data):
    """Test portfolio metrics calculation."""
    metrics = calculate_metrics(sample_returns_data, risk_free_rate=0.02)

    # Check all expected metrics are present
    expected_metrics = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    }
    assert set(metrics.keys()) == expected_metrics

    # Check metric values
    assert isinstance(metrics["total_return"], float)
    assert isinstance(metrics["sharpe_ratio"], float)
    assert metrics["max_drawdown"] <= 0.0
    assert metrics["annualized_volatility"] >= 0.0


def test_edge_cases():
    """Test edge cases for metrics calculation."""
    # Single data point
    single_point = pl.DataFrame(
        {"date": ["2024-01-01"], "returns": [0.01], "cumulative_returns": [0.01]}
    )
    metrics = calculate_metrics(single_point)
    assert np.isfinite(metrics["total_return"])

    # No drawdown case
    no_drawdown = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "returns": [0.01, 0.01],
            "cumulative_returns": [0.01, 0.02],
        }
    )
    metrics = calculate_metrics(no_drawdown)
    assert metrics["max_drawdown"] == 0.0
    assert metrics["calmar_ratio"] == np.inf


if __name__ == "__main__":
    pytest.main([__file__])
