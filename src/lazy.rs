use std::f64::consts::E;

use polars::lazy::dsl::{all, col, lit, when};
use polars::prelude::*;

fn join_dataframes(
    df1: &DataFrame,
    df2: &DataFrame,
    join_col: &str,
    suffix1: &str,
    suffix2: &str,
) -> PolarsResult<LazyFrame> {
    // Convert DataFrames to LazyFrames
    let df1_lazy = df1.clone().lazy();
    let df2_lazy = df2.clone().lazy();

    // Parse dates to a consistent format
    let options = StrptimeOptions {
        format: Some("%Y-%m-%d".into()),
        strict: false,
        exact: true,
        cache: true,
    };

    let parse_date = col(join_col)
        .str()
        .strptime(DataType::Date, options, lit("raise"));

    let df1_lazy = df1_lazy.with_column(parse_date.clone().alias(join_col));
    let df2_lazy = df2_lazy.with_column(parse_date.alias(join_col));

    // Get column names excluding join column
    let df1_cols: Vec<Expr> = df1
        .get_column_names()
        .iter()
        .filter(|name| name.as_str() != join_col)
        .map(|name| col(name.as_str()).alias(&format!("{}_{}", name, suffix1)))
        .collect();

    let df2_cols: Vec<Expr> = df2
        .get_column_names()
        .iter()
        .filter(|name| name.as_str() != join_col)
        .map(|name| col(name.as_str()).alias(&format!("{}_{}", name, suffix2)))
        .collect();

    // Add join column to selections
    let df1_renamed = df1_lazy.select(
        vec![col(join_col)]
            .into_iter()
            .chain(df1_cols)
            .collect::<Vec<Expr>>(),
    );
    let df2_renamed = df2_lazy.select(
        vec![col(join_col)]
            .into_iter()
            .chain(df2_cols)
            .collect::<Vec<Expr>>(),
    );

    // Join the DataFrames
    let joined = df1_renamed.join(
        df2_renamed,
        [col(join_col)],
        [col(join_col)],
        JoinArgs::new(JoinType::Left),
    );

    Ok(joined)
}

pub fn lazy_backtest(prices_df: &DataFrame, weights_df: &DataFrame) -> PolarsResult<LazyFrame> {
    let joined = join_dataframes(prices_df, weights_df, "date", "price", "weight")?;

    let joined = joined
        .with_column(
            (col("AAPL_price").log(E) - col("AAPL_price").shift(lit(1)).log(E))
                .alias("AAPL_log_return"),
        )
        .with_column(lit(0.0f64).alias("AAPL_pos"))
        .with_column(
            when(col("AAPL_weight").is_null())
                .then(col("AAPL_pos") * col("AAPL_log_return").exp())
                .otherwise(col("AAPL_price") * col("AAPL_weight"))
                .alias("AAPL_pos"),
        );

    Ok(joined)
}
