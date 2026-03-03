import pandas as pd
import polars as pl

# データ全体の集約と代表的な集約関数
## 基礎集計
path = "/Users/yuki_kotani/data_science/preprocess_basic/data/reservation.parquet"

### pandasの場合
pd.set_option("display.float_format","{:.3f}".format)

reservation = pd.read_parquet(path = path)
reservation = (
    reservation.query("status != 'canceled'")
                .agg(
                    reservation_cnt = ("reservation_id", "count"),
                    sales = ("total_price", "sum"),
                    mean_sales = ("total_price", "mean"),
                    min_sales = ("total_price","min"),
                    max_sales = ("total_price", "max"),
                    var_sales = ("total_price", "var"),
                    std_sales = ("total_price", "std")
                )
            )
print(reservation)

### polarsの場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .filter(pl.col("status") != "canceld")
    .select([
        pl.col("reservation_id").len().alias("reservation_cnt"),
        pl.col("total_price").sum().alias("num_sales"),
        pl.col("total_price").mean().alias("unit_price_mean"),
        pl.col("total_price").max().alias("max_unit_price"),
        pl.col("total_price").min().alias("min_unit_price"),
        pl.col("total_price").var().alias("var_unit_rpice"),
        pl.col("total_price").std().alias("std_unit_price")])
)
print(query.collect())