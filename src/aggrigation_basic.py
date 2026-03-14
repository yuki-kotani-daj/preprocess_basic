import pandas as pd
import numpy as np
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

## ホテル予約データについて、キャンセルを除外して顧客のユニーク数を調べたい
### pandasの場合
reservation = pd.read_parquet(path = path)
reservation = (
    reservation
    .query("status != 'canceled'")
    .customer_id.nunique()
)
print("unique_customers:",reservation)

### polarsの場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .filter(pl.col("status") != "canceled")
    .select(pl.n_unique("customer_id").alias("num_customer"))
)
reservation2 = query.collect()
print(reservation2)

### polars_nullを除いてユニークカウントしたい場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .filter(pl.col("status") != "canceled")
    .select(pl.col("customer_id").drop_nulls().n_unique().alias("num_customer"))
)
reservation2 = query.collect()
print("nullを除外したユニーク顧客数：\n",reservation2)

## ホテル予約データについて、キャンセルを除いて予約単価のパーセンタイル値を算出したい
### pandasの場合
pd.set_option("display.float_format","{:.3f}".format)
reservation = pd.read_parquet(path = path)
reservation = (
    reservation
    .query("status != 'canceled'")
    .agg(
        median_sales = ("total_price","median"),
        p25_sales = ("total_price", lambda s: s.quantile(0.25)),
        p75_sales = ("total_price", lambda s: s.quantile(0.75))
    )
)
print(reservation)

### polarsの場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .filter(pl.col("status") != "canceled")
    .select([
        pl.col("total_price").quantile(0.25, interpolation = "linear").alias("p25_sales"),
        pl.col("total_price").median().alias("median_sales"),
        pl.col("total_price").quantile(0.75, interpolation= "linear").alias("p75_sales")
    ])
)
reservation2 = query.collect()
print(reservation2)

## ホテル予約データにて、キャンセルを除外して、ホテル毎の予約人数の最頻値を知りたい。
### pandasの場合_最頻値が複数個ある場合に、行を足して全ての候補を表示する
reservation = pd.read_parquet(path = path)
reservation = (
    reservation
    .query("status != 'canceled'")
    .groupby("hotel_id")
    .agg(mode_people_num = ("people_num", lambda s : s.mode()))
    .explode("mode_people_num")
)
print("pandasの場合：\n",reservation)

### pandasの場合_最頻値が複数個ある場合に、一番最初の値だけ表示する
reservation = pd.read_parquet(path = path)
reservation = (
    reservation
    .query("status != 'canceled'")
    .groupby("hotel_id")
    .agg(mode_people_num = ("people_num", lambda s : s.mode().iloc[0]))
)
print("pandasの場合：\n",reservation)

## polarsの場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .filter(pl.col("status") != "canceled")
    .group_by(pl.col("hotel_id"))
    .agg(pl.col("people_num").mode().first().alias("mode_people_num"))
)
reservation2 = query.collect()
print("polarsの場合：\n",reservation2)

## キャンセルを除いて、ホテル毎の売り上げを算出したい
### pandaasの場合
reservation = pd.read_parquet(path = path)
reservation = (
    reservation
    .query("status != 'canceled'")
    .groupby("hotel_id").agg(sales = ("total_price","sum"))
)
print("pandasの場合：\n",reservation)

### polarsの場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .filter(pl.col("status") != "canceled")
    .group_by(pl.col("hotel_id")).agg(pl.sum("total_price"))
)
reservation2 = query.collect()
print("polarsの場合：\n",reservation2)

## キャンセルを除いて、ホテル毎、顧客毎の予約数を計算
### pandasの場合
reservation = pd.read_parquet(path = path)
reservation = (
    reservation
    .query("status != 'canceled'")
    .groupby(["hotel_id","customer_id"]).size()
)
print("pandasの場合：\n",reservation)

### polarsの場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .filter(pl.col("status") != "canceled")
    .group_by(["hotel_id","customer_id"]).agg(pl.len().alias("num_reserve"))
)
reservation2 = query.collect()
print("polarsの場合：\n",reservation2)

## 価格帯毎にホテル数を集計したい
### pandasの場合
reservation = pd.read_parquet(path = path)
reservation = (
    reservation
    .assign(unit_price_range = lambda df:
            (np.floor(df.total_price / 5000) * 5000).astype(int))
    .groupby("unit_price_range").size()
)
print("pandasの場合：\n",reservation)

### polarsの場合
reservation2 = pl.scan_parquet(path)
query = (
    reservation2
    .group_by((pl.col("total_price") / 5000).floor().cast(pl.Int32) * 5000)
    .agg(pl.len())
)
reservation2 = query.collect()
print(reservation2)