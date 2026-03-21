import pandas as pd
import numpy as np
import polars as pl

# sample-file
path = "/Users/yuki_kotani/data_science/preprocess_basic/data/reservation.parquet"
path2 = "/Users/yuki_kotani/data_science/preprocess_basic/data/hotel.parquet"

# 結合（基本はSQLで行う。どうしてもpythonでやらないといけないのか？よく検討せよ）
## ビジネスホテルかつ宿泊人数が１名の予約履歴を抽出したい
### pandasの場合
reservation = pd.read_parquet(path = path)
hotel = pd.read_parquet(path = path2)

reservation_check = (
    reservation
    .query("people_num == 1")
    .merge(hotel.query("hotel_type == 'ビジネスホテル'")["hotel_id"],
           how = "inner", on = "hotel_id")
)
print("pandasの場合：\n",reservation_check)

### polarsの場合
reservation2 = pl.scan_parquet(path)
hotel2 = pl.scan_parquet(path2)

query = (
    reservation2
    .filter(pl.col("people_num") == 1)
    .join(
        hotel2.filter(pl.col("hotel_type") == "ビジネスホテル")
        .select(pl.col("hotel_id")),on = "hotel_id", how = "inner")
)

reservation_check2 = query.collect()
print("polarsの場合：\n",reservation_check2)

### polars_semi結合の場合
query2 = (
    reservation2
    .filter(pl.col("people_num") == 1)
    .join(hotel2.filter(pl.col("hotel_type") == "ビジネスホテル")
          ,on = "hotel_id", how = "semi")
)
reservation_check3 = query2.collect()
print("semi結合の場合：\n",reservation_check3)

## 2019年の売上と予約数をホテルマスタへ追加したい
### pandasの場合
reservation = pd.read_parquet(path = path)
hotel = pd.read_parquet(path = path2)

hotel_master = (
    hotel
    .merge(
        reservation
        .query("status != 'canceled' and checkout_date.dt.year == 2019")
        .groupby("hotel_id").total_price.agg(["sum","size"]),
        on = "hotel_id", how = "left"
    )
    .fillna({
        "sum":0,
        "size":0
    })
)
print("pandasの場合：\n",hotel_master)

### polarsの場合
reservation2 = pl.scan_parquet(path)
hotel2 = pl.scan_parquet(path2)

query = (
    hotel2
    .join(
        reservation2
        .filter((pl.col("status") != "canceled") &
                (pl.col("checkout_date").dt.year() == 2019)
                )
                .group_by("hotel_id").agg(
                    sales = pl.col("total_price").sum(),
                    num_reservation = pl.len()
                ),on = "hotel_id", how = "left"
                )
    .with_columns([
        pl.col("sales").fill_null(0),
        pl.col("num_reservation").fill_null(0)
    ])
)

hotel_master2 = query.collect()
print("polarsの場合：\n",hotel_master2)