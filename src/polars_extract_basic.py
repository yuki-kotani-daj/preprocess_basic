# 抽出
import pandas as pd
import polars as pl

path = "/Users/yuki_kotani/data_science/preprocess_basic/data/reservation.parquet"

## pandasを用いた抽出
### []を用いる場合
df = pd.read_parquet(path = path)
df = df[["reservation_id","hotel_id","customer_id","checkin_date","checkout_date"]]
print(df)

### df.loc[]を用いる場合
df = pd.read_parquet(path= path)
df = df.loc[:,[
    "reservation_id",
    "hotel_id",
    "customer_id",
    "checkin_date",
    "checkout_date"
]]
print(df)

## polarsを利用した抽出
df2 = pl.scan_parquet(path)

query = (
    df2
    .select(pl.col(["reservation_id","hotel_id","customer_id","checkin_date","checkout_date"]))
)

print(query.collect())

## 条件指定による列の抽出
path2 = "/Users/yuki_kotani/data_science/preprocess_basic/data/hotel.parquet"

### pandasを用いる場合
pd_hotel = pd.read_parquet(path = path2)
pd_hotel = pd_hotel.loc[:, lambda df3: df3.columns.str.startswith("tag_")]
print(pd_hotel)

### polarsを用いる場合
#### polarsのpl.col関数で正規表現を用いた条件抽出を行う場合、必ず先頭を表す"^"で始まり、末尾を表す"$"を記載するルールを守る。
df2 = pl.scan_parquet(path2)
query = (
    df2
    .select(pl.col("^tag_.*$"))
)
print(query.collect())

## 欠損のある列の抽出
path3 = "/Users/yuki_kotani/data_science/preprocess_basic/data/customer.parquet"

### pandasを用いる場合
pd_customer = pd.read_parquet(path = path3)
pd_customer = pd_customer.loc[:, lambda df3 : df3.isnull().any()]
print(pd_customer)

### polarsを用いる場合
#### lazy_frameではfor文が使えない点に注意
df2 = pl.read_parquet(path3)
cols = [col.name for col in df2.select(pl.all().is_null()) if col.any()]
print(df2.select(cols))