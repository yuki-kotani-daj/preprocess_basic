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

## 特定のデータ型の列を抽出する
### pandasの場合
df = pd.read_parquet(path = path)
df = df.select_dtypes(include="number")
print(df)

### polarsの場合
df2 = pl.scan_parquet(path)
query = (
    df2.select(pl.col(pl.Int64))
)
print(query.collect())

### 複数のデータ型を抽出する場合(例：数値とテキスト)
#### 引数にリストを渡す。
df = pd.read_parquet(path = path)
df = df.select_dtypes(["number","str"])
print(df)

df2 = pl.scan_parquet(path)
query = (
    df2.select(pl.col([pl.Int64,pl.String]))
)
print(query.collect())

## 条件を用いた行の抽出
### pandasを用いた場合
df = pd.read_parquet(path = path)
df = df.query("2 <= people_num <= 4")
print(df)

### polarsを用いた場合(&演算子を利用した場合。andは利用できないので注意)
df2 = pl.scan_parquet(path)
query = (
    df2
    .filter((pl.col("people_num") >= 2) & (pl.col("people_num") <= 4))
    .select(pl.all()) ### このpl.select(pl.all())は省略可能。
)
print(query.collect())

### polarsを用いた場合（.is_between()を利用した場合）
df2 = pl.scan_parquet(path)
query = (
    df2
    .filter(pl.col("people_num").is_between(2, 4))
)
print(query.collect())

## ランダムサンプリング（元データの構成比を考慮しない場合）
### pandasを用いる場合(実数を指定してサンプリング)
df = pd.read_parquet(path = path)
df = df.sample(20000)
print(df)
### pandasを用いたサンプリング（割合とランダム値を指定した場合)
df = pd.read_parquet(path = path)
df = df.sample(frac = 0.01,random_state= 42)
print(df)

### polarsを用いる場合(サンプリング割合とランダム値を指定した場合)
df2 = pl.scan_parquet(path)
query = (
    df2
    .select(pl.all()
            .sample(fraction= 0.01, seed= 42))
)
print(query.collect())