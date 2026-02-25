# polarsの基本機能
import polars as pl

df = pl.DataFrame({
    "col1" : [1, 2, 3],
    "col2" : [10.0, 20.0, 30.0],
    "col3" : ["a", "b", "c"]
})

expr = pl.col("col1").pow(2) + pl.col("col2")
print(df.select(expr))

df2 = df.with_columns(col4 = expr)
print(df2)

## 遅延実行（eager / lazy)
## pl.read_**()でeagerモードで実行。関数を実行するたびにデータを処理を実行。戻り値はDataFrame。
path = "/Users/yuki_kotani/data_science/preprocess_basic/data/reservation.parquet"
df = pl.read_parquet(path)
print(
    df
    .filter(pl.col("reserved_at").dt.year() >= 2016)
    .filter(pl.col("people_num") == 1)
    .select(["reservation_id", "total_price"])
)

## pl.scan_**()でlazyモードで実行。データ処理の手順をLazyFrameに格納し、collect関数を実行したタイミングで一連の処理を実行。
df2 = pl.scan_parquet(path)
query = (
    df2
    .filter(pl.col("reserved_at").dt.year() >= 2016)
    .filter(pl.col("people_num") == 1)
    .select(["reservation_id", "total_price"])
)

print(query.explain(optimized= False))
print(query.explain(optimized=True))
print(query.collect())