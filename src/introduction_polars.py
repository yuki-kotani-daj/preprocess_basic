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