from datetime import date
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
        .filter((
            pl.col('status') != 'canceled') &
            (pl.col('checkin_date').dt.year() == 2019)
        )
        .group_by('hotel_id').agg(
            num_reservation = pl.len(),
            sales = pl.col('total_price').sum()
        ),on = 'hotel_id', how = 'left'
    )
    .with_columns(
        num_reservation = pl.col('num_reservation').fill_null(0),
        sales = pl.col('sales').fill_null(0)
    )
)

hotel_master2 = query.collect()
print('polarの場合:\n',hotel_master2)

## 顧客毎、ホテルタイプ毎の予約数を集計し、顧客マスタへ付与したい
### pandasの場合
path3 = '/Users/yuki_kotani/Downloads/awesomebook_v2-main/data/customer.parquet'

reservation = pd.read_parquet(path = path)
hotel = pd.read_parquet(path = path2)
customer = pd.read_parquet(path = path3)

master_customer = (
    customer
    .merge(
        reservation[['customer_id','hotel_id','reservation_id']]
        .merge(
            hotel[['hotel_id','hotel_type']],on = 'hotel_id', how = 'left'
        )
        .assign(
            ryokan = lambda df:
            np.where(df.hotel_type == '旅館',1,0),
            resort_hotel = lambda df:
            np.where(df.hotel_type == 'リゾートホテル',1,0),
            business_hotel = lambda df:
            np.where(df.hotel_type == 'ビジネスホテル',1,0),
            minsyuku = lambda df:
            np.where(df.hotel_type == '民宿',1,0)
        )
        .groupby('customer_id')[['ryokan','resort_hotel','business_hotel','minsyuku']].sum()
        ,on = 'customer_id', how = 'left'
    )
    .fillna({
        'ryokan':0,
        'resort_hotel':0,
        'business_hotel':0,
        'minsyuku':0
    })
)
print('pandasの場合：\n',master_customer)

### polarsの場合
reservation2 = pl.scan_parquet(path)
hotel2 = pl.scan_parquet(path2)
customer2 = pl.scan_parquet(path3)

query = (
    customer2
    .join(
        reservation2.select(['customer_id','hotel_id','reservation_id'])
        .join(
            hotel2.select(['hotel_id','hotel_type'])
        ,on = 'hotel_id', how = 'left'
    ).group_by('customer_id').agg(
        num_ryokan = pl.col('reservation_id')
        .filter(pl.col('hotel_type') == '旅館').len(),
        num_resort_hotel = pl.col('reservation_id')
        .filter(pl.col('hotel_type') == 'リゾートホテル').len(),
        num_business_hotel = pl.col('reservation_id')
        .filter(pl.col('hotel_type') == 'ビジネスホテル').len(),
        num_minsyuku = pl.col('reservation_id')
        .filter(pl.col('hotel_type') == '民宿').len()
    ),on = 'customer_id', how = 'left'
    ).with_columns(
        num_ryokan = pl.col('num_ryokan').fill_null(0),
        num_reserot_hotel = pl.col('num_resort_hotel').fill_null(0),
        num_business_hotel = pl.col('num_business_hotel').fill_null(0),
        num_minsyuku = pl.col('num_minsyuku').fill_null(0)
    )
)
master_customer2 = query.collect()
print('polarsの場合：\n',master_customer2)

## 顧客毎の月別の売り上げを計算したい（売上のない月も出力）
### pandasの場合
reservation = pd.read_parquet(path = path)
customer = pd.read_parquet(path = path3)

monthly_sales_summary = (
    customer[['customer_id']]
    .merge(
        pd.period_range('2019-01-01','2019-12-01',freq = 'M').to_series(name = 'month'),
        how = 'cross'
    )
    .merge(
        reservation
        .query('status != "canceled"')
        .assign(
            month = lambda df:
            df.checkout_date.dt.to_period('M')
        )
        .groupby(['customer_id','month']).total_price.sum(),
        how = 'left',on = ['customer_id','month']
    )
    .fillna({
        'total_price':0
    })
)
print('pandasの場合:\n',monthly_sales_summary.describe())

## polarsの場合
### クエリの連鎖を使って全てLazyuFrameで処理する。
reservation2 = pl.scan_parquet(path)
customer2 = pl.scan_parquet(path3)

start_date = date(2019,1,1)
end_date = date(2019,12,1)

sales_summary = (
    reservation2
    .filter(pl.col('status') != 'canceled')
    .with_columns(month = pl.col('checkout_date')
                  .dt.truncate('1mo').dt.date().alias('month'))
    .group_by(['customer_id','month'])
    .agg(sales = pl.col('total_price').sum())
)

month_df = (
    pl.LazyFrame({
        'month':pl.date_range(start_date,end_date,interval = '1mo',eager = True)
    })
)

customer_with_month = (
    customer2
    .select(pl.col('customer_id'))
    .join(
        month_df,how = 'cross'
    )
)

final_query = (
    customer_with_month
    .join(
        sales_summary,on = ['customer_id','month'],how = 'left'
    )
    .fill_null(0)
)

monthly_sales_summary2 = final_query.collect()
print('polarsの場合:\n',monthly_sales_summary2.describe())
