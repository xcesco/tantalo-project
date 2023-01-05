## Inspirato da
# https://www.influxdata.com/blog/how-to-write-points-from-csv-to-influxdb-v2-and-influxdb-cloud/
from influxdb_client import InfluxDBClient, WriteOptions
import pandas as pd
from influxdb_client.client.exceptions import InfluxDBError

token = 'yVaNlK-R0BORgIb35vftzZQfWcRVI02PXxe0GTf9Veyk32bedRYx_ipDu1m2JaF-jce5_3-TDbVPLkySSPmvAg=='


print(pd.to_datetime("5/28/2022", format="%m/%d/%Y"))

with InfluxDBClient(url="http://localhost:18086", token=token, org="fitness") as client:
    for df in pd.read_csv("./weight.csv", chunksize=1_000):
        df['time'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
        df.set_index(['time'])
        df.drop(columns=['Date', 'WeightP'])
        df.rename(columns={"WeightK": "weight"})

        with client.write_api() as write_api:
            try:
                write_api.write(
                    record=df,
                    bucket="weight",
                    data_frame_measurement_name="weight",
                    data_frame_tag_columns=[],
                    data_frame_timestamp_column="time",
                )
            except InfluxDBError as e:
                print(e)
