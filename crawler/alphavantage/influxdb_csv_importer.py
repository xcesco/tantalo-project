## Inspirato da
# https://www.influxdata.com/blog/how-to-write-points-from-csv-to-influxdb-v2-and-influxdb-cloud/
from influxdb_client import InfluxDBClient, WriteOptions
import pandas as pd
from influxdb_client.client.exceptions import InfluxDBError

token = 'yVaNlK-R0BORgIb35vftzZQfWcRVI02PXxe0GTf9Veyk32bedRYx_ipDu1m2JaF-jce5_3-TDbVPLkySSPmvAg=='

with InfluxDBClient(url="http://localhost:18086", token=token, org="trading") as client:
    for df in pd.read_csv("./ibm.csv", chunksize=1_000):
        df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
        df.set_index(['time'])

        with client.write_api() as write_api:
            try:
                write_api.write(
                    record=df,
                    bucket="btc",
                    data_frame_measurement_name="btc",
                    data_frame_tag_columns=[],
                    data_frame_timestamp_column="time",
                )
            except InfluxDBError as e:
                print(e)
