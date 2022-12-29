# InfluxDB
Serve a gestire le serie temporali.

```shell
https://en.wikipedia.org/wiki/InfluxDB
https://medium.com/coinograph/storing-and-processing-billions-of-cryptocurrency-market-data-using-influxdb-f9f670b50bbd
```

## Telegraf
Sembra essere un integratore di sistema. Da valutare se usarlo al posto di una app scritta ad hoc
per farlo
```shell
https://www.influxdata.com/blog/getting-started-apache-kafka-influxdb/
https://hub.docker.com/_/influxdb
```

## Client influxdb
```shell
https://github.com/influxdata/influxdb-client-java
```

from(bucket: "btc")
|> range(start: v.timeRangeStart, stop:v.timeRangeStop)

