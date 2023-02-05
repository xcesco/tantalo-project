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


## Configurazione
```shell
docker run \
      --name influxdb \
      -p 8086:8086 \
      -v myInfluxVolume:/var/lib/influxdb2 \
      influxdb:2.6.1
```

```shell
docker run \
      --name influxdb \
      -p 8086:8086 \
      influxdb:2.6.1
```

Per eseguire il setup di 

https://medium.com/geekculture/deploying-influxdb-2-0-using-docker-6334ced65b6c


```shell
docker run --name influxdb -d \
  -p 8086:8086 \
  --volume `pwd`/influxdb2:/var/lib/influxdb2 \
  --volume `pwd`/config.yml:/etc/influxdb2/config.yml \
  influxdb:2.6.1
```

Per creare il setup

```shell
docker exec influxdb influx setup \
  --bucket "bucket01" \
  --org "org01" \
  --password "admin1234" \
  --username "admin" \
  --token "2abdb6a7-ea6e-4044-a051-5f1a99fed42c" \
  --force
```

Per visualizzare i token

```shell
docker exec influxdb influx auth list
```

```shell
docker exec influxdb influx setup \
  --bucket "bucket02" \
  --force
```

query

```shell
from(bucket: "bucket01")
|> range(start: v.timeRangeStart, stop: v.timeRangeStop) 
```

```shell
import "timezone"

option location = timezone.location(name: "Europe/Rome")

from(bucket: "misure")
|> range(start: v.timeRangeStart, stop: v.timeRangeStop)
|> filter(fn: (r) => r["_measurement"] == "peso")
|> filter(fn: (r) => r["_field"] == "indice_grasso_corporeo" or r["_field"] == "muscle")
|> filter(fn: (r) => r["_value"]>0.0)
|> filter(fn: (r) => r["persona"] == "cesco")
|> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
|> truncateTimeColumn(unit: 1d)
|> yield(name: "mean")
```

Per creare dai `tick` un bucket 