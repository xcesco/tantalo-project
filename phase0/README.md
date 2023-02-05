# Fase 0
In questa fase l'obiettivo è quello di impostare i vari componenti.

- Crawler: recupera i dati dai vari exchange e li inserisce nel sistema mediante Kafka
- Kafka: sistema di streming basato sul pattern publish-subscriber
- Grafana: sistema di monitoring

Il sistema costruito eseguendo
```shell
docker-compose up -d
```

Grafana risponde con user `kafka/kafka`:
```shell
http://localhost:13000/
```

Kafka UI risponde senza pwd all'indirizzo:
```shell
http://localhost:18080/
```

Kafka risponde si interfaccia ai publisher ed ai consumer dalla porta 
```shell
http://localhost:19094
```
Alcune informazioni sulla configurazione. Alcuni link:

- [kafka-listeners-explained](https://www.confluent.io/blog/kafka-listeners-explained/)
- [running-kafka-locally-with-docker](https://lankydan.dev/running-kafka-locally-with-docker)
- [kafka-client-cannot-connect-to-broker-on-aws-on-docker-etc](https://www.confluent.io/blog/kafka-client-cannot-connect-to-broker-on-aws-on-docker-etc/).

## Influxdb, database NOSQL per serie storiche
Influxdb sembra essere una buona soluzione per memorizzare i dati delle quotazioni delle varie valute.


```shell
flux
http://influxdb:8086/
trading
btc
```
https://www.influxdata.com/blog/getting-started-apache-kafka-influxdb/
https://stackoverflow.com/questions/50492828/influxdb-for-a-financial-application
https://towardsdatascience.com/the-simplest-way-to-create-an-interactive-candlestick-chart-in-python-ee9c1cde50d8
https://medium.com/coinograph/storing-and-processing-billions-of-cryptocurrency-market-data-using-influxdb-f9f670b50bbd
https://docs.influxdata.com/influxdb/v2.6/write-data/developer-tools/api/
https://docs.influxdata.com/influxdb/v1.7/guides/querying_data/
https://docs.influxdata.com/influxdb/v2.6/reference/config-options/#ui-disabled

https://docs.influxdata.com/influxdb/v2.6/reference/sample-data/#bitcoin-sample-data

Influxdb
18086

https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/time-series/


## Grafana
Accessibile all'url http://localhost:13000/. Le credenziali di default sono `admin/admin1234`,
che codificate in base64 diventano `YWRtaW46YWRtaW4xMjM0`.

In `phase0` l'obiettivo è fare in modo che `grafana` punti a `infludb`. Per fare questo
ho prima configurato a mano, e poi cercato riportare la configurazione ottenuta come
configurazione via script. Le indicazioni le ho trovate [qui](https://community.grafana.com/t/data-source-on-startup/8618/2).

La documentazione sulle API di grafana è disponibile a [questo indirizzo](https://grafana.com/docs/grafana/latest/developers/http_api/data_source/).