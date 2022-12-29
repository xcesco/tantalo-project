# Grafana

Serve a visualizzare i dati in modo grafico. Utile per visualizzare le candlestick recuperate dai vari exchange.

```
https://grafana.com/grafana/
https://logz.io/blog/grafana-vs-kibana/
https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/
https://logz.io/blog/prometheus-vs-graphite/
https://hub.docker.com/r/grafana/grafana-oss
```

Per scaricare grafana
```shell
docker pull grafana/grafana-oss:9.2.3
```

Per eseguire grafana

```shell
docker run -d
-v "${PWD}/grafana/data:/data"
-e "GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource,marcusolsson-csv-datasource"
--name=grafana -p 3000:3000 grafana/grafana-oss:9.2.3
--volume "<your volume mapping here>" grafana/grafana-enterprise:8.2.0

```

npm install --global http-server
http-server grafana/data/

## Come configurare Kafka, Prometeus e Grafana
Serve al monitoraggio di Kafka

https://levelup.gitconnected.com/kafka-primer-for-docker-how-to-setup-kafka-start-messaging-and-monitor-broker-metrics-in-docker-b4e018e205d1
https://www.metricfire.com/blog/kafka-monitoring/

https://github.com/rohsin47/kafka-docker-kubernetes/tree/main/docker
