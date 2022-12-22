https://grafana.com/grafana/

https://logz.io/blog/grafana-vs-kibana/

https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/

https://logz.io/blog/prometheus-vs-graphite/


https://hub.docker.com/r/grafana/grafana-oss
docker pull grafana/grafana-oss:9.2.3

docker run -d \
    -v "${PWD}/grafana/data:/data" \
    -e "GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource,marcusolsson-csv-datasource" \
    --name=grafana -p 3000:3000 grafana/grafana-oss:9.2.3



--volume "<your volume mapping here>" grafana/grafana-enterprise:8.2.0


npm install --global http-server
http-server grafana/data/
