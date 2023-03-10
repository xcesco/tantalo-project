# Tantalo Project: Automated Trading System

Progetto relativo alla studio dei sistemi di trading e delle realizzazione di un sistema automatico di trading (si spera).

![schema.drawio.svg](docs%2Fschema.drawio.svg)

In fase 0, il sistema e' sviluppato in ambiente docker-compose. Gli elementi utilizzati:

 - Crawler: recupera i dati dai vari broker
 - Kafka: sistema di streaming per far colloquiare i vari componenti senza effettuare pressioni.
 - InfluxDB: base dati per le serie temporali
 - Grafana: serve per la visualizzazione dei dati
 - Sistema decisionale: TODO

## Studio
Di seguito sono riportati alcuni elementi di studio:

 - [Studio](./STUDIO.md)

## Indice degli argomenti

* [Crawler](crawler/CRAWLER.md)

## Concetti base

* [Concetti base](./CONCETTI_BASE.md)

## Elastic search

* [Elastic search](./elk/ELASTIC.md)
