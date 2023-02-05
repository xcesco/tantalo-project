#!/bin/sh

set -e
# crediamo organization
influx org create -n trading

# ed i bucket
influx bucket create -n ticks -o trading -r 72h
influx bucket create -n historical_tick -o trading -r 0