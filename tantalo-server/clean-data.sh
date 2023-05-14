#!/bin/bash

# Influxdb
rm -rf ${PWD}/influxdb/data
mkdir ${PWD}/influxdb/data
touch ${PWD}/influxdb/data/volume.txt
rm -rf ${PWD}/twinconfig/h2db
mkdir ${PWD}/twinconfig/h2db/
touch ${PWD}/twinconfig/h2db/volume.txt