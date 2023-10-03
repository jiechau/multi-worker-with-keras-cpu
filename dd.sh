#!/bin/bash

# ./dd.sh -1
# ./dd.sh 0
# ./dd.sh 1

git pull
export TF_ENABLE_ONEDNN_OPTS=0

# "172.17.2.15:8090" # ap07
# "172.17.2.11:8090" # ap06
# "172.17.2.10:8090" # ap05
# "172.17.2.14:8090" # ap04
# "172.17.2.12:8090" # ap03
# "172.17.2.16:8090" # ap02
# "172.17.2.13:8090" # ap01

if [ "$1" -eq -1 ]; then
    export TF_CONFIG="{
        \"cluster\": {
            \"chief\": [\"172.17.2.15:8090\"],
            \"worker\": [\"172.17.2.11:8090\", \"172.17.2.10:8090\"]
        },
        \"task\": {
            \"type\": \"chief\",
            \"index\": 0
        }
    }"
else
    export TF_CONFIG="{
        \"cluster\": {
            \"chief\": [\"172.17.2.15:8090\"],
            \"worker\": [\"172.17.2.11:8090\", \"172.17.2.10:8090\"]
        },
        \"task\": {
            \"type\": \"worker\",
            \"index\": $1
        }
    }"
fi

python dmn.py
