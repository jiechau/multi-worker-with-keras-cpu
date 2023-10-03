# Multi Worker With Keras cpu



## run

with an args
- run 
- command

```
./dd.sh 0
./dd.sh 1
```

## memo

```
export TF_CONFIG='{
    "cluster": {
        "chief": ["172.17.2.13:8090"],
        "ps": ["172.17.2.16:8090"],
        "evaluator": ["172.17.2.12:8090"],
        "worker": ["172.17.2.14:8090", "172.17.2.10:8090"]
    },
    "task": {
        "type": "worker",
        "index": 0
    }
}'
```

## ref

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb#scrollTo=_ESVtyQ9_xjx





