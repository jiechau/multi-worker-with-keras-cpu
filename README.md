# Multi Worker With Keras cpu



## run

- command

```
git pull; ./dd.sh -1 # for chief
git pull; ./dd.sh 0 # for worker 0
git pull; ./dd.sh 1 # for worker 1
```

## memo

```
export TF_CONFIG='{
    "cluster": {
        "chief": ["172.17.2.13:8090"],
        "worker": ["172.17.2.14:8090", "172.17.2.10:8090"]
    },
    "task": {
        "type": "cluster",
        "index": 0
    }
}'

export TF_CONFIG='{
    "cluster": {
        "chief": ["172.17.2.13:8090"],
        "worker": ["172.17.2.14:8090", "172.17.2.10:8090"]
    },
    "task": {
        "type": "worker",
        "index": 0
    }
}'
```

## ref

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/distributed_training.ipynb#scrollTo=QGX_QAEtFQSv

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb#scrollTo=_ESVtyQ9_xjx





