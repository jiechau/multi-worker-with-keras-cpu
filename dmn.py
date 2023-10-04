import tensorflow as tf
from tensorflow import keras

import json
import os

tf_config = json.loads(os.environ['TF_CONFIG'])
cluster_spec = tf_config['cluster'] 
task_type = tf_config['task']['type']
task_id = tf_config['task']['index']
num_workers = len(tf_config['cluster']['worker'])
print(tf_config)

#print(task_id)
#import sys
#sys.exit(0)


communication_options=tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.RING)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
# would stop here until every cluster member is ready

#cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
#strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver)


# ndarray, 60000 train, and 10000 test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# dataset
per_worker_batch_size = 64
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)
train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)

with strategy.scope():

    # 构建模型
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # 编译和训练模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=2)
#model.fit(x_train, y_train, epochs=10)
model.fit(multi_worker_dataset, epochs=2)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)

model.save('/tmp/my_model_mn')
