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

def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


communication_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.RING) # support CPU
#communication_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
#strategy = tf.distribute.MultiWorkerMirroredStrategy()
# would stop here until every cluster member is ready

#cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
#strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver)


# ndarray, 60000 train, and 10000 test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# dataset
#num_workers = 2
per_worker_batch_size = 64
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(global_batch_size)


# only model build and compile in scope()
with strategy.scope():

    # (1) build
    model = build_model()
    # (2) load
    #model = keras.models.load_model('/tmp/my_model_mn') # all workers should use chief's version

    # compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Convert to graph to enable optimizations
    #multi_worker_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).as_graph_def().as_graph_element() 

    # experimental_distribute_dataset
    dist_dataset = strategy.experimental_distribute_dataset(multi_worker_dataset) 

## BackupAndRestore
## one worker down, all stuck.
## re-start met errors
## doesn't work
#callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/my_model_ckpt')]
#
## ModelCheckpoint
## only chief save model
callbacks = [tf.keras.callbacks.ModelCheckpoint('/tmp/my_model_mn', save_freq='epoch')]

#model.fit(x_train, y_train, epochs=2, batch_size=64) # default batch_size=32
#model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=int(60000/global_batch_size))
model.fit(multi_worker_dataset, epochs=5, steps_per_epoch=int(60000/global_batch_size), callbacks=callbacks)
#model.fit(dist_dataset, epochs=10, steps_per_epoch=int(60000/global_batch_size), callbacks=callbacks)

# evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print('global_batch_size', global_batch_size)

# this way, every worker save model
# but not a good practice
model.save('/tmp/my_model_mn')
