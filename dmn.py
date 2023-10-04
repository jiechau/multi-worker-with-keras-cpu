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
per_worker_batch_size = 64
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(global_batch_size)


# only model build and compile in scope()
with strategy.scope():

    # (1) build
    #model = build_model()
    # (2) load
    model = keras.models.load_model('/tmp/my_model_mn') # all workers should use chief's version

    # compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


# checkpoint_manager
#checkpoint_dir = '/tmp/ckpt'
#checkpoint = tf.train.Checkpoint(model=model)
#latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
#checkpoint.restore(latest_checkpoint)

#checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)

dist_dataset = strategy.experimental_distribute_dataset(multi_worker_dataset) 


# BackupAndRestore
## one worker down, all stuck.
## re-start met errors
## doesn't work
# Checkpoint saving and restoring
#callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/my_model_ckpt')]
#callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir=checkpoint_dir)]
#callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir=checkpoint_dir, save_freq=100)]

# ModelCheckpoint 
callbacks = [tf.keras.callbacks.ModelCheckpoint('/tmp/my_model_h5/model_{epoch}', save_freq='epoch')]




#model.fit(x_train, y_train, epochs=2, batch_size=64) # default batch_size=32
#model.fit(multi_worker_dataset, epochs=1, steps_per_epoch=int(60000/global_batch_size))
# callbacks
model.fit(dist_dataset, epochs=10, steps_per_epoch=int(60000/global_batch_size), callbacks=callbacks)
#model.fit(multi_worker_dataset, epochs=10, steps_per_epoch=int(60000/global_batch_size), callbacks=callbacks)

#checkpoint_manager.save()

# evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)

# for mninference.py
model.save('/tmp/my_model_mn')
