import tensorflow as tf
from tensorflow import keras

# 加载已训练好的模型
model = keras.models.load_model('/tmp/my_model_mn') 

# random pick one
import numpy as np
from PIL import Image
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
index = np.random.randint(0, len(x_train))
img = x_train[index].reshape(28,28)
img = Image.fromarray(img) 
img.save("/tmp/image_mn.png")

# 载入要预测的图像,并进行预处理
img = keras.preprocessing.image.load_img(
            '/tmp/image_mn.png', target_size=(28, 28), color_mode='grayscale'
            )
img_array = keras.preprocessing.image.img_to_array(img)
img_array = img_array.reshape(1, 28, 28, 1) / 255.0

# 使用模型进行预测
predictions = model.predict(img_array)

# 得到预测结果类别
predicted_index = np.argmax(predictions[0])
predicted_label = str(predicted_index)

print("Predicted digit: ", predicted_label)
#print("predictions: ", predictions)
for row_index, row in enumerate(predictions):
    for col_index, value in enumerate(row):
        print(f"{col_index} - {value}")