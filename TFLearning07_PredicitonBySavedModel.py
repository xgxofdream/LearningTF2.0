# -*- coding: utf-8 -*-
"""


#---------------------------------使用 SavedModel 完整导出模型---------------------------
#在部署模型时，我们的第一步往往是将训练好的整个模型完整导出为一系列标准格式的文件，然后即可在不同的平台上部署模型文件。
#这时，TensorFlow 为我们提供了 SavedModel 这一格式。与前面介绍的 Checkpoint 不同，
#SavedModel 包含了一个 TensorFlow 程序的完整信息： 不仅包含参数的权值，还包含计算的流程（即计算图） 。
#当模型导出为 SavedModel 文件时，无需建立模型的源代码即可再次运行模型，
#这使得 SavedModel 尤其适用于模型的分享和部署。后文的 TensorFlow Serving（服务器端部署模型）、TensorFlow Lite（移动端部署模型）以及 TensorFlow.js 都会用到这一格式。



Created on Sat Dec 28 20:44:51 2019

@author: liujie
"""
import numpy as np
import tensorflow as tf



#-----------------------------定义一些模型超参数------------------------------
num_epochs = 1
batch_size = 50
learning_rate = 0.01


#-----------------------------数据获取及预处理------------------------------
#tf.keras.datasets
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

#-----------------------------模型的构建------------------------------
#Keras Sequential API 模式建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

#---------------------------载入数据和Fit模型-------------------------------------------
data_loader = MNISTLoader()
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

#---------------------------保存模型-------------------------------------------
tf.saved_model.save(model, "saved\\1\\")

#---------------------------载入模型-------------------------------------------
model = tf.saved_model.load("saved\\1\\")

#---------------------------预测-------------------------------------------
data_loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())