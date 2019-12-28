# -*- coding: utf-8 -*-
"""

#-----------------Keras Pipeline---------------------------
#以上示例均使用了 Keras 的 Subclassing API 建立模型，即对 tf.keras.Model 类进行扩展以定义自己的新模型，
#同时手工编写了训练和评估模型的流程。
#这种方式灵活度高，且与其他流行的深度学习框架（如 PyTorch、Chainer）共通，是本手册所推荐的方法。
#不过在很多时候，我们只需要建立一个结构相对简单和典型的神经网络（比如上文中的 MLP 和 CNN），并使用常规的手段进行训练。
#这时，Keras 也给我们提供了另一套更为简单高效的内置方法来建立、训练和评估模型。

#-------------------Keras Sequential/Functional API 模式建立模型------------------------------ 
#最典型和常用的神经网络结构是将一堆层按特定顺序叠加起来，那么，我们是不是只需要提供一个层的列表，就能由 Keras 将它们自动首尾相连，
#形成模型呢？Keras 的 Sequential API 正是如此。
#通过向 tf.keras.models.Sequential() 提供一个层的列表，就能快速地建立一个 tf.keras.Model 模型并返回

#https://tf.wiki/zh/basic/models.html

Created on Sat Dec 28 19:49:26 2019

@author: liujie
"""
import numpy as np
import tensorflow as tf


#-----------------------------定义一些模型超参数------------------------------
num_epochs = 5
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
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

#-----------------------------数据读取和Fit模型------------------------------
data_loader = MNISTLoader()
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

#----------------------------------------------模型的评估------------------------------
print(model.evaluate(data_loader.test_data, data_loader.test_label))