# -*- coding: utf-8 -*-
"""

#-----------------Keras Pipeline---------------------------
#以上示例均使用了 Keras 的 Subclassing API 建立模型，即对 tf.keras.Model 类进行扩展以定义自己的新模型，
#同时手工编写了训练和评估模型的流程。
#这种方式灵活度高，且与其他流行的深度学习框架（如 PyTorch、Chainer）共通，是本手册所推荐的方法。
#不过在很多时候，我们只需要建立一个结构相对简单和典型的神经网络（比如上文中的 MLP 和 CNN），并使用常规的手段进行训练。
#这时，Keras 也给我们提供了另一套更为简单高效的内置方法来建立、训练和评估模型。

#-------------------Keras Functional API 模式建立模型------------------------------ 
#不过Keras Sequential API这种层叠结构并不能表示任意的神经网络结构。
#为此，Keras 提供了 Functional API，帮助我们建立更为复杂的模型，例如多输入 / 输出或存在参数共享的模型。
#其使用方法是将层作为可调用的对象并返回张量（这点与之前章节的使用方法一致），
#并将输入向量和输出向量提供给 tf.keras.Model 的 inputs 和 outputs 参数。

#https://tf.wiki/zh/basic/models.html

Created on Sat Dec 28 19:49:26 2019

@author: liujie
"""
import numpy as np
import tensorflow as tf


#-----------------------------定义一些模型超参数------------------------------
# 这些参数都会影响模型性能
num_epochs = 5
batch_size = 50
learning_rate = 0.01

#-----------------------------数据获取及预处理------------------------------
#tf.keras.datasets
#数据准备方式影响计算速度
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
#Keras Functional API 模式建立模型
# 模型的设计影响模型的性能
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model.compile封装义了梯度下降的算法（Gradient Descent）：
# 随机赋值模型的参数w，开始计算loss(𝐿)
# 根据所得loss(𝐿)，计算导数grads=𝜕𝐿/𝜕𝑤
# 优化器依据所得导数，和学习率learning_rate(𝜂)，得出下一步的模型的参数：𝑤 ← 𝑤 − 𝜂𝜕𝐿/𝜕𝑤
model.compile(
    # 优化器optimizer的作用：根据learning_rate(𝜂)和loss(𝐿)计算模型参数。模型的初始参数w0是随机给定的。
    # 即，w = optimizer (𝐿,𝜂,w0)
    # 优化器optimizer在工作过程中，它还会随着迭代epoch而调整学习率learning_rate(𝜂)。思路是：
    # 初始阶段始选择大的𝜂=𝜂0(即初始𝜂)，随着迭代的深入，由于我们在接近局部或者全局最优loss(𝐿)，我们调小𝜂。
    # 因此，选择什么样的优化器决定了𝜂调整的好坏。有的优化器方法还涉及到动量Momentum的设定。
    # 优化器函数optimizer和损失函数loss一起影响模型性能，loss函数是优化器函数optimizer的上游。
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

#-----------------------------数据读取和Fit模型------------------------------
data_loader = MNISTLoader()
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

#----------------------------------------------模型的评估------------------------------
print(model.evaluate(data_loader.test_data, data_loader.test_label))
