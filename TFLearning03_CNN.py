# -*- coding: utf-8 -*-
"""
Website link: https://tf.wiki/zh/basic/models.html

Created on Sat Dec 28 19:16:55 2019

@author: liujie
"""
import tensorflow as tf
import numpy as np

#-----------------------------定义一些模型超参数------------------------------
# 这些参数都会影响模型性能
num_epochs = 5
batch_size = 50 #批次大小影响模型性能，小一点会更好
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
#： tf.keras.Model 和 tf.keras.layers
# 模型的设计影响模型的性能
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


#-----------------------------实例化模型和数据读取类------------------------------
#并实例化一个 tf.keras.optimizer 的优化器（这里使用常用的 Adam 优化器）
model = CNN()
data_loader = MNISTLoader()

# 优化器optimizer的作用：根据learning_rate(𝜂)和loss(𝐿)计算模型参数。模型的初始参数w0是随机给定的。
# 即，w = optimizer (𝐿,𝜂,w0)
# 优化器optimizer在工作过程中，它还会随着迭代epoch而调整学习率learning_rate(𝜂)。思路是：
# 初始阶段始选择大的𝜂=𝜂0(即初始𝜂)，随着迭代的深入，由于我们在接近局部或者全局最优loss(𝐿)，我们调小𝜂。
# 因此，选择什么样的优化器决定了𝜂调整的好坏。有的优化器方法还涉及到动量Momentum的设定。
# 优化器函数optimizer和损失函数loss一起影响模型性能，loss函数是优化器函数optimizer的上游。
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#-----------------------------迭代训练模型------------------------------
#然后迭代进行以下步骤：

#从 DataLoader 中随机取一批训练数据；

#将这批数据送入模型，计算出模型的预测值；

#将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；

#计算损失函数关于模型变量的导数；

#将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数

num_batches = int(data_loader.num_train_data // batch_size * num_epochs)


for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    # 1.开始执行梯度下降（Gradient Descent），过程中计算loss
    with tf.GradientTape() as tape:
        # 2.随机赋值模型的参数w，计算loss
        y_pred = model(X)
        # 2.1 采用交叉熵crossentropy方法计算loss。合适的loss计算方式决定模型性能。
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    # 3.根据所得loss(𝐿)，计算导数𝜕𝐿/𝜕𝑤
    grads = tape.gradient(loss, model.variables)
    # 4.依据#3所得导数，和学习率learning_rate(𝜂)，得出下一步的模型的参数：𝑤 ← 𝑤 − 𝜂𝜕𝐿/𝜕𝑤
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
 
    
#----------------------------------------------模型的评估------------------------------
#tf.keras.metrics
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())
