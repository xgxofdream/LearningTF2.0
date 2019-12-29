# -*- coding: utf-8 -*-
"""
#--------------------使用 Keras 中预定义的经典卷积神经网络结构-------------------------
#tf.keras.applications 中有一些预定义好的经典卷积神经网络结构，如 VGG16 、 VGG19 、 ResNet 、 MobileNet 等。
#我们可以直接调用这些经典的卷积神经网络结构（甚至载入预训练的参数），而无需手动定义网络结构。

#https://tf.wiki/zh/basic/models.html

Created on Sat Dec 28 19:49:26 2019

@author: liujie
"""
import tensorflow as tf
import tensorflow_datasets as tfds

#-----------------------------定义一些模型超参数------------------------------
# 这些参数都会影响模型性能
num_epochs = 1000
batch_size = 50 #批次大小影响模型性能，小一点会更好
learning_rate = 0.01

#-----------------------------数据获取及预处理------------------------------
#数据准备方式影响计算速度
dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)).shuffle(1024).batch(32)

#-----------------------------模型的构建------------------------------
#使用 Keras 中预定义的经典卷积神经网络结构MobileNetV2
# 模型的设计影响模型的性能
model = tf.keras.applications.MobileNetV2(weights=None, classes=5)

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

for images, labels in dataset:
    # 1.开始执行梯度下降（Gradient Descent），过程中计算loss
    with tf.GradientTape() as tape:
        # 2.随机赋值模型的参数w，计算loss
        labels_pred = model(images)
        # 2.1 采用交叉熵crossentropy方法计算loss。合适的loss计算方式决定模型性能。
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
        loss = tf.reduce_mean(loss)
        print("loss %f" % loss.numpy())
    # 3.根据所得loss(𝐿)，计算导数grads=𝜕𝐿/𝜕𝑤
    grads = tape.gradient(loss, model.trainable_variables)
    # 4.依据#3所得导数，和学习率learning_rate(𝜂)，得出下一步的模型的参数：𝑤 ← 𝑤 − 𝜂𝜕𝐿/𝜕𝑤
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
