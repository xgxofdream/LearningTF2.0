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
num_batches = 1000
batch_size = 50
learning_rate = 0.01

#-----------------------------数据获取及预处理------------------------------
dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)).shuffle(1024).batch(32)

#-----------------------------模型的构建------------------------------
#使用 Keras 中预定义的经典卷积神经网络结构MobileNetV2

model = tf.keras.applications.MobileNetV2(weights=None, classes=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for images, labels in dataset:
    with tf.GradientTape() as tape:
        labels_pred = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
        loss = tf.reduce_mean(loss)
        print("loss %f" % loss.numpy())
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
