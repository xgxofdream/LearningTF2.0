# -*- coding: utf-8 -*-
"""
#---------------------------------使用 SavedModel 完整导出模型---------------------------
#Keras 模型均可方便地导出为 SavedModel 格式。
#不过需要注意的是，因为 SavedModel 基于计算图，所以对于使用继承 tf.keras.Model 类建立的 Keras 模型，
#其需要导出到 SavedModel 格式的方法（比如 call ）都需要使用 @tf.function 修饰（ @tf.function 的使用方式见 前文 ）。

Created on Sat Dec 28 20:44:51 2019

@author: liujie
"""
import numpy as np
import tensorflow as tf



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
#： tf.keras.Model 和 tf.keras.layers
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
    
    #使用继承 tf.keras.Model 类建立的 Keras 模型同样可以以相同方法导出，唯须注意 call 方法需要以 @tf.function 修饰，
    #以转化为 SavedModel 支持的计算图，代码如下：
    @tf.function
    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
#-----------------------------定义一些模型超参数------------------------------
num_epochs = 1
batch_size = 50
learning_rate = 0.01

#-----------------------------实例化模型和数据读取类------------------------------
#并实例化一个 tf.keras.optimizer 的优化器（这里使用常用的 Adam 优化器）
model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#-----------------------------迭代------------------------------
#然后迭代进行以下步骤：

#从 DataLoader 中随机取一批训练数据；

#将这批数据送入模型，计算出模型的预测值；

#将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；

#计算损失函数关于模型变量的导数；

#将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数

num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size 
        #模型导入并测试性能的过程也相同，唯须注意模型推断时需要显式调用 call 方法，即使用：
        y_pred = model.call(data_loader.test_data[start_index: end_index])
        
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
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

#---------------------------保存模型-------------------------------------------
tf.saved_model.save(model, "saved\\2\\")

#---------------------------载入模型-------------------------------------------
model = tf.saved_model.load("saved\\2\\")

#---------------------------预测-------------------------------------------
data_loader = MNISTLoader()
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())