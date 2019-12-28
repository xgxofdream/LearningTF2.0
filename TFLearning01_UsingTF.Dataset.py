# -*- coding: utf-8 -*-
"""
#---------------------tf.data ：数据集的构建与预处理------------------------- 
#很多时候，我们希望使用自己的数据集来训练模型。然而，面对一堆格式不一的原始数据文件，将其预处理并读入程序的过程往往十分繁琐，
#甚至比模型的设计还要耗费精力。比如，为了读入一批图像文件，我们可能需要纠结于 python 的各种图像处理包（比如 pillow ），
#自己设计 Batch 的生成方式，最后还可能在运行的效率上不尽如人意。为此，TensorFlow 提供了 tf.data 这一模块，
#包括了一套灵活的数据集构建 API，能够帮助我们快速、高效地构建数据输入的流水线，尤其适用于数据量巨大的场景。

#----------------------------------------------------------------------
#-----------------Cat & Dog images 来源于Cifar10数据集-----------------
----------train_cats_dir = data_dir + '/train/cat/'-------------------
----------train_dogs_dir = data_dir + '/train/dog/'------------------
----------test_cats_dir = data_dir + '/test/cat/'-------------------
----------test_dogs_dir = data_dir + '/test/dog/'---------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


Website link: https://tf.wiki/zh/basic/tools.html#tf-data

Created on Sat Dec 28 19:16:55 2019

@author: liujie
"""

import tensorflow as tf
import os

#------------------------------------Initialization------------------------------------------
num_epochs = 2
batch_size = 32
learning_rate = 0.01
data_dir = 'F:/Google Drive/MedicalCNN/Testcode/data'
train_cats_dir = data_dir + '/train/cat/'
train_dogs_dir = data_dir + '/train/dog/'
test_cats_dir = data_dir + '/test/cat/'
test_dogs_dir = data_dir + '/test/dog/'

#------------------------------------Data Preparation------------------------------------------
def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)            # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [32, 32]) / 255.0
    return image_resized, label

if __name__ == '__main__':
    
    # 构建训练数据集
    train_cat_filenames = tf.constant([train_cats_dir + filename for filename in os.listdir(train_cats_dir)])
    train_dog_filenames = tf.constant([train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    train_labels = tf.concat([
        tf.zeros(train_cat_filenames.shape, dtype=tf.int32), 
        tf.ones(train_dog_filenames.shape, dtype=tf.int32)], 
        axis=-1)
    # 配置数据集的使用方案
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=23000)    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#------------------------------------Build Model------------------------------------------
    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

#------------------------------------Train Model------------------------------------------
    # Fit模型
    model.fit(train_dataset, epochs=num_epochs)

#------------------------------------Grad-CAM and Visualisation------------------------------------------
#
    #


#------------------------------------Test and Prediction------------------------------------------    
    # 构建测试数据集
    test_cat_filenames = tf.constant([test_cats_dir + filename for filename in os.listdir(test_cats_dir)])
    test_dog_filenames = tf.constant([test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)])
    test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=-1)
    test_labels = tf.concat([
        tf.zeros(test_cat_filenames.shape, dtype=tf.int32), 
        tf.ones(test_dog_filenames.shape, dtype=tf.int32)], 
        axis=-1)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(_decode_and_resize)
    test_dataset = test_dataset.batch(batch_size)

    print(model.metrics_names)
    print(model.evaluate(test_dataset))