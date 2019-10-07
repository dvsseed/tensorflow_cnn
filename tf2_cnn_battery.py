from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('Keras:', tf.keras.__version__)
from tensorflow.keras import backend, utils, callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau  # , EarlyStopping
# from tensorflow.keras.models import Model
# import numpy as np
# from tensorflow.keras import optimizers, losses, backend, datasets, utils, models, regularizers, callbacks
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU
# from kerastuner.tuners import RandomSearch
import pathlib
import random
# import IPython.display as display
import matplotlib.pyplot as plt
import time
from datetime import datetime
# import sys
# import argparse
# import platform
from numba import cuda
from pynvml import *
# # # include sub-procedure
# from parse_arguments import *
from cnn_model import *
# from cnn_model1 import *  # BatchNormalization
# from cnn_model2 import *  # Conv2D+Dropout
# from kerastuner_cnn_model import *  # Keras Tuner::An hyperparameter tuner for Keras
# from parse_arguments import *
from tools import *
from logger import *
# 屏蔽warning信息
import os

# import logging

# 基礎設定
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%y-%m-%d %H:%M',
#                     handlers=[logging.FileHandler('tf2_cnn_battery.log', 'w', 'utf-8'), ])
# 定義 handler 輸出 sys.stderr
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# 設定輸出格式
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# handler 設定輸出格式
# console.setFormatter(formatter)
# 加入 hander 到 root logger
# logging.getLogger('').addHandler(console)
# logging.disable(logging.WARNING)
if not args.show_tf_cpp_log:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log = Logger('tf2_cnn_all.log', level='debug')
# log.logger.debug('debug')
# log.logger.info('info')
# log.logger.warning('警告')
# log.logger.error('報錯')
# log.logger.critical('嚴重')
# Logger('error.log', level='error').logger.error('error')

# Destroys the current TF graph and creates a new one
try:
    nvmlInit()  # 初始化
    print("Driver Version:", nvmlSystemGetDriverVersion())  # 显示驱动信息
    # 查看设备
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("GPU", i, ":", nvmlDeviceGetName(handle))
        # 查看显存、温度、风扇、电源
        info = nvmlDeviceGetMemoryInfo(handle)
        print("Memory Total: ", info.total)
        print("Memory Free: ", info.free)
        print("Memory Used: ", info.used)
        print("Temperature is %d C" % nvmlDeviceGetTemperature(handle, 0))
        print("Fan speed is", nvmlDeviceGetFanSpeed(handle))
        print("Power status", nvmlDeviceGetPowerState(handle))
    nvmlShutdown()  # 最后要关闭管理工具

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(1)
    cuda.close()
except NVMLError as error:
    print(error)

backend.clear_session()

# 可以改变training和testing的状态
backend.set_learning_phase(True)

# print('GPUs: {}'.format(backend.tensorflow_backend._get_available_gpus()))
# tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

# 解決：Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
# 1、將圖片尺寸改小，小到佔用的內存比顯存。
# 2、不使用GPU進行預測，只使用CPU預測，因爲一般CPU內存要大於顯存的。但裝的又是GPU版的TensorFlow，所以需要在預測程序進行更改。程序在前兩行加入下面代碼：
# 指定使用哪块gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# 限制使用比率
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# 按需分配
# config.gpu_options.allow_growth = True  # allocate when needed


'''
Load images
用 tf.data 加载图片
本教程提供一个如何使用 tf.data 加载图片的简单例子。
本例中使用的数据集分布在图片文件夹中，一个文件夹含有一类图片。
'''

'''
配置
'''

# try:
# %tensorflow_version only exists in Colab.
# %tensorflow_version 2.x
# except Exception:
#     pass

AUTOTUNE = tf.data.experimental.AUTOTUNE

# record the from datetime
fromtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# # # Parse input arguments Area


'''
下载并检查数据集
'''
'''
检索图片
在你开始任何训练之前，你将需要一组图片来教会网络你想要训练的新类别。 
你已经创建了一个文件夹，存储了最初使用的拥有创作共用许可的花卉照片
'''
# data_root_orig = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#     fname='flower_photos', untar=True)
# 判斷作業系統
# sysstr = platform.system()
# if sysstr == "Darwin":  # Mac
#     data_root_orig = "/Users/davis/tensorflow_code/lithiumBattery"
# elif sysstr == "Linux":  # Ubuntu
#     data_root_orig = "/home/d1075102/tensorflow_code/lithiumBattery"
# 自動判定目錄
currentDirectory = os.getcwd()  # 印出目前工作目錄
# print(type(currentDirectory))
# imgDirectory = os.path.join(currentDirectory, "/lithiumBattery")  # 組合路徑
# print(imgDirectory)
data_root_orig = currentDirectory + "/lithiumBattery"
# print(data_root_orig)

data_root = pathlib.Path(data_root_orig)
# print(data_root)

# 下载了 218 MB 之后，你现在应该有花卉照片副本
# for dirs in data_root.iterdir():
#     print(dirs)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

# Split into test and train pairs
# train_image_paths = all_image_paths[:700]    # 700
# test_image_paths = all_image_paths[700:]     # 300
# train_image_paths = all_image_paths[:750]    # 750
# test_image_paths = all_image_paths[750:]     # 250
train_image_paths = all_image_paths[:800]      # 800
test_image_paths = all_image_paths[800:]       # 200

# image_count = len(all_image_paths)
train_image_count = len(train_image_paths)
test_image_count = len(test_image_paths)
# print(image_count)
# print(train_image_count)
# print(test_image_count)

# print(all_image_paths[:10])
# print(train_image_paths[:10])
# print(test_image_paths[:10])

# 检查图片
# 现在让我们快速浏览几张图片，这样你知道你在处理什么
# attributions = (data_root / "LICENSE.txt").open(encoding='utf-8').readlines()[4:]
# attributions = [line.split(' CC-BY') for line in attributions]
# attributions = dict(attributions)


# def caption_image(image_path):
#     image_rel = pathlib.Path(image_path).relative_to(data_root)
#     return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


# for n in range(3):
# image_path = random.choice(all_image_paths)
# print(image_path)
# display.display(display.Image(image_path))
# print(caption_image(image_path))
# print()

# 确定每张图片的标签
# 列出可用的标签
label_names = sorted(dirs.name for dirs in data_root.glob('*/') if dirs.is_dir())
# print(label_names)
# ['bottom_NG', 'bottom_OK', 'top_NG', 'top_OK']

# 为每个标签分配索引
label_to_index = dict((name, index) for index, name in enumerate(label_names))
# print(label_to_index)
# {'bottom_NG': 0, 'bottom_OK': 1, 'top_NG': 2, 'top_OK': 3}

# 创建一个列表，包含每个文件的标签索引
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

# print("First 10 labels indices: ", all_image_labels[:10])

# train_image_labels = all_image_labels[:700]    # 700
# test_image_labels = all_image_labels[700:]     # 300
# train_image_labels = all_image_labels[:750]    # 750
# test_image_labels = all_image_labels[750:]     # 250
train_image_labels = all_image_labels[:800]      # 800
test_image_labels = all_image_labels[800:]       # 200
print(len(all_image_labels), 'All examples')
print(len(train_image_labels), 'Train examples')
print(len(test_image_labels), 'Test examples')

'''
加载和格式化图片
'''

# TensorFlow 包含加载和处理图片时你需要的所有工具
# img_path = all_image_paths[0]
# print(img_path)

# 以下是原始数据：
# img_raw = tf.io.read_file(img_path)
# print(repr(img_raw)[:100] + "...")

# 将它解码为图像 tensor（张量）
# img_tensor = tf.image.decode_image(img_raw)
# print(img_tensor.shape)
# print(img_tensor.dtype)

# 根据你的模型调整其大小
# img_final = tf.image.resize(img_tensor, [256, 256])
# img_final = img_final / 255.0
# print(img_final.shape)
# print(img_final.numpy().min())
# print(img_final.numpy().max())


# 秀一張圖
if 1 == 0:
    # img_path = all_image_paths[0]
    # label = all_image_labels[0]
    img_path = train_image_paths[0]
    label = train_image_labels[0]
    # print(img_path)
    # print(label)
    plt.imshow(load_and_preprocess_image_rgb(img_path))
    plt.grid(False)
    # plt.xlabel(caption_image(img_path))
    plt.xlabel(label_names[label] + '=' + str(label))
    plt.title(label_names[label].title())
    plt.show()
    print()

'''
构建一个 tf.data.Dataset
'''
# 一个图片数据集
# 构建 tf.data.Dataset 最简单的方法就是使用 from_tensor_slices 方法
# 将字符串数组切片，得到一个字符串数据集
# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
train_path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)

# shapes（维数） 和 types（类型）描述数据集里每个数据项的内容。在这里是一组标量二进制字符串
# print(path_ds)

# 现在创建一个新的数据集，通过在路径数据集上映射 preprocess_image 来动态加载和格式化图片
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# print(image_ds)

# 秀四張圖
if 1 == 0:
    plt.figure(figsize=(8, 8))
    for n, image in enumerate(train_image_ds.take(4)):
        plt.subplot(2, 2, n + 1)
        # print(image.shape)  # (256, 256, 1)
        # print(np.squeeze(image).shape)
        img = np.squeeze(image)  # squeeze将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了
        plt.imshow(img)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel(caption_image(all_image_paths[n]))
        plt.xlabel(n)
    plt.show()

'''
一个 (图片, 标签) 对数据集
'''
# 使用同样的 from_tensor_slices 方法你可以创建一个标签数据集
# label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_image_labels, tf.int64))
test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_image_labels, tf.int64))
# for label in label_ds.take(10):
#     print(label_names[label.numpy()])


# 由于这些数据集顺序相同，你可以将他们打包在一起得到一个 (图片, 标签) 对数据集
# image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))

# 这个新数据集的 shapes（维数）和 types（类型）也是维数和类型的元组，用来描述每个字段
# print(image_label_ds)

# 注意：当你拥有形似 all_image_labels 和 all_image_paths 的数组，tf.data.dataset.Dataset.zip 的替代方法是将这对数组切片
# ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))

# image_label_ds = ds.map(load_and_preprocess_from_path_label)
# print(image_label_ds)
train_image_label_ds = train_ds.map(load_and_preprocess_from_path_label)
test_image_label_ds = test_ds.map(load_and_preprocess_from_path_label)
# print(train_image_label_ds)
# print(test_image_label_ds)

'''
训练的基本方法
要使用此数据集训练模型，你将会想要数据：
* 被充分打乱。
* 被分割为 batch。
* 永远重复。
* 尽快提供 batch。
使用 tf.data api 可以轻松添加这些功能
'''
# BATCH_SIZE = 256

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
# ds = image_label_ds.shuffle(buffer_size=image_count)
# ds = ds.repeat()
# ds = ds.batch(BATCH_SIZE)
train_ds = train_image_label_ds.shuffle(buffer_size=train_image_count)
train_ds = train_ds.batch(batch_size)
# print(train_ds)
# train_ds = train_ds.repeat()
# train_ds = train_ds.batch(batch_size)
# train_ds = train_ds.batch(750)
test_ds = test_image_label_ds.batch(batch_size)
# test_ds = test_image_label_ds.shuffle(buffer_size=test_image_count)
# test_ds = test_ds.repeat()
# test_ds = test_ds.batch(batch_size)
# test_ds = test_ds.batch(250)
# `prefetch` lets the dataset fetch batches in the background while the model is training.
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
# ds = ds.prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
# print(ds)

'''
这里有一些注意事项：
1. 顺序很重要。
   * 在 .repeat 之后 .shuffle，会在 epoch 之间打乱数据（当有些数据出现两次的时候，其他数据还没有出现过）。
   * 在 .batch 之后 .shuffle，会打乱 batch 的顺序，但是不会在 batch 之间打乱数据。
2. 你在完全打乱中使用和数据集大小一样的 buffer_size（缓冲区大小）。较大的缓冲区大小提供更好的随机化，但使用更多的内存，直到超过数据集大小。
3. 在从随机缓冲区中拉取任何元素前，要先填满它。所以当你的 Dataset（数据集）启动的时候一个大的 buffer_size（缓冲区大小）可能会引起延迟。
4. 在随机缓冲区完全为空之前，被打乱的数据集不会报告数据集的结尾。Dataset（数据集）由  .repeat 重新启动，导致需要再次等待随机缓冲区被填满。
最后一点可以通过使用 tf.data.Dataset.apply 方法和融合过的 tf.data.experimental.shuffle_and_repeat 函数来解决:
'''
# ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE)
# ds = ds.prefetch(buffer_size=AUTOTUNE)
# print(ds)
# train_ds = train_image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=train_image_count))
# train_ds = train_ds.batch(batch_size)
# train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
# print(train_ds)
# test_ds = test_image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=test_image_count))
# test_ds = test_ds.batch(batch_size)
# test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
# print(test_ds)


'''
传递数据集至模型
从 tf.keras.applications 取得 MobileNet v2 副本。
该模型副本会被用于一个简单的迁移学习例子。
设置 MobileNet 的权重为不可训练：
'''
# mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
# mobile_net.trainable = False


# keras_ds = ds.map(change_range)
# train_keras_ds = train_ds.map(change_range)
# test_keras_ds = test_ds.map(change_range)

# 传递一个 batch 的图片给它，查看结果：
# The dataset may take a few seconds to start, as it fills its shuffle buffer.
# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
# image_batch, label_batch = next(iter(keras_ds))
# train_image_batch, train_label_batch = next(iter(train_keras_ds))
# test_image_batch, test_label_batch = next(iter(test_keras_ds))
train_image_batch, train_label_batch = next(iter(train_ds))
test_image_batch, test_label_batch = next(iter(test_ds))

# CNN-Keras
train_label_batch = utils.to_categorical(train_label_batch, num_classes)
test_label_batch = utils.to_categorical(test_label_batch, num_classes)
# train_image_batch = train_image_batch.reshape((750, 256, 256, 1))
# test_image_batch = test_image_batch.reshape((250, 256, 256, 1))
print('train:', train_image_batch.shape)
print('test: ', test_image_batch.shape)

# # # Model Area

# Logging custom scalars
# What if you want to log custom values, such as a dynamic learning rate? To do that, you need to use the TensorFlow Summary API.
# Retrain the regression model and log a custom learning rate. Here's how:
# 1.Create a file writer, using tf.summary.create_file_writer().
# 2.Define a custom learning rate function. This will be passed to the Keras LearningRateScheduler callback.
# 3.Inside the learning rate function, use tf.summary.scalar() to log the custom learning rate.
# 4.Pass the LearningRateScheduler callback to Model.fit().

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

# lr_callback = callbacks.LearningRateScheduler(lr_schedule)
# callback = callbacks.LearningRateScheduler(scheduler)
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=1e-07, cooldown=0, min_lr=0.001)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1e-5, patience=0, verbose=1, min_delta=1e-9, cooldown=1, min_lr=1e-9)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1e-5, patience=0, verbose=1, min_delta=1e-7, min_lr=1e-7)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=1e-5, patience=5, verbose=1, mode='auto', min_delta=0, cooldown=0, min_lr=1e-6)
# reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=1e-5, patience=5, verbose=1, mode='auto', min_delta=1e-6, cooldown=0, min_lr=1e-6)
# reduce_lr = ReduceLROnPlateau(monitor='lr', factor=1e-5, patience=0, verbose=1, mode='auto', min_delta=0, cooldown=0, min_lr=1e-6)

# earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=1)
# earlystop_callback = EarlyStopping()

'''Keras Tuner<<<---
hypermodel = CNNHyperModel(num_classes=num_classes)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    directory='.',
    project_name='tf2_cnn_battery')

tuner.search(train_image_batch,
             train_label_batch,
             epochs=epochs,
             validation_data=(test_image_batch, test_label_batch))
--->>>Keras Tuner'''

'''cnn_model<<<--'''

# cnn_model = cnn_model([128, 256, 512], [1024, 512, 256])   # 0.9531, 6315s
# cnn_model = cnn_model([128, 256, None], [256, 128, None])  # 0.9218, 2654s
# cnn_model = cnn_model([64, 128, 256], [256, 128, 64])      # 0.9609, 1309s
# cnn_model = cnn_model([64, 128, 256], [256, 128, None])    # 0.9453, 1366s
# cnn_model = cnn_model([64, 128, 256], [128, 64, 32])       # 0.9375, 1243s
# cnn_model = cnn_model([64, 128, None], [128, 64, None])    # 0.9609, 904s
# cnn_model = cnn_model([64, 128, 256], [128, 64, None])     # 0.9335, 1247s
# cnn_model = cnn_model([64, 128, None], [256, 128, None])   # 0.9648, 1024s
# cnn_model = cnn_model([64, 256, None], [256, 64, None], name='Training')  # 0.9726, s
# cnn_model = cnn_model([64, 256, None], [256, 64, 16], name='Training')    # 0.9335, s
# cnn_model = cnn_model([64, 64, 32], [64, 32, None], name='Training')      # 0.7929, 690s
# cnn_model = cnn_model([64, 64, 64], [64, 64, 64], name='Training')        # 0.7656, 716s
# cnn_model = cnn_model([128, 128, 128], [128, 128, 128], name='Training')  # 0.6835, 1745s
# cnn_model = cnn_model([32, 32, 32], [32, 32, 32], name='Training')        # 0.7304, 339s
# cnn_model = cnn_model([64, 64, None], [128, 128, None], name='Training')  # 0.9023, s
# cnn_model = cnn_model([64, 64, None], [256, 256, None], name='Training')  # 0.9531, s
# cnn_model = cnn_model([64, 512, None], [256, 32, None], name='Training')  # 0.9609, s
# cnn_model = cnn_model([64, 256, None], [256, 64, None], name='Training-model1')   # 0.7070, s
# cnn_model = cnn_model([64, 256, None], [256, 64, None], name='Training-model')    # 0.9531, s
# cnn_model = cnn_model([64, 256, None], [256, 32, None], name='Training-model')    # 0.9335, 1189s
# cnn_model = cnn_model([64, 128, 256], [256, 128, None], name='Training-model')    # 0.9718, 1635s
# cnn_model = cnn_model([64, 128, 256], [256, 128, 64], name='Training-model')      # 0.9718, 1634s
# cnn_model = cnn_model([64, 128, None], [256, 128, None], name='Training-model2')  # 0.9687, 3063s
# cnn_model = cnn_model([64, 128, 256], [256, 128, None], name='Training-model')    # 0.9765, 1972s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], name='Training-model')  # 0.9791, 4021s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], name='Training-model')      # 0.9947, 3977s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], name='Training-model')      # 0.9869, 3963s
# cnn_model = cnn_model([128, 256, 512], [256, 128, 64], name='Training-model')       # 0.9687, 5662s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], name='Training-model')    # 0.9791, 3857s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [0.1, 0.1, 0.1], name='Training-model')  # 0.9843, 3488s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [0.1, 0.1, 0.1], name='Training-model')  # 0.9687, 3858s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [0.1, None, None], name='Training-model')  # 0.9635, 3842s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], [0.09, 0.09, 0.09], name='Training-model')  # 0.9661, 3845s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [0.01, 0.01, 0.01], name='Training-model')  # 0.9765, 3853s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [1e-3, 1e-3, None], name='Training-model')  # 0.9843, 4118s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [1e-3, 1e-3, 1e-3], name='Training-model')  # 0.9583, 3864s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [1e-3, 1e-3, None], name='Training-model')  # 0.9739, 4131s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [1e-4, 1e-4, None], name='Training-model')  # 0.9739, 3851s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], [1e-4, 1e-4, None], name='Training-model')  # 0.9843, 4018s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], [1e-4, 1e-4, None], name='Training-model')  # 0.9843, 3866s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], [1e-4, 1e-4, None], name='Training-model')  # 0.9453, 3850s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], [1e-5, 1e-5, None], name='Training-model')  # 0.9739, 3844s
# cnn_model = cnn_model([128, 256, None], [256, 128, None], [1e-5, 1e-5, None], name='Training-model')  # 0.9687, 3867s
# cnn_model = cnn_model([128, 128, 256], [256, 128, None], [1e-5, 1e-5, None], name='Training-model')  # 0.9557, 2965s
# cnn_model = cnn_model([128, 256, 256], [256, 128, None], [1e-5, 1e-5, None], name='Training-model')  # 0.9895, 4457s
# cnn_model = cnn_model([256, 512, None], [512, 256, None], [1e-5, 1e-5, None], name='Training-model')  # Op:__inference_distributed_function_1348
# cnn_model = cnn_model([256, 256, None], [256, 128, None], [1e-5, 1e-5, None], name='Training-model')  # 0.9583, s
# cnn_model = cnn_model([128, 256, None], [128, 128, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9427, 3531s
# cnn_model = cnn_model([128, 256, None], [256, 128, 64], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9609, 3895s
# cnn_model = cnn_model([128, 256, None], [512, 256, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9947, 10439s
# cnn_model = cnn_model([128, 256, None], [512, 256, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9583, 10201s
# cnn_model = cnn_model([128, 256, None], [512, 256, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9947, 10590s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9869, 564s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [1e-6, 1e-6, 1e-6], name='Training-model')  # 0.9739, 586s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9739, 586s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.9, s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [1e-5, 1e-5, 1e-5], name='Training-model')  # 0.5390, 1059s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [1e-1, 1e-1, 1e-1], name='Training-model')  # 0.5195, 1050s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [0.2, 0.2, 0.2], name='Training-model')  # 0.9609, 1068s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [0.1, 0.1, 0.1], name='Training-model')  # 0.9765, 1074s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [0.1, 0.1, 0.1], name='Training-model-batch256-epoch50')  # 0.97, 255s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [0.1, 0.1, 0.1], name='Training-model-batch256-epoch100-adam')  # 0.97, 500s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [0.1, 0.1, 0.1], name='Training-model-batch256-epoch100-nadam')  # 0.975, 502s
# cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [0.5, 0.5, 0.5], name='Training-model-batch256-epoch100-adam')  # 0.6562, 81s
cnn_model = cnn_model([128, 256, 512], [512, 256, 128], [0.1, 0.1, 0.1], name='Training-model-batch256-epoch100-adam')  # 0.98, 303s

cnn_model.summary()

# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

# example_batch_loss = loss(train_label_batch, test_label_batch)

# Configures the model for training.
# Define the model optimizer, the loss function and the accuracy metrics
# 训练模式设置
# optimizer=optimizers.Adadelta(),  # 0.6413
# optimizer=optimizers.Adagrad(),   # 0.9699
# optimizer=optimizers.Ftrl(),      # 0.9807
# optimizer=optimizers.SGD(),       # 0.9878
# optimizer=optimizers.Adamax(),    # 0.9887
# optimizer=optimizers.RMSprop(),   # 0.9888
# optimizer=optimizers.Nadam(),     # 0.9913
cnn_model.compile(
    loss=losses.categorical_crossentropy,
    # loss=losses.sparse_categorical_crossentropy,
    # optimizer=optimizers.Adam(),
    optimizer=optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True,
        name='Adam'
    ),
    # optimizer=optimizers.Nadam(
    #     learning_rate=learning_rate,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-05,
    #     name='Nadam'
    # ),
    metrics=['accuracy'],
    run_eagerly=True
)
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# adam with weight decay
# adamw = AdamWOptimizer(weight_decay=1e-4)  # TensorFlow Core r1.14
# model.compile(loss=losses.categorical_crossentropy,
#               optimizer=adamw,  # 0.9903
#               metrics=['accuracy'])


# Train and validate model
# 模型训练参数设置 + 训练
# The fit() method - trains the model
print("Training ... With default parameters, this takes less than 10 seconds.")
training_history = cnn_model.fit(train_image_batch,  # input
                                 train_label_batch,  # output
                                 epochs=epochs,
                                 shuffle=True,
                                 verbose=1,  # Suppress chatty output; use Tensorboard instead
                                 validation_data=(test_image_batch, test_label_batch),
                                 # batch_size=args.batch_size,
                                 # validation_freq=1,
                                 # callbacks=[tensorboard_callback, lr_callback],)
                                 # callbacks=[tensorboard_callback, callback],)
                                 # callbacks=[tensorboard_callback, earlystop_callback, reduce_lr],
                                 callbacks=[tensorboard_callback, reduce_lr],
                                 # callbacks=[tensorboard_callback],
                                 use_multiprocessing=True,
                                 validation_freq=1
                                 )

# Examining loss using TensorBoard
# Now, start TensorBoard, specifying the root log directory you used above.
# Wait a few seconds for TensorBoard's UI to spin up.
# $ tensorboard --logdir logs/scalars


# 模型评估
# The evaluate() method - gets the loss statistics
# score = model.evaluate(train_image_batch, train_label_batch,
score = cnn_model.evaluate(test_image_batch,
                           test_label_batch,
                           batch_size=64,
                           use_multiprocessing=True,
                           verbose=0)

# 模型预测
# The predict() method - predict the outputs for the given inputs
# model.predict(test_image_batch, test_label_batch, batch_size=args.batch_size, verbose=0)

# Print results
print('=' * 40)
print('Train Accuracy:', round(training_history.history['accuracy'][epochs - 1], 4))
print('Test Loss:', round(score[0], 4))
print('Test Accuracy:', score[1])

'''--->>>cnn_model'''

# plot_history([(training_history, 'loss'),
#               (training_history, 'accuracy')])

# 格式化日期、時間成 2019-02-20 11:45:39 形式
print('=' * 40)
print("From:", fromtime)
# logging root 輸出
# logging.info('開始時間: {}'.format(fromtime))
# logging.info('Train Accuracy: {}'.format(round(training_history.history['accuracy'][epochs - 1], 4)))
# logging.info('Test Loss: {}'.format(round(score[0], 4)))
# logging.info('Test Accuracy: {}'.format(score[1]))
log.logger.info('開始時間: {}'.format(fromtime))
log.logger.info('Train Accuracy: {}'.format(round(training_history.history['accuracy'][epochs - 1], 4)))
log.logger.info('Test Loss: {}'.format(round(score[0], 4)))
log.logger.info('Test Accuracy: {}'.format(score[1]))
endtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Now: ", endtime)
# logging.info('結束時間: {}'.format(endtime))
log.logger.info('結束時間: {}'.format(endtime))
# 時間相減得到秒數
endt = datetime.strptime(endtime, "%Y-%m-%d %H:%M:%S")
startt = datetime.strptime(fromtime, "%Y-%m-%d %H:%M:%S")
seconds = (endt - startt).seconds
minutes = round(seconds / 60, 2)
hours = round(minutes / 60, 2)
print("Elapsed time==", hours, "hours==", minutes, "minutes==", seconds, "seconds")
# logging.info('執行時間: {} 時 {} 分 {} 秒'.format(hours, minutes, seconds))
log.logger.info('執行時間: {} 時 {} 分 {} 秒'.format(hours, minutes, seconds))
print('=' * 60)

# 繪製趨勢圖
if plotting:
    plotout(training_history)

if 1 == 0:
    # Visualize the CNN model
    # ref. https://www.codeastar.com/visualize-convolutional-neural-network/
    # The first layer of our model, conv2d_1, is a convolutional layer
    # which consists of 30 learnable filters with 3-pixel width and height in size.
    # We do not need to define the content of those filters.
    # As the model will learn building filters by “seeing” some types of visual feature of input images,
    # such as an edge or a curve of an image.
    # The 30 filters of our first layer should look like:
    # get_weights [x, y, channel, nth convolutions layer ]
    weight_conv2d_1 = cnn_model.layers[0].get_weights()[0][:, :, 0, :]

    col_size = 6
    row_size = 5
    filter_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(12, 8))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(weight_conv2d_1[:, :, filter_index], cmap="gray")
            filter_index += 1
    plt.show()

    # In our first convolutional layer, each of the 30 filters connects to input images
    # and produces a 2-dimensional activation map per image.
    # Thus there are 30 * 42,000 (number of input images) = 1,260,000 activation maps from our first convolutional layer’s outputs.
    # We can visualize a output by using a random image from the 42,000 inputs.
    test_index = random.randrange(train_image_batch.shape[0])
    # print(test_index)
    test_img = train_image_batch[test_index]
    # print(np.squeeze(test_img).shape)
    # plt.imshow(test_img.reshape(256, 256), cmap='gray')
    plt.imshow(np.squeeze(test_img), cmap='gray')
    plt.title("Index:[{}], Value:{}".format(test_index, test_label_batch[test_index]))
    # print(test_label_batch[test_index])
    # print(np.argmax(test_label_batch[test_index]))  # inverse of to_categorical
    plt.xlabel(
        label_names[np.argmax(test_label_batch[test_index])] + '=' + str(np.argmax(test_label_batch[test_index])))
    plt.show()

    # Okay, the “7” digit image is our input.
    # We then create a “display_activation” function to show the activation maps within a selected layer.
    layer_outputs = [layer.output for layer in cnn_model.layers]
    activation_model = Model(inputs=cnn_model.input, outputs=layer_outputs)
    # activations = activation_model.predict(test_img.reshape(1, 256, 256, 1))  # error
    activations = activation_model.predict(tf.expand_dims(test_img, 0))


    def display_activation(activations, col_size, row_size, act_index):
        activation = activations[act_index]
        activation_index = 0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
        for row in range(0, row_size):
            for col in range(0, col_size):
                ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
                activation_index += 1


    # Then we display the 30 activation maps from the first layer in 6 columns and 5 rows.
    display_activation(activations, 6, 5, 0)

    # The image size is then drastically reduced.
    # Since we only keep important data like the feature’s existence rather than the feature’s exact location, we can avoid overfitting.
    # To visualize the outputs of the pooling layer,
    # we can use the “display_activation” function again with layer index set to 1 (the second layer, max_pooling2d_1).
    display_activation(activations, 6, 5, 1)

    # The Expanding Network
    # After the first 2 layers, we now have 1,260,000 inputs.
    # On our 3rd layer, another convolutional layer,
    # we are going to make 1,260,000 * 15 (number of filters) = 18,900,000 outputs.
    # But don’t panic, we only show one set of the activation maps here, i.e. 15 images.
    display_activation(activations, 5, 3, 2)

    # From the convolutional activation maps,
    # we know that our model can now find features of a “7” like a horizontal stroke,
    # a vertical stroke and a joining of strokes.
    # We apply a dropout layer as our 4th layer to reduce overfitting.
    # The dropping rate is set to 0.2, i.e. one input would be removed for every 5 inputs.
    display_activation(activations, 5, 3, 3)
    # The purpose of dropout layer is to drop certain inputs and force our model to learn from similar cases.
    # The result would be more obvious in a larger network.

    # Classification in Final Layer
    # We put outputs from the dropout layer into several fully connected layers.
    # Our model then classifies the inputs into 0 – 9 digit values at the final layer.
    act_dense_3 = activations[11]
    y = act_dense_3[0]
    x = range(len(y))
    plt.xticks(x)
    plt.bar(x, y)
    plt.show()
    # end of visualize a CNN

# serialize your model to a SavedModel object
# It includes the entire graph, all variables and weights
# Save model
# cnn_model.save('tf2_cnn_battery.h5', save_format='tf')


# load your saved model
# new_model = models.load_model('tf2_cnn_battery.h5')
# score2 = new_model.evaluate(test_image_batch, test_label_batch,
#                    batch_size=args.batch_size,
#                    verbose=0)

# print('=' * 40)
# print('Test2 Loss:', round(score2[0], 4))
# print('Test2 Accuracy:', score2[1])
# print('=' * 40)


# feature_map_batch = mobile_net(image_batch)
# print(feature_map_batch.shape)

# 构建一个包装了 MobileNet 的模型并在 tf.keras.layers.Dense 输出层之前使用 tf.keras.layers.GlobalAveragePooling2D 来平均那些空间向量：
# model = tf.keras.Sequential([
#     mobile_net,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(len(label_names), activation='softmax')])

# 现在它产出符合预期 shape(维数)的输出：
# logit_batch = model(image_batch).numpy()

# print("min logit:", logit_batch.min())
# print("max logit:", logit_batch.max())
# print()

# print("Shape:", logit_batch.shape)

# 编译模型以描述训练过程：
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy',
#               metrics=["accuracy"])

# 此处有两个可训练的变量 —— Dense 层中的 weights（权重）和 bias（偏差）：
# len(model.trainable_variables)

# model.summary()

# 你已经准备好来训练模型了。
# 注意，出于演示目的每一个 epoch 中你将只运行 3 step，但一般来说在传递给 model.fit() 之前你会指定 step 的真实数量，如下所示：
# steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()
# print(steps_per_epoch)

# model.fit(ds, epochs=1, steps_per_epoch=3)


'''
性能
注意：这部分只是展示一些可能帮助提升性能的简单技巧。深入指南，请看：输入 pipeline（管道）的性能。
上面使用的简单 pipeline（管道）在每个 epoch 中单独读取每个文件。在本地使用 CPU 训练时这个方法是可行的，但是可能不足以进行 GPU 训练并且完全不适合任何形式的分布式训练。
'''
# default_timeit_steps = 2 * steps_per_epoch + 1
#
# def timeit(ds, steps=default_timeit_steps):
#     overall_start = time.time()
#     # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
#     # before starting the timer
#     it = iter(ds.take(steps + 1))
#     next(it)
#
#     start = time.time()
#     for i, (images, labels) in enumerate(it):
#         if i % 10 == 0:
#             print('.', end='')
#     print()
#     end = time.time()
#
#     duration = end-start
#     print("{} batches: {} s".format(steps, duration))
#     print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))
#     print("Total time: {}s".format(end-overall_start))
#
# # 当前数据集的性能是：
# ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# print(ds)
#
# timeit(ds)
