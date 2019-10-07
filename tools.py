from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import losses
from parse_arguments import *


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


# 数据增广：
def augment_data(image, label):
    print("Augment data called!")
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    # Add more augmentation of your choice
    return image, label


# 将这些包装在一个简单的函数里，以备后用
def preprocess_image(image):
    # image = tf.image.decode_jpeg(contents=image, channels=3)  # RGB
    image = tf.image.decode_jpeg(contents=image, channels=1)  # grayscale, only .jpg
    # image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    # image = tf.image.decode_image(contents=image, channels=1)  # 1=grayscale, 2=RGB, can .jpg, .bmp
    # image = tf.io.decode_image(contents=image, channels=1)  # 1=grayscale, 2=RGB, can .jpg, .bmp
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    # Normalize the pixel values
    image /= 255.0  # normalize to [0,1] range
    return image


@tf.function
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_image_rgb(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(contents=image, channels=3)  # RGB
    image = tf.image.resize(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    # Normalize the pixel values
    image /= 255.0  # normalize to [0,1] range
    return image


# 数据集可视化
def show_image(train):
    plt.figure(figsize=(12, 12))

    for batch in train.take(1):
        for i in range(9):
            image, label = batch[0][i], batch[1][i]
            plt.subplot(3, 3, i + 1)
            plt.imshow(image.numpy())
            # plt.title(get_label_name(label.numpy()))
            plt.grid(False)
    # OR
    # for batch in tfds.as_numpy(train):
    #     for i in range(9):
    #         image, label = batch[0][i], batch[1][i]
    #         plt.subplot(3, 3, i+1)
    #         plt.imshow(image)
    #         plt.title(get_label_name(label))
    #         plt.grid(False)
    # We need to break the loop else the outer loop
    # will loop over all the batches in the training set
    # break


# The tuples are unpacked into the positional arguments of the mapped function
# 元组被解压缩到映射函数的位置参数中
@tf.function
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


# 该模型期望它的输出被标准化至 [-1,1] 范围内：
# 该函数使用“Inception”预处理，将 RGB 值从 [0, 255] 转化为 [-1, 1]
# 在将输出传递给 MobilNet 模型之前，需要将其范围从 [0,1] 转化为 [-1,1]
def change_range(image, label):
    return 2 * image - 1, label


# Visualizing Loss and Accuracy
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


def plotout(training_history):
    # plotting training and validation loss
    plt.plot(training_history.history['loss'], color='red', label='Training loss')
    plt.plot(training_history.history['val_loss'], color='green', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    # plotting training and validation accuracy
    plt.plot(training_history.history['accuracy'], color='red', label='Training acc')
    plt.plot(training_history.history['val_accuracy'], color='green', label='Validation acc')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.2
    # learning_rate = 0.001
    if epoch > 10:
        learning_rate = 0.02
        # learning_rate = 0.001
    if epoch > 20:
        learning_rate = 0.01
        # learning_rate = 0.001
    # if epoch > 30:
    #     learning_rate = 0.005
    # if epoch > 40:
    #     learning_rate = 0.001
    if epoch > 50:
        learning_rate = 0.005
        # learning_rate = 0.001
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def scheduler(epoch):
    if epoch < 10:
        return learning_rate
    else:
        return float(learning_rate * tf.math.exp(0.1 * (10 - epoch)))


# 定義模型預測結果跟正確解答之間的差異
# 因為全連接層沒使用 activation func
# from_logits= True
# def loss(y_true, y_pred):
#     return losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
def loss(labels, logits):
    return losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
