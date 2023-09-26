# coding=utf-8
import os
import gzip
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

np.random.seed(1337)  # 为了再见性
img_rows, img_cols = 28, 28


## 加载mnist数据集合
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# 获取原始的mnist 以及 fashion数据集
def get_origin_data(path, kind):
    # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if kind == 'train':
        img, label = load_mnist(path=path, kind='train')
        return img, label
    elif kind == 'test':
        img, label = load_mnist(path=path, kind='t10k')
        return img, label


# 设置图片格式
def set_img(img, img_rows, img_cols):
    # 根据不同的backend定下不同的格式
    # if K.image_data_format() == 'channels_first':
    #     img = img.reshape(img.shape[0], 1, img_rows, img_cols)
    # else:
    #     img = img.reshape(img.shape[0], img_rows, img_cols, 1)

    print('--------------------')
    img = img.reshape(img.shape[0], img_rows, img_cols)
    print(img.shape)

    img = img.astype('float32')
    img /= 255
    # print('train_img shape:', train_img.shape)
    # print(train_img.shape[0], 'train samples')
    # print(test_img.shape[0], 'test samples')
    return img


# 设置标签格式
def set_label(label, kinds_num):
    # 转换为one_hot类型
    label = to_categorical(label, kinds_num)
    return label


## 根据原始数据集生成新的数据集
## 新的数据集有两堆：
# 第一堆包含0-k的数据
# 第二堆一部分是(k+1)-9的数据 和 第一堆的一些数据
def gene_train_data(k, path, m):
    train_img, train_label = get_origin_data(path=path, kind='train')
    first_train = []
    first_label = []
    second_train = []
    second_label = []
    for cur_y in range(0, len(train_label)):
        if 0 <= train_label[cur_y] <= k:
            first_train.append(train_img[cur_y])
            first_label.append(train_label[cur_y])
        else:
            second_train.append(train_img[cur_y])
            second_label.append(train_label[cur_y])
    first_train = np.array(first_train)
    first_label = np.array(first_label)
    second_train = np.array(second_train)
    second_label = np.array(second_label)

    second_train = np.concatenate((first_train[:m], second_train), axis=0)
    second_label = np.concatenate((first_label[:m], second_label), axis=0)

    return norm_data(first_train, first_label, second_train, second_label)


def gene_train_data_all(path, kinds_num):
    train_img, train_label = get_origin_data(path=path, kind='train')
    train_img = set_img(train_img, img_rows, img_cols)
    train_label = set_label(train_label, kinds_num)

    return train_img, train_label


def gene_test_data_new(path, kinds_num):
    test_img, test_label = get_origin_data(path, 'test')
    test_img = set_img(test_img, img_rows, img_cols)
    test_label = set_label(test_label, kinds_num)

    return test_img, test_label


def gene_test_data(path):
    test_img, test_label = get_origin_data(path, 'test')
    test_img = set_img(test_img, img_rows, img_cols)
    test_label = set_label(test_label, 10)

    return test_img, test_label


def norm_data(first_train, first_label, second_train, second_label):
    first_train = set_img(first_train, img_rows, img_cols)
    second_train = set_img(second_train, img_rows, img_cols)
    first_label = set_label(first_label, 10)
    second_label = set_label(second_label, 10)

    return first_train, first_label, second_train, second_label
