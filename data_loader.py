import pickle
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_CIFAR_batch(filename):
    """加载单个 CIFAR-10 的 batch 数据"""
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        X = data_dict[b'data']  # 原始数据尺寸为 (10000, 3072)
        Y = data_dict[b'labels']
        # 调整形状为 (N, H, W, C)
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """加载 CIFAR-10 数据集，包含 5 个训练 batch 和 1 个测试 batch"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % b)
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_val, X_test):
    """
    数据预处理：
    - 将每张图片展平为一维向量
    - 对训练集计算均值，并将训练、验证、测试集减去均值
    """
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    return X_train, X_val, X_test
