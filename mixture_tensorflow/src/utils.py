import sys
import subprocess

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("numpy")
# install("matplotlib")
# install("scipy")
# install("tensorflow_probability==0.6.0")

import os
import numpy as np
import numpy.random as npr

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, fashion_mnist
from tensorflow.contrib.eager.python import tfe
from tensorflow.examples.tutorials.mnist import input_data


from scipy.stats import mode
import time
from scipy.special import expit
from scipy.io import loadmat

def set_optimizer(args, learning_rate, momentum):
    if args == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
    elif args == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif args == "sgdnesterov":
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=True)
    elif args == "sgd":
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=False)
    return optimizer

def load_data(dataname):
    if dataname == "mnist":
        mnist = input_data.read_data_sets("mnist/MNIST_data/", one_hot=True)
        train_data, train_labels = mnist.train.images, mnist.train.labels
        test_data, test_labels = mnist.test.images, mnist.test.labels

    elif dataname == "omniglot":
        omni_raw = loadmat('../../dat/raw/omniglot/chardata.mat')
        train_data = omni_raw['data'].T.astype(np.float32)
        train_labels = omni_raw['target'].T
        test_data = omni_raw['testdata'].T.astype(np.float32)
        test_labels = omni_raw['testtarget'].T

    elif dataname == "gmm":
        train_data, train_labels = make_gmm_data(num_classes=10, output_dim=25, num_samples=50000)
        test_data, test_labels = make_gmm_data(num_classes=10, output_dim=25, num_samples=10000)
        train_data, test_data = expit(train_data), expit(test_data)

    elif dataname == "pinwheel":
        num_classes = 5
        train_data, train_labels = make_pinwheel_data(0.3, 0.05, num_classes=num_classes, num_per_class=10000, rate=0.25)
        test_data, test_labels = make_pinwheel_data(0.3, 0.05, num_classes=num_classes, num_per_class=10000, rate=0.25)
        train_data, test_data = train_data.astype(np.float32), test_data.astype(np.float32)
        train_labels = np.eye(num_classes, dtype=int)[train_labels]
        test_labels = np.eye(num_classes, dtype=int)[test_labels]
        # train_data, test_data = expit(train_data).astype(np.float32), expit(test_data).astype(np.float32)

    elif dataname == "fashionmnist":
        ((train_data, train_labels), (test_data, test_labels)) = fashion_mnist.load_data()
        train_data = train_data.reshape((-1, 784)).astype("float32") / 255.0
        test_data = test_data.reshape((-1, 784)).astype("float32") / 255.0
        train_labels = np.eye(10, dtype=int)[train_labels]
        test_labels = np.eye(10, dtype=int)[test_labels]

    return train_data, train_labels, test_data, test_labels


def make_gmm_data(num_classes, output_dim, num_samples, sepval=100):
    # num_classes: number of clusters
    # output_dim: dimensionality of each cluster
    # num_samples:number of samples generated
    # sepval: constant that controls how separated the clusters are

    # simulate cluster center with standard normal
    W = npr.normal(size=(num_classes, output_dim)) * sepval

    labels = npr.uniform(size=num_samples)
    labels = np.floor(labels * num_classes)
    imgs = np.array([npr.normal(W[int(labels[i])]) 
        for i in range(len(labels))])

    # map gaussian mixture through neural network

    # exp_data = np.exp(tf.nn.relu(data.dot(W)))
    # max_val = 1e5
    # exp_data[exp_data > max_val] = max_val
    # img_data = npr.binomial(1,  1 / (1 + np.exp(data.dot(W))))
    # assert np.sum(np.isinf(img_data)) == 0

    return imgs, labels


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = npr.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))
    
    randperm = npr.permutation(num_classes*num_per_class)

    return 10*np.einsum('ti,tij->tj', features, rotations)[randperm], labels[randperm]





# def GaussianSample(mean, var, scope=None):
#     sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
#     sample.set_shape(mean.get_shape())
#     return sample

def cross_entropy_with_logits(logits, targets):
    log_q = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(targets * log_q, 1)

def log_bernoulli_with_logits(x, logits):
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis=-1)

# def log_normal(x, mu, var):
#     return -0.5 * tf.reduce_sum(
#         tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis=-1)

def z_testacc(fittedlogit, testlabel):
    # greedy match of labels, evaluate the informativeness of z
    cat_pred = tf.argmax(fittedlogit, axis=1)
    real_pred = np.zeros_like(cat_pred)
    for cat in range(fittedlogit.shape[1]):
        idx = (cat_pred.numpy() == cat)
        lab = testlabel[idx]
        if len(lab) == 0:
            continue
        real_pred[idx] = mode(lab).mode[0]
    return np.mean(real_pred == testlabel)


# def sample_gumbel(shape, eps=1e-20): 
#     """Sample from Gumbel(0, 1)"""
#     U = tf.random_uniform(shape,minval=0,maxval=1)
#     return -tf.log(-tf.log(U + eps) + eps)

# def gumbel_softmax_sample(logits, temperature): 
#     """ Draw a sample from the Gumbel-Softmax distribution"""
#     y = logits + sample_gumbel(tf.shape(logits))
#     return tf.nn.softmax( y / temperature)

# def gumbel_softmax(logits, temperature, hard=False):
#     """Sample from the Gumbel-Softmax distribution and optionally discretize.
#     Args:
#     logits: [batch_size, n_class] unnormalized log-probs
#     temperature: non-negative scalar
#     hard: if True, take argmax, but differentiate w.r.t. soft sample y
#     Returns:
#     [batch_size, n_class] sample from the Gumbel-Softmax distribution.
#     If hard=True, then the returned sample will be one-hot, otherwise it will
#     be a probabilitiy distribution that sums to 1 across classes
#     """
#     y = gumbel_softmax_sample(logits, temperature)
#     if hard:
#         k = tf.shape(logits)[-1]
#         #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
#         y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
#         y = tf.stop_gradient(y_hard - y) + y
#     return y

# def sample_gaussian(mu, log_var):
#     std = tf.exp(log_var * 0.5)
#     eps = tf.random_normal(std.shape)
#     return mu + eps * std

