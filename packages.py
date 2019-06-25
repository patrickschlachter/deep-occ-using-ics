#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2018/7/30 19:17
# @Author   : Yiwen Liao
# @File     : visualizations.py
# @Software : PyCharm
# @License  : Copyright(C), Yiwen Liao
# @Contact  : yiwen.liao93@gmail.com

import math, gc, os, glob
import time as t
import random as rn

import numpy as np
import seaborn as sns
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib2tikz import save as tikz_save
from skimage.measure import compare_ssim, compare_psnr
from skimage.feature import hog
from sklearn import svm, datasets
from sklearn.ensemble import IsolationForest as iForest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score

from keras import losses, metrics, optimizers, regularizers
from keras.datasets import fashion_mnist, mnist, cifar10, cifar100
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Add, Lambda, Reshape, Concatenate, Subtract, Multiply, ZeroPadding2D, Conv2DTranspose
from keras.layers import LeakyReLU, BatchNormalization, Conv2D, UpSampling2D, Flatten, MaxPooling2D, Cropping2D, Dropout
from keras.layers import AveragePooling2D, Activation, GlobalAveragePooling2D, PReLU
from keras.utils import plot_model, to_categorical
from keras.initializers import truncated_normal as tn
from keras.backend.tensorflow_backend import set_session
