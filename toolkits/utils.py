#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2018/7/30 19:18
# @Author   : Yiwen Liao
# @File     : utils.py
# @Software : PyCharm
# @License  : Copyright(C), Yiwen Liao
# @Contact  : yiwen.liao93@gmail.com


from useful_packages import *


# ==================== Data Preprocessing ====================
def _extract_data(data=None, label=None, target_lb=None):
    """Extract dataset regarding given normal / abnormal labels-

    :param data: A numpy tensor. First axis should be the number of samples.
    :param label: The corresponding labels for the data.
    :param target_lb: An integer value standing for the only one known class.
    :return: normal data, abnormal data, normal labels, abnormal labels
    """

    if isinstance(target_lb, int):
        idx_normal = np.where(label == target_lb)[0]
        idx_abnormal = np.where(label != target_lb)[0]
    else:
        raise ValueError('Target label should be a integer...')

    data_normal = data[idx_normal]
    data_abnormal = data[idx_abnormal]
    label_normal = label[idx_normal]
    label_abnormal = label[idx_abnormal]

    return data_normal, data_abnormal, label_normal, label_abnormal


def _reshape_data(data=None, data_shape=None, num_channels=None):
    """Reshape image data into vectors / matrices / tensors.

    :param data: A numpy tensor. First axis should be the number of samples.
    :param data_shape: Desired data shape. It should be a string.
    :param num_channels: Number of the channels of the given data.
    :return: Reshaped data.
    """

    num_samples = data.shape[0]
    data = data.reshape(num_samples, -1)
    num_features = data.shape[-1]
    height = int(np.sqrt(num_features / num_channels))
    width = height

    if data_shape == 'vector':
        print('Images data are transformed into vectors...')
    elif data_shape == 'matrix':
        if num_channels == 1:
            data = data.reshape(num_samples, height, width)
        elif num_channels == 3:
            data = data.reshape(num_samples, height, width, num_channels)
            data = 0.2989 * data[:, :, :, 0] + 0.5870 * data[:, :, :, 1] + 0.1140 * data[:, :, :, 2]
            data = data.reshape(num_samples, height, width)
        else:
            raise ValueError('The input data is neither gray-scale images nor color images. Please choose other data...')
    elif data_shape == 'tensor':
        data = data.reshape(num_samples, height, width, num_channels)
    else:
        raise ValueError('No suitable data shape is found. Please enter a desired data shape...')

    return data


def get_data(dataset=None, normal_class=None, data_format=None):
    """Obtain the dataset in a desired form stored in a dictionary.

    :param dataset: The name of desired dataset: mnist, fmnist or cifar10.
    :param normal_class: The class which is considered to be known during training.
    :param data_format: The desired data shape: vector, matrix or tensor.
    :return: A dictionary containing training and testing samples.
    """

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_channel = 1
    elif dataset == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        num_channel = 1
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_channel = 3
    else:
        raise ValueError('The dataset %s is not found...' % dataset)

    # Reshape data and its label into desired format
    y_train = np.reshape(y_train, newshape=(-1,))
    y_test = np.reshape(y_test, newshape=(-1,))

    x_train = _reshape_data(data=x_train, data_shape=data_format, num_channels=num_channel)
    x_test = _reshape_data(data=x_test, data_shape=data_format, num_channels=num_channel)

    # Image normalization
    x_train = (x_train / 255).astype('float32')
    x_test = (x_test / 255).astype('float32')

    x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    if normal_class is None:
        data = {'x_train_normal': x_train,
                'y_train_normal': y_train,
                'x_test_normal': x_test,
                'y_test_normal': y_test}
    else:
        train_set = _extract_data(data=x_train, label=y_train, target_lb=normal_class)
        test_set = _extract_data(data=x_test, label=y_test, target_lb=normal_class)

        data = {'x_train_normal': train_set[0], 'x_train_abnormal': train_set[1],
                'y_train_normal': train_set[2], 'y_train_abnormal': train_set[3],
                'x_test_normal': test_set[0], 'x_test_abnormal': test_set[1],
                'y_test_normal': test_set[2], 'y_test_abnormal': test_set[3]}
    return data


# ==================== Image Processing ====================
def cal_ssim(x, x_rec):
    """Calculates SSIM between x and y.
    x should be a batch of original images. y should be a batch of reconstructed images.

    :param x: 4D tensor in form of batch_size x img_width x img_height x img_channels.
    :param x_rec: 4D tensor in form of batch_size x img_width x img_height x img_channels.
    :return: Numpy array with ssim score for each image in the given batch.
    """

    res = []
    num_img = x_rec.shape[0]
    print('Calucating SSIM...')
    for i in range(num_img):
        temp_x = x[i, ...]
        temp_rec = x_rec[i, ...]
        temp = compare_ssim(temp_x, temp_rec, multichannel=True, gaussian_weights=True)
        res.append(temp)
    print('SSIM is calculated...')
    print('_____________________')
    res = np.asarray(res)
    return res


# ==================== Data Splitting ====================
def split_data(model=None, data=None, tau=None, split_method=None):
    """Split the given data into typical and atypical normal subsets.

    :param model: A trained autoencoder model.
    :param data: A numpy tensor standing for the training data.
    :param tau: Splitting ratio between 1 to 99.
    :param split_method: The name of splitting methods. Currently only 'ssim' is supported.
    :return: Indices of typical and atypical normal samples.
    """

    print('\nSplitting data...')
    if split_method == 'ssim':
        reconstruction = model.predict(data, batch_size=128)
        similarity_score = cal_ssim(data, reconstruction)
    else:
        raise ValueError('\nNo valid splitting methods...')

    sim_thr = np.percentile(similarity_score, tau)

    typical_index = np.where(similarity_score > sim_thr)[0]
    atypical_index = np.where(similarity_score <= sim_thr)[0]

    return typical_index, atypical_index


# ==================== Misc ====================
def set_seed(first_seed=2019):

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(first_seed)
    rn.seed(10)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.6
    tf.set_random_seed(42)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)