#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time     : 2018/11/21 16:37
# @Author   : Yiwen Liao
# @File     : visualizations.py 
# @Software : PyCharm 
# @License  : Copyright(C), Yiwen Liao
# @Contact  : yiwen.liao93@gmail.com

from useful_packages import *


def latent_visualize(normal_encoded, abnormal_encoded, to_save=False, name='', kde_mode=True, hist_mode=False):

    """Visualize the 2D latent vectors with distributions using kernel density estimation.
    Visualize the latent representations of images. The input should only has 2 dimensions. If not, the inputs will be
    reshaped into vectors, which might lead information and correlation loss.
    :param normal_encoded: Numpy array for latent vectors of normal samples.
    :param abnormal_encoded: Numpy array for latent vectors of abnormal samples.
    :param to_save: Default False. If True, then all the plots will be saved.
    :param name: File name for saving plots.
    :param kde_mode: Default True. Visualize the distribution using kernel density estimation.
    :return: None
    """

    num_se = normal_encoded.shape[0]
    num_re = abnormal_encoded.shape[0]

    fig_width = 15
    fig_height = 8

    if len(normal_encoded.shape) > 2 or len(abnormal_encoded.shape) > 2:
        print('The data has shape of ' + str(normal_encoded.shape) + '.\nTransforming data into vector form...')
        normal_encoded = normal_encoded.reshape(num_se, -1)
        abnormal_encoded = abnormal_encoded.reshape(num_re, -1)

    num_fea = normal_encoded.shape[-1]

    plt.figure(figsize=(fig_width, fig_height))

    if num_fea > 16:
        print('The number of features is more than 16. Only shows the first 16 features...')
        num_fea = 16

    fea_index = np.arange(0, num_fea)

    cols = int(np.sqrt(num_fea))
    rows = int(math.ceil(num_fea / cols))

    for i in range(num_fea):
        plt.subplot(rows, cols, i + 1)
        sns.distplot(normal_encoded[:, fea_index[i]], label="normal", kde=kde_mode, hist=hist_mode)
        sns.distplot(abnormal_encoded[:, fea_index[i]], label="abnormal", kde=kde_mode, hist=hist_mode)
        plt.legend()
        plt.tight_layout()
    if to_save:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


def img_visualize(data, interpolation=None, num_to_show=None, shuffle_img=False, to_save=False, name=''):

    """Visualize a batch of images with desired amount.
    Visualize a batch of images. Used as internal functions of other visualization functions.
    Input should only have 3 dimensions with (samples, width, height).
    :param data: Numpy array for a batch of images.
    :param interpolation: Boolean. If use interpolation for kernels.
    :param num_to_show: The number of images to show.
    :param shuffle_img:Boolean. If randomly select the images to be shown.
    :param to_save: Default False. If True, then all the plots will be saved.
    :param name: File name for saving plots.
    :return: None
    """

    fig_width = 10
    fig_height = 10

    num_sample = data.shape[0]
    n = min(num_to_show, int(np.sqrt(num_sample)))

    digit_size = data.shape[1]
    if len(data.shape) > 3 and data.shape[-1] == 3:
        figure = np.zeros((digit_size * n, digit_size * n, 3))
    else:
        figure = np.zeros((digit_size * n, digit_size * n))

    img_order = np.arange(0, n * n)

    if shuffle_img:
        np.random.shuffle(img_order)

    for i in range(n):
        for j in range(n):
            index = img_order[n * i + j]
            digit = data[index, ...]
            if len(digit.shape) > 2 and digit.shape[-1] == 1:
                digit = np.squeeze(digit, axis=-1)
            if len(data.shape) > 3 and data.shape[-1] == 3:
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size, :] = digit
            else:
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(fig_width, fig_height))
    if len(data.shape) > 3 and data.shape[-1] == 3:
        plt.imshow(figure, interpolation=interpolation)
    else:
        plt.imshow(figure, cmap='Greys_r', interpolation=interpolation)

    plt.axis('off')

    if to_save:
        plt.savefig(name, dpi=500)
        plt.close()
    else:
        plt.show()


def feature_visualize(data, num_to_show=None, to_save=False, name=''):

    num_fea = data.shape[-1]
    width = int(np.sqrt(num_fea))
    height = int(num_fea / width)
    img_data = np.reshape(data, newshape=(-1, width, height))
    img_visualize(data=img_data, num_to_show=num_to_show, to_save=to_save, name=name)


def conv_fea_visualize(data, interpolation='bilinear', num_to_show=None, to_save=False, name=''):

    num_channel = data.shape[-1]

    for ch_index in range(5):

        temp_name = name + '_ch_' + str(ch_index+1)
        img_visualize(data=data[:, :, :, ch_index], interpolation=interpolation,
                      num_to_show=num_to_show, to_save=to_save, name=temp_name)


def single_conv_fea_visualize(data, interpolation='bilinear', to_save=False, name=''):

    num_channel = data.shape[-1]
    width = data.shape[1]
    height = data.shape[2]
    img_index = 324

    feature_set = np.zeros(shape=(num_channel, width, height))
    for ch_index in range(num_channel):
        temp_img = data[img_index, :, :, ch_index].reshape((width, height))
        feature_set[ch_index, ...] = temp_img

    img_visualize(data=feature_set, interpolation=interpolation,
                  num_to_show=num_channel, to_save=to_save, name=name)


def dim2_visualize(data, label, use_tsne=False, to_save=False, name=''):
    """Visualize the latent representations with scatter plots.
    Scatter plot only illustrate the latent representations of abnormal samples in order to show
    whether the abnormal samples can be well clustered in the latent space.
    :param data: Numpy array for latent representations of samples.
    :param label: Numpy array for corresponding labels of samples, which should have shape of (samples, 1)
    :param use_tsne: Use T-SNE to visualize latent representations.
    :param to_save: Default False. If True, then all the plots will be saved.
    :param name: File name for saving plots.
    :return: None
    """

    if len(data.shape) > 2:
        print('The data has shape of ' + str(data.shape) + '. \nTransforming data into vector form...')
        data = data.reshape(data.shape[0], -1)

    if use_tsne:
        start_time = t.clock()
        data = TSNE().fit_transform(data)
        num_fea = 2
        end_time = t.clock()
        tsne_time = end_time - start_time
        print('It took %.2f seconds to perform T-SNE...' % tsne_time)
    else:
        num_fea = data.shape[-1]

    index = np.arange(0, num_fea)

    plt.figure(figsize=(20, 10))
    plt.scatter(data[:, index[0]], data[:, index[1]], c=label, cmap='Set3', alpha=0.9)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xlabel='Latent Feature 1', fontsize=22)
    plt.ylabel(ylabel='Latent Feature 2', fontsize=22)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=22)
    # plt.legend()

    if to_save:
        #tikz_save(name + '.tex')
        plt.savefig(name, dpi=200)
        plt.close()
    else:
        plt.show()
