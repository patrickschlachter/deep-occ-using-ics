#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 18/8/2 12:09
# @Author   : Yiwen Liao
# @File     : visualizations.py
# @Software : PyCharm
# @License  : Copyright(C), Yiwen Liao
# @Contact  : yiwen.liao93@gmail.com


from models.train import *
from toolkits.utils import set_seed


def run_model(dataset=None, normal_class=None):
    """Run ICSNET models

    :param dataset: Name of a desired dataset: mnist, fmnist or cifar10.
    :param normal_class: An integer value standing for the desired known class.
    :return: None
    """

    set_seed()

    data = get_data(dataset=dataset, normal_class=normal_class, data_format='tensor')
    name = dataset + '_%d' % normal_class

    train_autoencoder(data=data,
                      epoch=50,
                      batch_size=64,
                      reg=1e-5,
                      latent_fea=128,
                      name=name)

    train_icsnet(data=data,
                 thr=10,
                 epoch=800,
                 batch_size=64,
                 reg=1e-3,
                 latent_fea=128,
                 name=name)


if __name__ == '__main__':
    run_model('mnist', 6)
