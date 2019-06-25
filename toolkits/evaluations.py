#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2018/8/4 22:40
# @Author   : Yiwen Liao
# @File     : evaluations.py
# @Software : PyCharm
# @License  : Copyright(C), Yiwen Liao
# @Contact  : yiwen.liao93@gmail.com


import numpy as np
from sklearn.metrics import roc_auc_score


def one_class_evaluation(df_normal=None, df_abnormal=None):
    """Calculate AUC on test set.

    :param df_normal: Decision functions of normal samples in a test set.
    :param df_abnormal: Decision functions of abnormal samples in a test set.
    :return: Validation AUC and test AUC
    """

    num_valid_normal = int(0.3 * len(df_normal))
    num_valid_abnormal = int(0.3 * len(df_abnormal))

    df_normal_valid = df_normal[:num_valid_normal, ...]
    df_abnormal_valid = df_abnormal[:num_valid_abnormal, ...]

    df_normal_test = df_normal[num_valid_normal:, ...]
    df_abnormal_test = df_abnormal[num_valid_abnormal:, ...]

    valid_label = np.vstack([np.ones(shape=(num_valid_normal, 1)), np.zeros(shape=(num_valid_abnormal, 1))])
    valid_df = np.concatenate([df_normal_valid, df_abnormal_valid])

    test_label = np.concatenate([np.ones_like(df_normal_test), np.zeros_like(df_abnormal_test)])
    test_df = np.concatenate([df_normal_test, df_abnormal_test])

    auc_valid = roc_auc_score(valid_label, valid_df)
    auc_test = roc_auc_score(test_label, test_df)

    return auc_valid, auc_test
