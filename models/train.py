#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 18/8/2 10:57
# @Author   : Yiwen Liao
# @File     : train.py
# @Software : PyCharm
# @License  : Copyright(C), Yiwen Liao
# @Contact  : yiwen.liao93@gmail.com


from useful_packages import *
from toolkits.utils import *
from toolkits.evaluations import *
from toolkits.visualizations import *
from .build import *


INIT_SEED = 2018
RANDOM_STATE = np.random.RandomState(INIT_SEED)
RECORD_STEP = 100


def train_icsnet(data=None, thr=None, epoch=None, batch_size=None, reg=None, latent_fea=None, name=''):
    """Train an ICSNET.

    :param data: A dictionary containing the training data.
    :param thr: Splitting ratio between 1 and 99.
    :param epoch: Number of desired training epochs.
    :param batch_size: Desired number of batch size.
    :param reg: Float value for weight decay.
    :param latent_fea: A desired integer number of latent features.
    :param name: Name for saving models and results.
    :return: None
    """

    # ==================== Training data ====================
    x_train = data['x_train_normal']
    img_shape = x_train.shape[1:]
    num_train = x_train.shape[0]

    # ==================== Split the datasets ====================
    print('\nLoading the trained autoencoder...')
    autoencoder_path = './trained_models/ae_%s.h5' % name
    autoencoder = load_model(autoencoder_path, compile=False)

    typical_index, atypical_index = split_data(model=autoencoder, data=x_train, tau=thr, split_method='ssim')
    x_train_typical = x_train[typical_index]
    x_train_atypical = x_train[atypical_index]

    y_train = np.ones(shape=(num_train,))
    y_train[atypical_index] = 0
    y_ty = np.zeros(shape=(batch_size,))
    y_aty = np.ones(shape=(batch_size,))
    y_train_joint = np.ones(shape=(4*batch_size,))
    y_train_joint[2*batch_size:] = 0

    # ==================== Build ICS-classifier ====================
    customized_optimizer = optimizers.adam(lr=3e-4, beta_1=0.5, decay=1e-8)
    model_set = build_icsnet(img_shape=img_shape, reg=reg, latent_fea=latent_fea)

    icsnet = model_set[0]
    icsnet_joint = model_set[1]

    icsnet.compile(optimizer=customized_optimizer, loss='binary_crossentropy')
    icsnet_joint.compile(optimizer=customized_optimizer, loss=['binary_crossentropy', 'binary_crossentropy'],
                         loss_weights=[1., 1.], metrics=['accuracy'])

    # ==================== Train ICS-classifier ====================
    idx_ty = np.random.permutation(np.arange(0, len(x_train_typical)))
    idx_aty = np.random.permutation(np.arange(0, len(x_train_atypical)))
    idx_batch = np.random.permutation(np.arange(0, 4 * batch_size))

    baccu_val = []
    baccu_test = []
    valid_res_ref = 0
    best_test_res = 0
    for e in range(epoch):

        if (e+1) % RECORD_STEP == 0 or e == 0:
            print('\nTraining for epoch ' + str(e + 1) + '...')
            y_test_normal = icsnet.predict(data['x_test_normal'], batch_size=128)
            y_test_abnormal = icsnet.predict(data['x_test_abnormal'], batch_size=128)

            valid_baccu, test_baccu = one_class_evaluation(df_normal=y_test_normal, df_abnormal=y_test_abnormal)

            print('\nCurrent validation AUC: %.4f' % valid_baccu)
            print('Current test AUC: %.4f' % test_baccu)
            baccu_val.append(valid_baccu)
            baccu_test.append(test_baccu)

            if valid_baccu > valid_res_ref:
                icsnet.save('./trained_models/icsnet_%s_thr_%d_best.h5' % (name, thr))
                valid_res_ref = valid_baccu
                best_test_res = test_baccu

            print('\nBest valid AUC till now: %.4f' % valid_res_ref)
            print('Best test AUC till now: %.4f' % best_test_res)

        x_ty = x_train_typical[idx_ty][:batch_size]
        x_aty = x_train_atypical[idx_aty][:batch_size]

        x_ref_ty = x_train_typical[idx_ty][batch_size:2 * batch_size]
        x_ref_aty = x_train_atypical[idx_aty][batch_size:2 * batch_size]

        icsnet_joint.fit(x=[x_ty, x_ref_ty, x_aty, x_ref_aty],
                         y=[y_ty, y_aty],
                         epochs=3,
                         verbose=0,
                         batch_size=batch_size)

        x_train_batch = np.vstack([x_ty, x_ref_ty, x_aty, x_ref_aty])
        x_train_batch = x_train_batch[idx_batch]
        y_train_batch = y_train_joint[idx_batch]

        if (e + 1) % 1 == 0:
            icsnet.fit(x=x_train_batch, y=y_train_batch, batch_size=batch_size, epochs=1, verbose=0)

        np.random.shuffle(idx_ty)
        np.random.shuffle(idx_aty)
        np.random.shuffle(idx_batch)

    baccu_test = np.asarray(baccu_test)
    baccu_val = np.asarray(baccu_val)

    # Plot the balanced accuracy vs. epochs
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(0, len(baccu_test)) * RECORD_STEP,
             baccu_test, label='BACCU Test: Last score is %.4f.\nBest Score is %.4f' % (baccu_test[-1],
                                                                                        np.max(baccu_test)))
    plt.scatter(np.arange(0, len(baccu_test)) * RECORD_STEP, baccu_test)
    plt.plot(np.arange(0, len(baccu_val)) * RECORD_STEP,
             baccu_val, label='BACCU Valid: Last score is %.4f.\nBest Score is %.4f' % (baccu_val[-1],
                                                                                        np.max(baccu_val)))
    plt.scatter(np.arange(0, len(baccu_val)) * RECORD_STEP, baccu_val)

    plt.xlabel('Epoch')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig('AUC_epoch_%s_thr_%d.png' % (name, thr))
    plt.close()


def train_autoencoder(data=None, epoch=None, batch_size=None, reg=None, latent_fea=None, name=''):
    """Training autoencoder for data splitting.

    :param data: A dictionary containing the training data.
    :param epoch: Number of desired training epochs.
    :param batch_size: Desired number of batch size.
    :param reg: Float value for weight decay.
    :param latent_fea: A desired integer number of latent features.
    :param name: Name for saving models and results.
    :return: None
    """

    x_train = data['x_train_normal']
    img_shape = x_train.shape[1:]

    ae = build_ae(img_shape=img_shape, reg=reg, latent_fea=latent_fea)
    customized_optimizer = optimizers.rmsprop(lr=1e-3, decay=1e-8)
    ae.compile(optimizer=customized_optimizer, loss='mse')

    for e in range(epoch):

        print('Training for epoch %d...' % e)
        ae.save('./trained_models/ae_%s.h5' % name)

        np.random.shuffle(data['x_train_normal'])

        ae.fit(x=data['x_train_normal'],
               y=data['x_train_normal'],
               batch_size=batch_size,
               epochs=1,
               verbose=1)

    ae.save('./trained_models/ae_%s.h5' % name)

    pred = ae.predict(data['x_train_normal'][:100])
    img_visualize(data=pred, num_to_show=10, to_save=True, name='rec_train_normal_%s' % name)

    pred = ae.predict(data['x_test_normal'][:100])
    img_visualize(data=pred, num_to_show=10, to_save=True, name='rec_test_normal_%s' % name)

    pred = ae.predict(data['x_test_abnormal'][:100])
    img_visualize(data=pred, num_to_show=10, to_save=True, name='rec_test_abnormal_%s' % name)