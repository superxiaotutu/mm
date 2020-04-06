
import os
import svhn
import six
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.slim as slim

from adversarial_attack import *

tf.logging.set_verbosity(tf.logging.ERROR)

GPU_NUM = 4

learningrate = 0.01
global_step = tf.Variable(0, trainable=False)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

from wideresnet28model import *
def forward_pass(x, dropout_keep_prob, is_train=False, only_logits=False):
    x = x / 128 - 1
    end_points = {}
    with tf.variable_scope(name_or_scope='resnet9', reuse=tf.AUTO_REUSE):
        model = ResNetCifar10(is_training=is_train)
        global_pool = model.forward_pass(x)
        global_pool = tf.nn.dropout(global_pool, dropout_keep_prob)
        logits = model.fully_connected(global_pool, 10)
    if only_logits:
        return logits
    return logits, end_points


def tower_fn(is_training, feature, label, open_adv_trainn=False):
    if is_training:
        keepprob = 0.5
    else:
        keepprob = 1.0
    logits, end_point = forward_pass(feature, keepprob, is_train=is_training)

    clean_correct_p = tf.equal(tf.argmax(logits, -1), tf.cast(label, tf.int64))
    clean_accuracy = tf.reduce_mean(tf.cast(clean_correct_p, "float"))
    adv_accuracy = 0

    if open_adv_trainn:
        model_fn_eval_mode = lambda x: forward_pass(x, dropout_keep_prob=1.0, is_train=False, only_logits=True)
        bounds = (0, 255)
        adv_feature = generate_adversarial_examples(feature, bounds, model_fn_eval_mode, "pgd_12_3_10")
        adv_logits, adv_end_point = forward_pass(adv_feature, keepprob, is_train=is_training)

        clean_correct_p = tf.equal(tf.argmax(logits, -1), tf.cast(label, tf.int64))
        clean_accuracy = tf.reduce_mean(tf.cast(clean_correct_p, "float"))
        adv_correct_p = tf.equal(tf.argmax(adv_logits, -1), tf.cast(label, tf.int64))
        adv_accuracy = tf.reduce_mean(tf.cast(adv_correct_p, "float"))

        logits = tf.concat([logits, adv_logits], axis=0)
        label = tf.concat([label, label], axis=0)

    tower_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label))
    l2_var = tf.trainable_variables()
    tower_loss += 2e-4 * tf.add_n([tf.nn.l2_loss(v) for v in l2_var])

    model_params = tf.trainable_variables()
    tower_grad = tf.gradients(tower_loss, model_params)
    return tower_loss, zip(tower_grad, model_params), clean_accuracy, adv_accuracy


def input_fn(data_dir, subset, batch_size):
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train'
        dataset = svhn.SVHNDataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        return image_batch, label_batch


def model_fn(features, labels, is_training, open_adv_train=False):
    tower_features = tf.split(features, GPU_NUM)
    tower_labels = tf.split(labels, GPU_NUM)
    tower_losses = []
    tower_gradvars = []
    tower_clean_acc = []
    tower_adv_acc = []
    for i in range(GPU_NUM):
        worker_device = '/{}:{}'.format("GPU", i)
        with tf.device(worker_device):
            loss, gradvars, clean_AC, adv_AC = tower_fn(is_training, tower_features[i], tower_labels[i], open_adv_train)
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            tower_clean_acc.append(clean_AC)
            tower_adv_acc.append(adv_AC)
    # Now compute global loss and gradients.
    if is_training:
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradvars.append((avg_grad, var))
        with tf.device('/cpu:0'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.MomentumOptimizer(learning_rate=learningrate, momentum=0.9)
                train_op = [optimizer.apply_gradients(gradvars, global_step=global_step)]
                train_op = tf.group(*train_op)
    else:
        train_op = 0

    with tf.device('/cpu:0'):
        loss = tf.reduce_mean(tower_losses, name='loss')
        clean_acc = tf.reduce_mean(tower_clean_acc)
        adv_acc = tf.reduce_mean(tower_adv_acc)
    return train_op, loss, clean_acc, adv_acc
