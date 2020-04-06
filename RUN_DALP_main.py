# config
batch_size = 1024
eval_batch_size = 1000
data_path = '/home/kirin/DATA/SVHN_data'
work_path = "/home/kirin/PyCode/SVHN_cifarnet/"
work_name = "dalp_svhn/"
EPOCH = 100

import os
import svhn
import six
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.slim as slim

from adversarial_attack import *

tf.logging.set_verbosity(tf.logging.ERROR)

learningrate = 0.00001
global_step = tf.Variable(0, trainable=False)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)


from wideresnet28model import *
def forward_pass(x, attention_dropout=tf.ones([384]), is_train=False, only_logits=False):
    x = x / 128 - 1
    end_points = {}
    with tf.variable_scope(name_or_scope='WR28', reuse=tf.AUTO_REUSE):
        model = ResNetCifar10(10, is_training=is_train)
        global_pool = model.forward_pass(x)
        global_pool = global_pool * attention_dropout
        logits = model.fully_connected(global_pool, 10)
    if only_logits:
        return logits
    return logits, end_points


def tower_fn(is_training, feature, label):
    model_fn_eval_mode = lambda x: forward_pass(x, only_logits=True)
    bounds = (0, 255)
    adv_feature = generate_adversarial_examples(feature, bounds, model_fn_eval_mode, "pgd_12_3_10")
    total_label = tf.concat([label, label], axis=0)
    total_feature = tf.concat([feature, adv_feature], axis=0)
    onehot_label = tf.one_hot(total_label, 11)

    tmp_logits, tmp_end_point = forward_pass(total_feature)
    tmp_clean_logits, tmp_adv_logits = tf.split(tmp_logits, 2)
    tmp_attention = tf.split(tf.reduce_max(tf.nn.softmax(tmp_logits) * onehot_label, axis=1), 2)
    tmp_attention = (tmp_attention[0] + tmp_attention[1]) / 2
    tmp_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=tmp_logits, labels=total_label)) + \
               tf.reduce_mean(tf.reduce_mean(tf.square(tmp_clean_logits - tmp_adv_logits), axis=1) * tmp_attention) + \
               tf.add_n(slim.losses.get_regularization_losses())

    clean_correct_p = tf.equal(tf.argmax(tmp_clean_logits, -1), tf.cast(label, tf.int64))
    clean_accuracy = tf.reduce_mean(tf.cast(clean_correct_p, "float"))
    adv_correct_p = tf.equal(tf.argmax(tmp_adv_logits, -1), tf.cast(label, tf.int64))
    adv_accuracy = tf.reduce_mean(tf.cast(adv_correct_p, "float"))

    model_params = tf.trainable_variables()
    if not is_training:
        return tmp_loss, zip(model_params, model_params), clean_accuracy, adv_accuracy

    Attention_Dropout_tf = tf.gradients(tmp_loss, tmp_end_point['fc3'])[0] * tmp_end_point['fc3']
    Attention_Dropout_tf = tf.reduce_mean(tf.abs(Attention_Dropout_tf), axis=0)
    top_values, top_indices = tf.nn.top_k(Attention_Dropout_tf, k=384//2)
    Attention_Dropout_tf = (tf.sign(Attention_Dropout_tf - tf.reduce_min(top_values)) + 1) / 2

    logits, end_point = forward_pass(total_feature, attention_dropout=tf.stop_gradient(Attention_Dropout_tf))
    clean_logits, adv_logits = tf.split(logits, 2)
    final_attention = tf.split(tf.reduce_max(tf.nn.softmax(logits) * onehot_label, axis=1), 2)
    final_attention = (final_attention[0] + final_attention[1]) / 2
    final_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=total_label)) + \
               tf.reduce_mean(tf.reduce_mean(tf.square(clean_logits - adv_logits), axis=1) * final_attention) + \
               tf.add_n(slim.losses.get_regularization_losses())

    tower_grad = tf.gradients(final_loss, model_params)
    return final_loss, zip(tower_grad, model_params), clean_accuracy, adv_accuracy


def input_fn(data_dir, subset, batch_size):
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train'
        dataset = svhn.SVHNDataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        return image_batch, label_batch


def model_fn(features, labels, is_training):
    tower_features = tf.split(features, 4)
    tower_labels = tf.split(labels, 4)
    tower_losses = []
    tower_gradvars = []
    tower_clean_acc = []
    tower_adv_acc = []
    for i in range(4):
        worker_device = '/{}:{}'.format("GPU", i)
        with tf.device(worker_device):
            loss, gradvars, clean_AC, adv_AC = tower_fn(is_training, tower_features[i], tower_labels[i])
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
            optimizer = slim.train.AdamOptimizer(learning_rate=learningrate)
            # Create single grouped train op
            train_op = [optimizer.apply_gradients(gradvars, global_step=global_step)]
            train_op = tf.group(*train_op)
    else:
        train_op = 0

    with tf.device('/cpu:0'):
        loss = tf.reduce_mean(tower_losses, name='loss')
        clean_acc = tf.reduce_mean(tower_clean_acc)
        adv_acc = tf.reduce_mean(tower_adv_acc)
    return train_op, loss, clean_acc, adv_acc


import tqdm
from adversarial_attack import *

if __name__ == '__main__':
    train_img, train_label = input_fn(data_path, 'train', batch_size)
    val_img, val_label = input_fn(data_path, 'eval', eval_batch_size)

    train_OP, train_loss, train_clean_acc, train_adv_acc = model_fn(train_img, train_label, True)
    _, val_loss, val_clean_acc, val_adv_acc = model_fn(val_img, val_label, False)

    train_writer = tf.summary.FileWriter(work_path + work_name + "train", sess.graph)

    with tf.name_scope("loss"):
        tf.summary.scalar('total_loss', train_loss)

    tf.summary.image('images', train_img[-2:-1])
    with tf.name_scope("acc"):
        tf.summary.scalar('clean_accuracy', train_clean_acc)
        tf.summary.scalar('adv_accuracy', train_adv_acc)

    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    all_var = tf.global_variables()
    restore_vars = [var for var in all_var if 'Adam' not in var.name and var.name.startswith('cifarnet/')]

    saver = tf.train.Saver(restore_vars)

    ckpt = tf.train.get_checkpoint_state(work_path + work_name)
    saver.restore(sess, ckpt.model_checkpoint_path)

    save_path = work_path + work_name + 'svhn.ckpt'

    for i in range(1, 1 + EPOCH):
        pbar = tqdm.trange((73257 // batch_size + 1) * 5)
        for j in pbar:
            if j % 5 != 0:
                sess.run([train_OP])
            else:
                gs, tmp_sum, _, tmp_loss, tmp_acc_C, tmp_acc_A = sess.run(
                    [global_step, summary_op, train_OP, train_loss, train_clean_acc, train_adv_acc])
                pbar.set_description(
                    "loss:{:.2f}, clean_acc:{:.2f}, adv_acc:{:.2f}".format(tmp_loss, tmp_acc_C, tmp_acc_A))
                train_writer.add_summary(tmp_sum, gs)

        gs = sess.run(global_step)
        saver.save(sess, save_path, global_step=gs)
        val_L, val_CA, val_AA = 0, 0, 0
        for j in tqdm.trange(26000 // eval_batch_size):
            tmp_loss, tmp_acc_C, tmp_acc_A = sess.run([val_loss, val_clean_acc, val_adv_acc])
            val_L += tmp_loss
            val_CA += tmp_acc_C
            val_AA += tmp_acc_A
        print("########################")
        print()
        print(val_L / (26000 // eval_batch_size), val_CA / (26000 // eval_batch_size),
              val_AA / (26000 // eval_batch_size))
        print()
        print("########################")
