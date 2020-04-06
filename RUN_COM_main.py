# config
batch_size = 1024
data_path = '/home/kirin/DATA/SVHN_data'
work_path = "/home/kirin/PyCode/SVHN_cifarnet/"
work_name = "clean_svhn/"

EPOCH = 30
from model_base import *

import tqdm

if __name__ == '__main__':
    train_img, train_label = input_fn(data_path, 'train', batch_size)
    val_img, val_label = input_fn(data_path, 'eval', 1000)

    train_OP, train_loss, train_clean_acc, train_adv_acc = model_fn(train_img, train_label, True)
    _, val_loss, val_clean_acc, val_adv_acc = model_fn(val_img, val_label, False)

    train_writer = tf.summary.FileWriter(work_path + work_name + "train", sess.graph)

    with tf.name_scope("loss"):
        tf.summary.scalar('total_loss', train_loss)

    tf.summary.image('images', train_img[0:1])
    with tf.name_scope("acc"):
        tf.summary.scalar('clean_accuracy', train_clean_acc)

    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    save_path = work_path + work_name + 'svhn.ckpt'
    ckpt = tf.train.get_checkpoint_state(work_path + work_name)
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    val_L, val_A = 0, 0
    for j in range(26000 // 1000):
        tmp_loss, tmp_acc = sess.run([val_loss, val_clean_acc])
        val_L += tmp_loss
        val_A += tmp_acc
    print("########################")
    print()
    print(val_L / (26000 // 1000), val_A / (26000 // 1000))
    print()
    print("########################")

    for i in range(1, 1 + EPOCH):
        pbar = tqdm.trange(73257 // batch_size + 1)
        for j in pbar:
            tmp_img, tmp_label = sess.run([train_img, train_label])
            gs, tmp_sum, _, tmp_loss, tmp_acc = sess.run([global_step, summary_op, train_OP, train_loss, train_clean_acc])
            pbar.set_description("loss:{:.2f}, acc:{:.2f}".format(tmp_loss, tmp_acc))
            train_writer.add_summary(tmp_sum, gs)

        if i % 5 == 0:
            gs = sess.run(global_step)
            saver.save(sess, save_path, global_step=gs)
            val_L, val_A = 0, 0
            for j in range(26000//1000):
                tmp_img, tmp_label = sess.run([val_img, val_label])
                tmp_loss, tmp_acc = sess.run([val_loss, val_clean_acc])
                val_L += tmp_loss
                val_A += tmp_acc
            print("########################")
            print()
            print(val_L / (26000//1000), val_A / (26000//1000))
            print()
            print("########################")
