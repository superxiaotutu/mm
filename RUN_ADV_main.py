# config
batch_size = 1024
eval_batch_size = 1000
data_path = '/home/kirin/DATA/SVHN_data'
work_path = "/home/kirin/PyCode/SVHN_cifarnet/"
work_name = "adv_svhn/"
EPOCH = 500

from model_base import *
import tqdm
from adversarial_attack import *

if __name__ == '__main__':
    train_img, train_label = input_fn(data_path, 'train', batch_size)
    val_img, val_label = input_fn(data_path, 'eval', eval_batch_size)

    train_OP, train_loss, train_clean_acc, train_adv_acc = model_fn(train_img, train_label, True, True)
    _, val_loss, val_clean_acc, val_adv_acc = model_fn(val_img, val_label, False, True)

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
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    save_path = work_path + work_name + 'svhn.ckpt'

    for i in range(1, 1 + EPOCH):
        pbar = tqdm.trange(73257 // batch_size + 1)
        for j in pbar:
            if j % 5 != 0:
                sess.run([train_OP])
            else:
                gs, tmp_sum, _, tmp_loss, tmp_acc_C, tmp_acc_A = sess.run(
                    [global_step, summary_op, train_OP, train_loss, train_clean_acc, train_adv_acc])
                pbar.set_description(
                    "loss:{:.2f}, clean_acc:{:.2f}, adv_acc:{:.2f}".format(tmp_loss, tmp_acc_C, tmp_acc_A))
                train_writer.add_summary(tmp_sum, gs)

        if i % 5 == 0:
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
