# config
batch_size = 256
eval_batch_size = 100
data_path = '/home/kirin/DATA/Cifar'
work_path = "/home/kirin/PyCode/Cifar_WideResnet/"
work_name = "clean_cifar/"
EPOCH = 50

from model_base import *
import tqdm

if __name__ == '__main__':
    train_img, train_label = input_fn(data_path, 'train', batch_size)
    val_img, val_label = input_fn(data_path, 'eval', eval_batch_size)

    train_OP, train_loss, train_clean_acc, train_adv_acc = model_fn(train_img, train_label, True)
    _, val_loss, val_clean_acc, val_adv_acc = model_fn(val_img, val_label, False)

    train_writer = tf.summary.FileWriter(work_path + work_name + "train", sess.graph)
    test_writer = tf.summary.FileWriter(work_path + work_name + "test", sess.graph)

    with tf.name_scope("loss"):
        t1 = tf.summary.scalar('total_loss', train_loss)

    tf.summary.image('images', train_img[0:1])
    with tf.name_scope("acc"):
        t2 = tf.summary.scalar('clean_accuracy', train_clean_acc)
        t3 = tf.summary.scalar('adv_accuracy', train_adv_acc)

    summary_op = tf.summary.merge([t1, t2])

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = work_path + work_name + 'cifar.ckpt'

    ckpt = tf.train.get_checkpoint_state(work_path + work_name)
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(EPOCH):
        pbar = tqdm.trange(50000 // batch_size + 1)
        for j in pbar:
            gs, tmp_sum, _, tmp_loss, tmp_acc_C = sess.run(
                [global_step, summary_op, train_OP, train_loss, train_clean_acc])
            pbar.set_description(
                "loss:{:.2f}, clean_acc:{:.2f}".format(tmp_loss, tmp_acc_C))
            train_writer.add_summary(tmp_sum, gs)
        if i % 1 == 0:
            gs = sess.run(global_step)
            saver.save(sess, save_path, global_step=gs)
            val_L, val_A = 0, 0
            for j in tqdm.trange(10000//eval_batch_size):
                tmp_loss, tmp_acc = sess.run([val_loss, val_clean_acc])
                val_L += tmp_loss
                val_A += tmp_acc
            summary = tf.Summary()
            summary.value.add(tag='val_acc', simple_value=val_A / (10000//eval_batch_size))
            summary.value.add(tag='val_loss', simple_value=val_L / (10000//eval_batch_size))
            test_writer.add_summary(summary, i)
            print("########################")
            print()
            print(val_L / (10000//eval_batch_size), val_A / (10000//eval_batch_size))
            print()
            print("########################")

    gs = sess.run(global_step)
    saver.save(sess, save_path, global_step=gs)
