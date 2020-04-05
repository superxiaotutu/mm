batch_size = 1024
data_path = '/home/kirin/DATA/Cifar'
work_path = "/home/kirin/PyCode/Cifar_WideResnet/"
work_name = "clean_cifar/"


from model_base import *

from adversarial_attack import *
import tqdm

if __name__ == '__main__':
    val_img, val_label = input_fn(data_path, 'eval', 1000)
    _, val_loss, val_clean_acc, val_adv_acc = model_fn(val_img, val_label, False, True)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = "/home/kirin/PyCode/Cifar_WideResnet/clean_cifar/cifar.ckpt"

    saver.restore(sess, save_path)

    val_L, val_A, adv_A = 0, 0, 0
    for j in tqdm.trange(26000 // 1000):
        tmp_loss, tmp_acc, tmp_adv_A= sess.run([val_loss, val_clean_acc, val_adv_acc])

        val_L += tmp_loss
        val_A += tmp_acc
        adv_A += tmp_adv_A
    print("########################")
    print()
    print(adv_A / (26000 // 1000), val_A / (26000 // 1000))
    print()
    print("########################")

