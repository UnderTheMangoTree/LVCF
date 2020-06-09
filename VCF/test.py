import tensorflow as tf
import numpy as np
import LVCF.VCF.model as MD
import cv2

def get_max(mitrix):
    m = mitrix[0, :, :]
    x = np.where(m == np.max(m))[0][0]
    y = np.where(m == np.max(m))[1][0]
    return (x, y)

def get_data(mitrix, index):
    datas = mitrix[0, index[0], index[1], :]
    x = datas[0].numpy() * 32
    y = datas[1].numpy() * 35
    r = datas[2].numpy() * 64
    return (x, y),r



def TEST():
    path = "1.jpg"
    model = MD.MD()
    checkpoint = tf.train.Checkpoint(Mymodel=model)
    checkpoint.restore(tf.train.latest_checkpoint('./model1'))
    img = np.expand_dims(np.array(cv2.imread(path),dtype=np.float32),0)
    label_pred = model(img)
    pre_index = label_pred[:, :, :, -1]
    pre_loc = label_pred[:, :, :, 0:-1]
    index_pre = get_max(pre_index)
    data_pre, r_pre = get_data(pre_loc, index_pre)
    x_pre = index_pre[0] * 35 + data_pre[1]
    y_pre = index_pre[1] * 32 + data_pre[0]
    para_pre = (y_pre, x_pre)
    print("the parameters of pupil:",para_pre)

if __name__ == '__main__':
    TEST()


