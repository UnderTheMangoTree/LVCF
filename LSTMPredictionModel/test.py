import numpy as np
import LSTMPredictionModel.model as MD
import tensorflow as tf

def get_inputs_2(one,two):
    inputs = np.vstack((one, two))
    inputs = tf.reshape(inputs, (1, 2, 2))
    inputs = tf.cast(inputs, dtype = tf.float32)
    return inputs

def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))


def TEST():
    model = MD.model2_3()
    CP = tf.train.Checkpoint(Mymodel=model)
    CP.restore(tf.train.latest_checkpoint("./2_3"))
    index = 0
    path = '1.txt'
    datas = np.loadtxt(path)
    data_results = []
    for time in range(1, 50):
        print("times:", time)
        one = np.array(datas[index])
        two = np.array(datas[index + 1])
        inputs = get_inputs_2(one, two)
        index = index + 1
        data_results.append((one[0], one[1], 0))
        data_results.append((two[0], two[1], 0))
        for o in range(3):
            three = np.array(model(inputs)[0])
            print("precition results:",three)
            error = eucliDist(three,datas[index+1])
            print("the corresponding error:",error)
            one = two
            two = three
            inputs = get_inputs_2(one, two)
            index = index + 1
        index += 1

if __name__ == '__main__':
    TEST()


