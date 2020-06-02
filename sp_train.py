import tensorflow as tf
from config import get_config
from Agent import Agent
from data_generate import  DataLoader
import numpy as np
import tqdm

def acc(order, out):
    batch = order.shape[0]
    len = order.shape[1]
    acc = 0
    for i in range(batch):
        flag = 1;
        for j in range(len):
            if(order[i][j] != out[i][j]):
                flag = 0
                break

        acc+=flag

    return acc


label, batch_input, order= DataLoader(batch_size=128).from_txt()
config, _= get_config()
input_ = np.array(batch_input,dtype=np.float32)
input_batch = tf.convert_to_tensor(input_)
agent = Agent(config, input_batch)
position,_,pointing, mask_score = agent.compute()


loss_ = tf.losses.mean_squared_error(pointing, label)
loss = tf.reduce_mean(loss_)
opt = tf.train.AdamOptimizer(learning_rate=0.00001)
train_op = opt.minimize(loss_)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(5000):
        sess.run(train_op)
        if i%10==0:
            out = sess.run(position)
            print(acc(order, out))
            print(sess.run(loss))











