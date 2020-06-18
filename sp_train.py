import tensorflow as tf
from config import get_config
from Agent import Agent
from data_generate import  DataLoader
import numpy as np
import matplotlib.pyplot as plt
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


label, batch_input, order = DataLoader(batch_size=1).from_txt()
config, _ = get_config()
input_ = np.array(batch_input,dtype=np.float32)
input_batch = tf.convert_to_tensor(input_)
agent_sp = Agent(config, input_batch, label=label)

x = []
y = []
#
# loss = tf.nn.softmax_cross_entropy_with_logits(logits=pointing, labels=label)
#
# opt = tf.train.AdamOptimizer(learning_rate=0.0001)
#
# train_op = opt.minimize(loss)
#
# soft_max = tf.nn.softmax(pointing)
#
# cross_entropy = -tf.reduce_sum(label*tf.log(soft_max))

print(label)
plt.ion()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(5000):

        if(i%5==0):
            x.append(i)
            _, loss, softmax, loss_by_me, position= sess.run([agent_sp.train_op,agent_sp.reduce_loss, agent_sp.soft_max, agent_sp.loss_by_me, agent_sp.positions])
            print(position)
            print(acc(order, position))
            y.append(loss)
            plt.plot(x, y)
            plt.pause(0.01)
