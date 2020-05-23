import tensorflow as tf
from config import get_config
from Agent import Agent
from data_generate import  DataLoader
import numpy as np
import tqdm

label, input_, order= DataLoader(batch_size=1).gen_label_train()
config, _= get_config()
input_ = np.array(input_,dtype=np.float32)
input_batch = tf.convert_to_tensor(input_)
agent = Agent(config, input_batch)
position,_,pointing = agent.compute()

loss_ = tf.losses.mean_squared_error(pointing, label)
loss = tf.reduce_mean(loss_)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(order)
    for i in range(5000):
        sess.run(train_op)
        if i%10==0:
            print(sess.run(position))
            print(sess.run(loss))











