import tensorflow as tf
from config import get_config
from Agent import Agent
from data_generate import  DataLoader
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def acc(order, out):
    order = np.array(order)
    out = np.array(out).reshape((order.shape[0], -1))
    batch = order.shape[0]
    len = order.shape[1]
    acc = 0
    for i in range(batch):
        for j in range(len):
            if(order[i][j] == out[i][j]):
                acc+=1

    return acc/(len * batch)



def train_model():
    config, _ = get_config()
    label, batch_input, order = DataLoader(batch_size=config.batch_size).from_txt("train.txt")

    input_ = np.array(batch_input, dtype=np.float32)
    input_batch = tf.convert_to_tensor(input_)
    agent_sp = Agent(config, input_batch, type=0, label=label)

    x = []
    y = []
    acc_axis=[]
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
    variable_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variable_to_save, keep_checkpoint_every_n_hours=1.0)
    plt.ion()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        acc_total = 0
        for i in range(5000):
            feed = {agent_sp.input_:input_}
            _, loss, softmax, loss_by_me, position, soft_max, scores = sess.run(
                [agent_sp.train_op, agent_sp.reduce_loss, agent_sp.soft_max, agent_sp.loss_by_me, agent_sp.positions,
                 agent_sp.soft_max, agent_sp.scores],feed_dict=feed)
            acc1 = acc(order, position)
            acc_total += acc1
            if(i%5==0):
                x.append(i)
                acc_axis.append(acc_total/5)
                print(position)
                print(acc_total / 5)

                y.append(loss)
                plt.subplot(1,2,1)
                plt.plot(x, y, color = 'r')
                plt.title("epoch-loss")
                plt.subplot(1,2,2)
                plt.plot(x, acc_axis, color='g')
                plt.title("epoch_acc")
                acc_total = 0
                plt.pause(0.01)
            if(i % 100 == 0):
                save_path = saver.save(sess, "save/" + config.sp_save + "/tmp.ckpt", global_step=i)
                print("save success at step %d" % i)

def test_model():
    config, _ = get_config()
    label, batch_input, order = DataLoader(batch_size=config.batch_size).from_txt("train.txt")

    input_ = np.array(batch_input, dtype=np.float32)
    input_batch = tf.convert_to_tensor(input_)
    agent_sp = Agent(config, input_batch, type=0, label=label)

    x = []
    y = []
    acc_axis=[]
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
    variable_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variable_to_save, keep_checkpoint_every_n_hours=1.0)
    model_file = tf.train.latest_checkpoint("save/" + config.sp_save + "/tmp.ckpt")
    with tf.Session() as sess:
        saver.restore(sess, "save/" + config.sp_save + "/tmp.ckpt-1400")
        feed = {agent_sp.input_: input_}
        position = sess.run([agent_sp.positions], feed_dict=feed)
        print(acc(order, position))

if __name__ == '__main__':

   train_model()