
import tensorflow as tf

def model_1():
    with tf.variable_scope("var_a"):
        a = tf.Variable(initial_value=[1, 2, 3], name="a")

    vars = [var for var in tf.trainable_variables() if var.name.startswith("var_a")]
    print(len(vars))
    return vars

def model_2():

    vars1 = model_1()

    with tf.variable_scope("var_b"):
        a = tf.Variable(initial_value=[1, 2, 3], name="a")

    vars2 = [var for var in tf.trainable_variables() if var.name.startswith("var")]
    print(len(vars2))
    return vars1


def pretrain_model1():
    print("-------- model 1 ------")
    vars = model_1()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, "./test_model/tmp.ckpt")

def train_model2():
    print("-------- model 2 ------")

    model1_vars = model_2()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=model1_vars)
        saver.restore(sess, "./test_model/tmp.ckpt")
        vars = sess.run([model1_vars])
        for var in vars:
            print(var)

step = 2
if step == 1:
    pretrain_model1()
else:
    train_model2()