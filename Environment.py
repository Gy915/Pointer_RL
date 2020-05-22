import tensorflow as tf
import numpy as np


# input order_seq: dimension , city + 1(end = first), batch
# return distance: batch_size
class Environment(object):
    def __init__(self, ordered_input_):
        self.ordered_input_ = ordered_input_
        # Ordered coordinates
        ordered_x_ = self.ordered_input_[0]  # [seq length +1, batch_size]
        self.delta_x2 = tf.transpose(tf.square(ordered_x_[1:] - ordered_x_[:-1]),
                                [1, 0])  # [batch_size, seq length]        delta_x**2
        ordered_y_ = self.ordered_input_[1]  # [seq length +1, batch_size]
        self.delta_y2 = tf.transpose(tf.square(ordered_y_[1:] - ordered_y_[:-1]),
                                [1, 0])  # [batch_size, seq length]        delta_y**2
    def get_distance(self):
        with tf.name_scope('environment'):
            # Get tour length (euclidean distance)
            inter_city_distances = tf.sqrt(self.delta_x2+self.delta_y2) # sqrt(delta_x**2 + delta_y**2) this is the euclidean distance between each city: depot --> ... ---> depot      [batch_size, seq length]
            self.distances = tf.reduce_sum(inter_city_distances, axis=1) # [batch_size]
            return self.distances

