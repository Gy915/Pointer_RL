import numpy as np
from scipy import spatial
# Batch * seq * d ; seq
# return len
def length(x_y, order):
    seq = x_y.shape[0]
    distance_matrix = spatial.distance.cdist\
(x_y, x_y, metric='euclidean')
    _len = sum([distance_matrix[order[i % seq], order[(i + 1) % seq]]
         for i in range(seq)])
    return _len

def one_hot(order):
    seq = order.shape[0]
    o_h_label = []
    for i in range(seq):
        zeros = np.zeros(seq)
        zeros[order[i]] = 1
        o_h_label.append(zeros)
    return o_h_label

# Batch * seq * d
# return [b, s, s]
def label_data(_input):
    _input = np.array(_input)
    batch = _input.shape[0]
    seq = _input.shape[1]
    label = []
    for i in range(batch):
        # Here, we get the best solution
        order = []
        best_len = 100000000
        for j in range(200):
            seed = np.random.rand(seq)
            t_order = np.argsort(seed)
            x_y = _input[i, :, :]
            _len = length(x_y, t_order)
            if(_len < best_len):
                best_len = _len
                order = t_order
        o_h_oreder = one_hot(order)
        label.append(o_h_oreder)
    return np.array(label), np.array(order)

if __name__ == '__main__':
    input_batch = []
    batch_size = 10
    for _ in range(batch_size):
        input_ = np.random.rand(20, 2)
        input_batch.append(input_)
    label = label_data(np.array(input_batch))
    print(label)