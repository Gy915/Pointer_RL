import numpy as np
from scipy import spatial
import get_label

# return train & test DataSet
class DataLoader(object):
    def __init__(self, batch_size = 128, max_length=20, dimension=2):
        self.batch_size = batch_size
        self.max_len = max_length
        self.dimension = dimension

    def gen_instance(self):
        seq = np.random.rand(self.max_len, self.dimension)
        return seq

    # return batch_size * len * dimension
    def gen_train_dataset(self):
        input_batch = []
        for _ in range(self.batch_size):
            input_ = self.gen_instance()
            input_batch.append(input_)

        return input_batch

    def gen_test_dataset(self):
        input_ = self.gen_instance()

        # enlarge scale of test data to batch
        input_batch = np.tile(input_, (self.batch_size, 1, 1))
        return input_batch

    def gen_label_train(self):
        _input_batch = self.gen_train_dataset() #B,seq, d
        label, order = get_label.label_data(_input_batch)
        return label, _input_batch, order


if __name__ == '__main__':
    dataloader = DataLoader(5, 20, 2)
    label, input_b = dataloader.gen_label_train()
    print(label)