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

    def from_txt(self, path):
        type = 0
        batch_input = []
        label = []
        order = []
        num = 0;
        for line in open(path):
            if(type==0):
                coor = []
                line = line[1:-2] #[,],[,],[,],[,]
                l = line.split(']') # [ '[a, b', '[a,b',  ]
                for _data in l:
                    i = 0
                    if(i == len(_data)):
                        break
                    while(_data[i]!='['):
                        i+=1

                    data = _data[i + 1:]
                    x, y = data.split(',')
                    x = float(x)
                    x = round(x, 2)
                    y = float(y[1:])
                    y = round(y, 2)
                    coor.append([x, y])
                batch_input.append(coor)
                type = 1
            else:
                list = line[1:-2].split(" ")
                _label = []
                _order = []
                for i in list:
                    if(i!=''):
                        _label.append(int(i))
                        _order.append(int(i))
                _label = get_label.one_hot(np.array(_label))
                label.append(_label)
                order.append(_order)

                type = 0

            num += 1
            if(num == self.batch_size * 2 ):
                break

        return np.array(label), np.array(batch_input), np.array(order)




if __name__ == '__main__':
    dataloader = DataLoader(5, 20, 2)

    label, batch_input = dataloader.from_txt()
    print(label, batch_input)