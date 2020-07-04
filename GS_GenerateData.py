
import numpy as np

from scipy import spatial
from sko.GA import GA_TSP
import matplotlib.pyplot as plt

def Get_Data(_num_points = 20):

    num_points = _num_points



    points_coordinate = np.random.rand(num_points, 2) * 10  # generate coordinate of points

    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')





    def cal_total_distance(routine):

        '''The objective function. input routine, return total distance.

        cal_total_distance(np.arange(num_points))

        '''

        num_points, = routine.shape

        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])





    # %% do GA






    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)

    best_points, best_distance = ga_tsp.run()
    return points_coordinate.tolist(), best_points, best_distance


if __name__ == '__main__':
    f = open("test.txt", "a")
    for i in range(128):
        points_coor, best_points, best_dis = Get_Data()
        recoder = str(points_coor) + '\n' + str(best_points) + '\n'
        f.writelines(recoder)
    f.close()

