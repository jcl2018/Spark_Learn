import numpy as np
import matplotlib.pyplot as plt


class KMeans(object):
    def __init__(self, k=10, max_iter=20):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X, init_centroids=None):

        # 0th iteration
        X_P_list = self.partition(X, init_centroids)
        init_cost = self.cost_cal(X_P_list, init_centroids)
        print("Init cost is: \t", init_cost)
        C = self.recalc_centroid(X_P_list)

        for iter in range(self.max_iter):
            X_P_list = self.partition(X, C)
            cost = self.cost_cal(X_P_list, C)
            print("Iteration%d cost is: \t" % iter, cost)
            C = self.recalc_centroid(X_P_list)

    def partition(self, X, C):
        # X: np.array(num_X, M)
        # C: np.array(num_partition, M)
        num_X, M = X.shape
        num_partition, _ = C.shape

        all_X_list = [[] for i in range(num_partition)]

        for i in range(num_X):
            part_idx = -1
            min_dist = np.finfo(X.dtype).max
            for p in range(num_partition):
                tmp_dist = self.dist_func(X[i], C[p])
                if tmp_dist < min_dist:
                    part_idx = p
                    min_dist = tmp_dist

            all_X_list[part_idx].append(X[i])

        # stack up!
        all_X_np = []
        for i in range(len(all_X_list)):
            all_X_np.append(np.stack(all_X_list[i]))

        return all_X_np

    def recalc_centroid(self, X):
        # X: list of np.array(num_X, M), num_partition
        num_partition = len(X)
        M = X[0].shape[1]
        C = np.zeros((num_partition, M))
        for p in range(num_partition):
            C[p] = np.mean(X[p], axis=0)  # mean of each feature
        return C

    def cost_cal(self, X, C):
        # X: list of np.array(num_X, M), num_partition
        # C: np.array(num_partition, M)
        cost = 0.0
        num_partition = len(X)
        for p in range(num_partition):
            X_P = X[p]  # (num_X, M)
            C_P = C[p]  # (M)
            num_X, M = X[p].shape

            for n in range(num_X):
                cost += self.dist_func(X_P[n], C_P)
        return cost

    def dist_func(self, x, c):
        # X, C: np.array(M)
        tmp = np.square(x - c)
        tmp = np.sum(tmp)
        return tmp

    def plot(self, X, C=None):

        plt.scatter(X[:, 0], X[:, 1])

        plt.show()
        plt.savefig('test.png')
