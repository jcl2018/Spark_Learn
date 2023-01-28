import numpy as np
import matplotlib.pyplot as plt

from kmeans_non_spark import KMeans

N_TOTAL = 4601
N_C1 = 10
N_C2 = 10
M = 58
MAX_ITER = 20

def load_data(path, data):
    with open(path, 'r') as f:
        line = f.readline()
        line_idx = 0
        while line:
            tmp_list = line.split(' ')
            # len(tmp_list) should be M
            for i in range(M):
                data[line_idx][i] = float(tmp_list[i])
            line_idx += 1
            line = f.readline()


def plot_cost(all_cost1, all_cost2):
    fig, axs = plt.subplots(2)
    fig.suptitle('Cost vs Iteration')

    plt.setp(axs, xticks=[0, 5, 10, 15, 20], xticklabels=['0', '5', '10', '15', '20'])

    idx = [i for i in range(len(all_cost1))]
    axs[0].plot(idx, all_cost1)
    axs[0].set_title('C1.txt')
    axs[1].plot(idx, all_cost2)
    axs[1].set_title('C2.txt')

    fig.tight_layout()
    plt.savefig("test.png")


def run_non_spark():
    C1 = np.zeros((N_C1, M))
    load_data("../hw2-bundle/kmeans/data/c1.txt", C1)
    # print(C1)

    C2 = np.zeros((N_C2, M))
    load_data("../hw2-bundle/kmeans/data/c2.txt", C2)

    X = np.zeros((N_TOTAL, M))
    load_data("../hw2-bundle/kmeans/data/data.txt", X)

    km_non_spark = KMeans(k=N_C1, max_iter=MAX_ITER)
    all_cost1 = km_non_spark.fit(X, init_centroids=C1)

    km_non_spark1 = KMeans(k=N_C2, max_iter=MAX_ITER)
    all_cost2 = km_non_spark1.fit(X, init_centroids=C2)

    plot_cost(all_cost1, all_cost2)

if __name__ == "__main__":
    run_non_spark()

