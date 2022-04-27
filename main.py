import numpy as np
from itertools import product


G = lambda row_s, temp: \
    np.exp((1/temp) * sum(np.multiply(row_s[:-1], row_s[1:])))

F = lambda row_s, row_t, temp: \
    np.exp((1/temp) * sum(np.multiply(row_s, row_t)))


def neighborsSet(X):  # I prefer returning an Iterator, saving a lot of space. But how..?
    neighbors = []
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            neighbors.append([X[i][j], X[i][j+1]]) if j+1 < X.shape[1] else None
            neighbors.append([X[i][j], X[i+1][j]]) if i+1 < X.shape[0] else None
    return neighbors


def Z_temp(temp, n):
    ret = 0
    for X in product([-1, 1], repeat=n*n):
        X = np.array(X).reshape(n, n)
        ret += np.exp(1/temp * sum(map(lambda s: s[0]*s[1], neighborsSet(X))))
    return ret


if __name__ == '__main__':
    print(*[f'Z(temp={i}, 2x2 lattice)  =  {Z_temp(temp=i, n=2)}\n' for i in [1, 1.5, 2]])
    print(*[f'Z(temp={i}, 3x3 lattice)  =  {Z_temp(temp=i, n=3)}\n' for i in [1, 1.5, 2]])





