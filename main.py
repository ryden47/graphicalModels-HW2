import numpy as np
from itertools import product


G = lambda row_s, temp: \
    np.exp((1/temp) * sum(np.multiply(row_s[:-1], row_s[1:])))

F = lambda row_s, row_t, temp: \
    np.exp((1/temp) * sum(np.multiply(row_s, row_t)))


'''I made an option to return the Set of all neighbors, just in case we'll need that in the future.
    By default, we just return the sum of all Xs*Xt, when Xs Xt are neighbors in X'''
def neighbors(X, returnSet=False):
    sum, neighbors = 0, []
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            if j+1 < X.shape[1]:
                neighbors.append([X[i][j], X[i][j+1]]) if returnSet else None
                sum += X[i][j] * X[i][j+1]
            if i+1 < X.shape[0]:
                neighbors.append([X[i][j], X[i + 1][j]]) if returnSet else None
                sum += X[i][j]*X[i+1][j]
    return neighbors if returnSet else sum


def Z_temp(temp, n):
    ret = 0
    for X in product([-1, 1], repeat=n*n):
        X = np.array(X).reshape(n, n)
        ret += np.exp(1/temp * neighbors(X))
    return ret


if __name__ == '__main__':
    print(*[f'Z(temp={i}, 2x2 lattice)  =  {Z_temp(temp=i, n=2)}\n' for i in [1, 1.5, 2]])
    print(*[f'Z(temp={i}, 3x3 lattice)  =  {Z_temp(temp=i, n=3)}\n' for i in [1, 1.5, 2]])





