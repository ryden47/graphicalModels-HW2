import numpy as np
from itertools import product


G = lambda row_s, temp: \
    np.exp((1 / temp) * sum(row_s[:-1]*row_s[1:]))

F = lambda row_s, row_t, temp: \
    np.exp((1 / temp) * sum(row_s*row_t))


def neighbors(X, returnSet=False):
    '''
    :param X: The lattice matrix
    :param returnSet:  False[DEFAULT] = Return sum of Xi*Xj (when Xi and Xj are neighbors)
                       True = Return the set of all neighbors.
    :return: Depends on returnSet
    '''
    sum, neighbors = 0, []
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            if j + 1 < X.shape[1]:
                neighbors.append([X[i][j], X[i][j + 1]]) if returnSet else None
                sum += X[i][j] * X[i][j + 1]
            if i + 1 < X.shape[0]:
                neighbors.append([X[i][j], X[i + 1][j]]) if returnSet else None
                sum += X[i][j] * X[i + 1][j]
    return neighbors if returnSet else sum


def y2row(y, width=8):
    """
    y: an integer in (0,...,(2**width)-1)
    """
    if not 0 <= y <= (2 ** width) - 1:
        raise ValueError(y)
    my_str = np.binary_repr(y, width=width)
    my_list = list(map(int, my_str))  # Python 3
    my_array = np.asarray(my_list)
    my_array[my_array == 0] = -1
    row = my_array
    return row


def T(k, temp):
    if k == 1:
        return lambda y2: sum(G(y2row(y1), temp) *
                              F(y2row(y1), y2row(y2), temp)
                                for y1 in range(256))
    if k in [2,3,4,5,6,7]:
        return lambda y_k_plus1: sum(T(k-1, temp)(y_k) *
                                     G(y2row(y_k), temp) *
                                     F(y2row(y_k), y2row(y_k_plus1), temp)
                                        for y_k in range(256))
    if k == 8:
        return sum(T(7, temp)(y_8) *
                   G(y2row(y_8), temp)
                    for y_8 in range(256))


def Z_temp(temp, ex):
    '''
    :param temp: The desired temperature
    :param ex: Number of exercise (3,4,5,6)
    :return: The computation of Z_temp
    '''
    ans = 0
    n = 2 if ex in [3, 5] else 3    # Dimension of the lattice, depends on the Exercise
    if ex == 3 or ex == 4:
        ans = sum(np.exp(1/temp * neighbors(np.array(X).reshape(n, n)))
                  for X in product([-1, 1], repeat=n*n))
    elif ex == 5:
        ans = sum(G(y2row(Y[0], width=2), temp) *
                  G(y2row(Y[1], width=2), temp) *
                  F(y2row(Y[0], width=2), y2row(Y[1], width=2), temp)
                    for Y in product(range(4), repeat=2))
    elif ex == 6:
        ans = sum(G(y2row(Y[0], width=3), temp) *
                  G(y2row(Y[1], width=3), temp) *
                  G(y2row(Y[2], width=3), temp) *
                  F(y2row(Y[0], width=3), y2row(Y[1], width=3), temp) *
                  F(y2row(Y[1], width=3), y2row(Y[2], width=3), temp)
                    for Y in product(range(8), repeat=3))
    elif ex == 7:
        ans = T(8, temp=temp)
    return ans


if __name__ == '__main__':
    # print("Exercise 3 (2x2 lattice):")
    # print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=3)}\n' for i in [1, 1.5, 2]])
    #
    # print("Exercise 4 (3x3 lattice):")
    # print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=4)}\n' for i in [1, 1.5, 2]])
    #
    # print("Exercise 5 (2x2 lattice):")
    # print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=5)}\n' for i in [1, 1.5, 2]])
    #
    # print("Exercise 6: (3x3 lattice)")
    # print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=6)}\n' for i in [1, 1.5, 2]])

    print(Z_temp(temp=1, ex=7)(y2=1))


    # import time
    # tic_s1 = time.perf_counter()
    # for i in range(1000):
    #     [f'Z(temp={i})  =  {Z_temp(temp=i, ex=4)}\n' for i in [1, 1.5, 2]]
    # tic_e1 = time.perf_counter()
    #
    # tic_s2 = time.perf_counter()
    # for i in range(1000):
    #     [f'Z(temp={i})  =  {Z_temp(temp=i, ex=6)}\n' for i in [1, 1.5, 2]]
    # tic_e2 = time.perf_counter()
    #
    # print(f' First methods time: {tic_e1 - tic_s1:0.4f} seconds')
    # print(f'Second methods time: {tic_e2 - tic_s2:0.4f} seconds')
