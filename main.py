import numpy as np
from itertools import product
import time

G_table = np.array([None]*256) # [None for i in range(256)]
F_table = np.array([None]*(256*256)).reshape(256, 256)


def G_(y, temp):
    if G_table[y] is None:
        G_table[y] = G(y2row(y), temp)
    return G_table[y]


def F_(y1, y2, temp):
    if F_table[y1][y2] is None:
        F_table[y1][y2] = F(y2row(y1), y2row(y2), temp)
    return F_table[y1][y2]


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


def T_(k, temp):
    if k == 1:
        return lambda y2: sum(G_(y1, temp) *
                              F_(y1, y2, temp)
                              for y1 in range(256))
    if k == 8:
        return sum(T_(7, temp)(y8) *
                   G_(y8, temp)
                   for y8 in range(256))
    else:   # k == 2,3,4,5,6,7
        return lambda y_kPlus1: sum(T_(k-1, temp)(y_k) *
                                    G_(y_k, temp) *
                                    F_(y_k, y_kPlus1, temp)
                                    for y_k in range(256))


def T(k, temp):
    if k == 1:
        return lambda y2: sum(G(y2row(y1), temp) *
                              F(y2row(y1), y2row(y2), temp)
                              for y1 in range(256))
    if k == 8:
        return sum(T(7, temp)(y8) *
                   G(y2row(y8), temp)
                   for y8 in range(256))
    else:  # k == 2,3,4,5,6,7
        return lambda y_kPlus1: sum(T(k-1, temp)(y_k) *
                                    G(y2row(y_k), temp) *
                                    F(y2row(y_k), y2row(y_kPlus1), temp)
                                    for y_k in range(256))
    # # Works for 3x3 lattice
    # if k == 1:
    #     return lambda y2: sum(G(y2row(y1, width=3), temp) *
    #                           F(y2row(y1, width=3), y2row(y2, width=3), temp)
    #                             for y1 in range(8))
    # if k == 2:
    #     return lambda y3: sum(T(1, temp)(y2) *
    #                           G(y2row(y2, width=3), temp) *
    #                           F(y2row(y2, width=3), y2row(y3, width=3), temp)
    #                             for y2 in range(8))
    # if k == 3:
    #     return sum(T(2, temp)(y_3) *
    #                G(y2row(y_3, width=3), temp)
    #                 for y_3 in range(8))


def T_iterative(temp):
    T_res = np.array([None]*256*8).reshape(8, 256)  # memo - keeping results iteratively. saving HUGE time.
    T_call = np.array([None]*8)  # we store all the lambda's here, helps with what comes next at the for loop.

    T_call[1] = lambda y2: sum(G_(y1, temp) * F_(y1, y2, temp)
                               for y1 in range(256))
    T_call[2] = lambda y3: sum(T_res[1][y2] * G_(y2, temp) * F_(y2, y3, temp)
                               for y2 in range(256))
    T_call[3] = lambda y4: sum(T_res[2][y3] * G_(y3, temp) * F_(y3, y4, temp)
                               for y3 in range(256))
    T_call[4] = lambda y5: sum(T_res[3][y4] * G_(y4, temp) * F_(y4, y5, temp)
                               for y4 in range(256))
    T_call[5] = lambda y6: sum(T_res[4][y5] * G_(y5, temp) * F_(y5, y6, temp)
                               for y5 in range(256))
    T_call[6] = lambda y7: sum(T_res[5][y6] * G_(y6, temp) * F_(y6, y7, temp)
                               for y6 in range(256))
    T_call[7] = lambda y8: sum(T_res[6][y7] * G_(y7, temp) * F_(y7, y8, temp)
                               for y7 in range(256))

    total = 0
    for t in range(1, 8):  # 1,2,...,7
        tic = time.perf_counter()
        for y in range(256):
            T_res[t][y] = T_call[t](y)
        toc = time.perf_counter()
        print(f'iteration {t}/7 took: {toc-tic:0.4f} seconds')
        total += (toc-tic)
    print(f'total runtime: {total:0.4f}')

    T8 = sum(T_res[7][y8] * G_(y8, temp) for y8 in range(256))

    # tic = time.perf_counter()
    # for y2 in range(256):
    #     T1_res[y2] = T1(y2)
    # toc = time.perf_counter()
    # print(f'{toc - tic:0.4f} seconds')
    b = 1
    return T8


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
        # ans = T_(8, temp=temp)
        ans = T_iterative(temp)
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

    print("Ztemp =", Z_temp(temp=1, ex=7))


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
