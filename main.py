import numpy as np
from itertools import product
import time


def init_tables(n=8):
    """
    It's important to execute init_tables BEFORE start using G_ or F_ inside a function!
    """
    p = 2**n
    global G_table
    global F_table
    G_table = np.array([None]*p)
    F_table = np.array([None]*(p*p)).reshape(p, p)


def G_(y, temp, width=8):  # memo version of G
    try:
        if G_table[y] is None:
            G_table[y] = G(y2row(y, width), temp)
    except NameError:
        raise NameError("Warning: trying to use G_() before executing init_tables()")
    return G_table[y]


def F_(y1, y2, temp, width=8):  # memo version of F
    try:
        if F_table[y1][y2] is None:
            F_table[y1][y2] = F(y2row(y1, width), y2row(y2, width), temp)
    except NameError:
        raise NameError("Warning: trying to use F_() before executing init_tables()")
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


def T(k, temp):  # OLD VERSION - takes forever runtime!
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


def T_iterative(temp, n=8):
    # tic = time.perf_counter()
    init_tables()
    T_res = np.array([None]*(2**n)*n).reshape(n, 2**n)  # memo - keeping results iteratively. saving HUGE time.

    for t in range(1, n):
        if t == 1:
            for y in range(2**n):
                T_res[t][y] = sum(G_(y1, temp, width=n) * F_(y1, y, temp, width=n)
                                  for y1 in range(2**n))
        else:
            for y in range(2**n):
                T_res[t][y] = sum(T_res[t-1][y0] * G_(y0, temp, width=n) * F_(y0, y, temp, width=n)
                                  for y0 in range(2**n))

    T_final = sum(T_res[n-1][y] * G_(y, temp) for y in range(2**n))
    # print(f'T_iterative took {time.perf_counter()-tic:0.4f} seconds!')
    return T_final, T_res


def Z_temp(temp, ex):
    '''
    :param temp: The desired temperature
    :param ex: Number of exercise (3,4,5,6)
    :return: The computation of Z_temp
    '''
    n = 2 if ex in [3, 5] else 3    # Dimension of the lattice, depends on the Exercise
    if ex == 3 or ex == 4:
        return sum(np.exp(1/temp * neighbors(np.array(X).reshape(n, n)))
                   for X in product([-1, 1], repeat=n*n))
    elif ex == 5:
        return sum(G(y2row(Y[0], width=2), temp) *
                   G(y2row(Y[1], width=2), temp) *
                   F(y2row(Y[0], width=2), y2row(Y[1], width=2), temp)
                   for Y in product(range(4), repeat=2))
    elif ex == 6:
        return sum(G(y2row(Y[0], width=3), temp) *
                   G(y2row(Y[1], width=3), temp) *
                   G(y2row(Y[2], width=3), temp) *
                   F(y2row(Y[0], width=3), y2row(Y[1], width=3), temp) *
                   F(y2row(Y[1], width=3), y2row(Y[2], width=3), temp)
                   for Y in product(range(8), repeat=3))
    elif ex == 7:
        Z, T_res = T_iterative(temp, n=8)
        return Z, T_res
    return None


if __name__ == '__main__':
    print("Exercise 3 (2x2 lattice):")
    print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=3)}\n' for i in [1, 1.5, 2]])

    print("Exercise 4 (3x3 lattice):")
    print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=4)}\n' for i in [1, 1.5, 2]])

    print("Exercise 5 (2x2 lattice):")
    print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=5)}\n' for i in [1, 1.5, 2]])

    print("Exercise 6: (3x3 lattice)")
    print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=6)}\n' for i in [1, 1.5, 2]])

    print("Exercise 7: (8x8 lattice)      --  NOT FINISHED!")
    print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=7)[0]}\n' for i in [1, 1.5, 2]])
