import numpy as np
from itertools import product
import time
import matplotlib.pyplot as plt


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


def find_P(temp, n=8):
    '''
    :param temp: Temperature
    :param n: Dimension of lattice (n x n)
    :return: P = A list of P calculations. P[1]=P_1|2, P[2]=P_2|3, ..., P[8]=P_8
                [example: in the matrix P[7], we say: given y8 (the column), the distribution of y7=k is (P[7])[y8][y7]]
    '''
    init_tables()
    T_res = np.array([None]*(2**n)*n).reshape(n, 2**n)  # memo - keeping results iteratively. saving HUGE time.
    T_res[0][:] = 1

    for t in range(1, n):
        for y in range(2**n):
            T_res[t][y] = sum(T_res[t-1][y0] * G_(y0, temp, width=n) * F_(y0, y, temp, width=n)
                              for y0 in range(2**n))
    T_n = sum(T_res[n-1][y] * G_(y, temp) for y in range(2**n))

    P = [["skip idx 0"]]
    for k in range(1, n):
        P_yk_yk1 = np.array([[(T_res[k-1][y_k] * G_(y_k, temp, n) * F_(y_k, y_k1, temp, n)) / T_res[k][y_k1]
                            for y_k1 in range(2**n)] for y_k in range(2**n)])
        P.append(P_yk_yk1)
    P_n = np.array([(T_res[n-1][y] * G_(y, temp, n)) / T_n
                    for y in range(2**n)], dtype='float64')
    P.append(P_n)
    return P


def generateImage(P):
    n = len(P)-1
    y_n = np.random.choice(range(2**n), p=P[n])
    image = y2row(y_n, width=n)
    y_iPlus1 = y_n
    for i in range(n-1, 0, -1):
        y_i = np.random.choice(range(2**n), p=P[i].transpose()[y_iPlus1])
        image = np.append(y2row(y_i, width=n), image)
        y_iPlus1 = y_i
    image = image.reshape(n, n)
    return image


def exercise7(temps, imgPerTemp, dim):
    P_all = [find_P(temp=t, n=dim) for t in temps]
    images = [[generateImage(P_all[t]) for i in range(imgPerTemp)] for t in range(len(temps))]
    fig, ax = plt.subplots(len(temps), imgPerTemp)
    for t in range(len(temps)):
        for m in range(10):
            ax[t, m].imshow(images[t][m], interpolation='None', cmap='Greys', vmin=-1, vmax=1)
    for ax1, t_label in zip(ax[:, 0], [f"Temp {t}" for t in temps]):
        ax1.set_ylabel(t_label, rotation=90, size=24)
    fig.set_size_inches(25, 10)
    fig.tight_layout()
    fig.suptitle(f"Exercise 7: Results for a {dim}x{dim} lattice:", fontsize=32)
    plt.show()


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
    exercise7(temps=[1, 1.5, 2], imgPerTemp=10, dim=8)
    # print(*[f'Z(temp={i})  =  {Z_temp(temp=i, ex=7)[0]}\n' for i in [1, 1.5, 2]])

    # P = Z_temp(temp=1, ex=7)[2]
    # pc = [None for i in range(9)]
    # pp = np.array([None for i in range(9)])
    # for i in range(8, 0, -1):
    #     pc[i] = np.random.choice(range(256))
    # for i in range(8, 0, -1):
    #     if i==8:
    #         pp[i] = P[i][pc[i]]
    #     else:
    #         pp[i] = P[i][pc[i]][pc[i+1]]
    # p8 = np.random.choice(range(256))

