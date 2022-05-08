import numpy as np
import time, random
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
    """
    :param temp: Temperature
    :param n: Dimension of lattice (n x n)
    :return: P = A list of P calculations. P[1]=P_1|2, P[2]=P_2|3, ..., P[8]=P_8
                [example: in the matrix P[7], we say: given y8 (the column), the distribution of y7=k is (P[7])[y8][y7]]
    """
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


def ex7(temps, imgPerTemp, dim):
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
    return temps, P_all


def ex8(P, temps):
    images = [[generateImage(P[t]) for i in range(10000)] for t in range(len(temps))]
    for i, temp in enumerate(temps):
        a = 1/10000 * sum(x[0][0]*x[1][1] for x in images[i])
        b = 1/10000 * sum(x[0][0]*x[7][7] for x in images[i])
        print(f'E_(temp={temp})(x11,x22)  :=  {a:0.4f}')
        print(f'E_(temp={temp})(x11,x88)  :=  {b:0.4f}')


def p(i, j, ext_sample, temp):
    up = 1/temp * (ext_sample[i+1][j] + ext_sample[i-1][j] + ext_sample[i][j+1] + ext_sample[i][j-1])
    denominator = np.exp(up) + np.exp(-up)
    return np.exp(up)/denominator if ext_sample[i][j] == 1 else np.exp(-up)/denominator


def ex9(temps, n=8, num_of_samples=10000, num_of_sweeps=25):
    images_list = [[] for t in range(len(temps))]
    for t, temp in enumerate(temps):
        for s in range(num_of_samples):
            if s%100==0:  # just for debugging
                percent = int(s*len(temps)/num_of_samples*100)
                print(f'|| {percent}%', end="\n" if percent%10==0 else " ")
            sample = np.random.randint(low=0, high=2, size=(n, n))*2-1
            extend_sample = np.zeros((n+2, n+2))
            extend_sample[1:n+1, 1:n+1] = sample
            for sweep in range(num_of_sweeps):
                for idx, site in np.ndenumerate(extend_sample[1:n+1, 1:n+1]):
                    px = p(*idx, extend_sample, temp=temp)
                    if px < 1:
                        extend_sample[idx] *= -1
                    elif px == 1:
                        extend_sample[idx] = random.choice([-1, 1])
            images_list[t].append(extend_sample)
    for t, temp in enumerate(temps):
        e_x11x22 = 1/num_of_samples * sum(x[1][1]*x[2][2] for x in images_list[t])
        e_x11x88 = 1/num_of_samples * sum(x[1][1]*x[8][8] for x in images_list[t])
        print(f'E_(temp={temp})(x11,x22)  :=  {e_x11x22:0.4f}')
        print(f'E_(temp={temp})(x11,x88)  :=  {e_x11x88:0.4f}')
    i_stopped_here = 1


def Z_temp(temp, ex):
    """
    :param temp: The desired temperature
    :param ex: Number of exercise (3,4,5,6)
    :return: The computation of Z_temp by method from exercise ex
    """
    values = [-1, 1]
    if ex == 3:
        return sum(np.exp(1 / temp * (x1 * x2 + x1 * x3 + x2 * x4 + x3 * x4))
                   for x1 in values for x2 in values for x3 in values for x4 in values)
    elif ex == 4:
        return sum(np.exp(1/temp * (x1*x2 + x2*x3 + x4*x5 + x5*x6 + x7*x8 + x8*x9 + x1*x4 + x2*x5 + x3*x6 + x4*x7 + x5*x8 + x6*x9))
                   for x1 in values for x2 in values for x3 in values for x4 in values for x5 in values for x6 in values for x7 in values for x8 in values for x9 in values)
    elif ex == 5:
        values = [y2row(y, width=2) for y in range(4)]
        return sum(G(y1, temp) * G(y2, temp) * F(y1, y2, temp)
                   for y1 in values for y2 in values)
    elif ex == 6:
        values = [y2row(y, width=3) for y in range(8)]
        return sum(G(y1, temp) * G(y2, temp) * G(y3, temp) * F(y1, y2, temp) * F(y2, y3, temp)
                   for y1 in values for y2 in values for y3 in values)
    return None


if __name__ == '__main__':
    # print("Exercise 3 (2x2 lattice):", end="")
    # print(*[f"\nZ(temp={i})  =  {Z_temp(temp=i, ex=3)}" for i in [1, 1.5, 2]])
    #
    # print("\nExercise 4 (3x3 lattice):", end="")
    # print(*[f'\nZ(temp={i})  =  {Z_temp(temp=i, ex=4)}' for i in [1, 1.5, 2]])
    #
    # print("\nExercise 5 (2x2 lattice):", end="")
    # print(*[f'\nZ(temp={i})  =  {Z_temp(temp=i, ex=5)}' for i in [1, 1.5, 2]])
    #
    # print("\nExercise 6: (3x3 lattice)", end="")
    # print(*[f'\nZ(temp={i})  =  {Z_temp(temp=i, ex=6)}' for i in [1, 1.5, 2]])

    # print("\nExercise 7: Printing images... (8x8 lattice)")
    # temps, P_all = ex7(temps=[1, 1.5], imgPerTemp=10, dim=8)
    # print("Printed successfully!")
    #
    # print("\nExercise 8: Calculating empirical expectations... ")
    # ex8(P_all, temps=temps)

    print("\nExercise 9: --- NOT FINISHED ---")
    ex9(temps=[1])


