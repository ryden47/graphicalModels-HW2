import numpy as np
import time, sys
import matplotlib.pyplot as plt
import threading


def display_timer(seconds, temp, timer_display):
    for remaining in range(seconds, -1, -1):
        if timer_display.flag:
            sys.stdout.flush()
            break  # stop display
        sys.stdout.write("\r")
        sys.stdout.write("\t\t\t{:2d}:{} remaining for temp={} (approximately)".format(remaining//60, remaining%60, temp))
        sys.stdout.flush()
        time.sleep(1)
    while not timer_display.flag:
        sys.stdout.write("\r\t\tany second now...")
    sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.write("\r")


class TimerDisplay:
    def __init__(self):
        self.flag = False

    def stop(self):
        self.flag = True
        time.sleep(1.1)


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
    for ax1, t_label in zip(ax[:, 0], [f"Temp {t}           " for t in temps]):
        ax1.set_ylabel(t_label, rotation=45, size=24)
    fig.set_size_inches(25, 10)
    fig.tight_layout()
    fig.suptitle(f"Exercise 7: Results for a {dim}x{dim} lattice:", fontsize=32)
    plt.show()
    return temps, P_all


def ex8(P, temps):
    for t, temp in enumerate(temps):
        e_11_22, e_11_88 = 0, 0
        for i in range(10000):
            img = generateImage(P[t])
            e_11_22 = update_average(e_11_22, i, img[0, 0]*img[1, 1])
            e_11_88 = update_average(e_11_88, i, img[0, 0]*img[7, 7])
        print(f'\tE_(temp={temp:<3})(x11,x22)  =  {e_11_22:0.4f}')
        print(f'\tE_(temp={temp:<3})(x11,x88)  =  {e_11_88:0.4f}')


def single_sweep(ext_sample, temp, y=None, by_most_likely=False):       # y=x+eta is the noisy sample. relevant only for exercise 10
    for i in range(1, ext_sample.shape[0]-1):
        for j in range(1, ext_sample.shape[1]-1):
            sum_neighbors = 1/temp * (ext_sample[i + 1][j] + ext_sample[i - 1][j] + ext_sample[i][j + 1] + ext_sample[i][j - 1])
            a = np.exp(+sum_neighbors - (((1/(2*4))*((y[i, j]-1)**2)) if y is not None else 0))    # Xs =  1
            b = np.exp(-sum_neighbors - (((1/(2*4))*((y[i, j]+1)**2)) if y is not None else 0))    # Xs = -1
            coin_bias = a/(a+b)  # <coin_bias> chance to get 1. otherwise, -1
            if by_most_likely:
                ext_sample[i, j] = 1 if a > b else -1
            else:
                ext_sample[i, j] = np.random.binomial(1, coin_bias) * 2 - 1


update_average = lambda old_avg, old_total, to_add: \
    ((old_total * old_avg) + to_add) / (old_total + 1)


def create_random_sample(n):
    sample = np.random.randint(low=0, high=2, size=(n, n)) * 2 - 1
    extend_sample = np.zeros((n + 2, n + 2))
    extend_sample[1:n + 1, 1:n + 1] = sample
    return extend_sample


def independent_method(temps, n, num_of_samples, num_of_sweeps):  # this one takes A LOT of time to complete..
    print("\tCalculating empirical mean (Independent method)...")
    for t, temp in enumerate(temps):
        start, timer_display = time.perf_counter(), TimerDisplay()
        e_x11x22, e_x11x88 = 0, 0
        for s in range(num_of_samples):
            if s == 100:  # just for the display timer
                approx_seconds = int((time.perf_counter()-start)*((num_of_samples//100)-1))
                threading.Thread(target=display_timer, args=(approx_seconds,temp,timer_display,)).start()
            extend_sample = create_random_sample(n)
            for sweep in range(num_of_sweeps):
                single_sweep(extend_sample, temp=temp)
            e_x11x22 = update_average(e_x11x22, s, extend_sample[1, 1] * extend_sample[2, 2])
            e_x11x88 = update_average(e_x11x88, s, extend_sample[1, 1] * extend_sample[8, 8])
        timer_display.stop()
        print(f'\t\tE_(temp={temp:<3})(x11,x22)  =  {e_x11x22}')
        print(f'\t\tE_(temp={temp:<3})(x11,x88)  =  {e_x11x88}')


def ergodicity(temps, n, num_of_sweeps):
    print("\tCalculating empirical mean (Ergodicity method)...")
    e_x11x22, e_x11x88 = 0, 0
    for t, temp in enumerate(temps):
        start, timer_display = time.perf_counter(), TimerDisplay()
        extend_sample = create_random_sample(n)
        for s in range(-100, num_of_sweeps-100):
            if s == -1:  # just for the display timer
                approx_seconds = int((time.perf_counter()-start)*((num_of_sweeps//100)-1))
                threading.Thread(target=display_timer, args=(approx_seconds,temp,timer_display,)).start()
            single_sweep(extend_sample, temp=temp)
            if s >= 0:
                e_x11x22 = update_average(e_x11x22, s, extend_sample[1, 1] * extend_sample[2, 2])
                e_x11x88 = update_average(e_x11x88, s, extend_sample[1, 1] * extend_sample[8, 8])
        timer_display.stop()
        print(f'\t\tE_(temp={temp:<3})(x11,x22)  =  {e_x11x22}')
        print(f'\t\tE_(temp={temp:<3})(x11,x88)  =  {e_x11x88}')


def ex9(temps, n=8):
    independent_method(temps, n, num_of_samples=10000, num_of_sweeps=25)
    ergodicity(temps, n, num_of_sweeps=25000)


def ex10(temps):
    fig, ax = plt.subplots(len(temps), 5)
    for t, temp in enumerate(temps):
        ax[t, 0].set_ylabel(f"Temp={temp}           ", rotation=45, size=24)

        x = create_random_sample(n=100)
        for i in range(50):
            single_sweep(x, temp)
        ax[t, 0].imshow(x, interpolation='None', cmap='Greys', vmin=-1, vmax=1)
        ax[t, 0].set_xlabel("x", size=20)

        eta = 2*np.random.standard_normal(size=(100, 100))
        extend_eta = np.zeros((102, 102))
        extend_eta[1:101, 1:101] = eta
        y = x + extend_eta
        ax[t, 1].imshow(y, interpolation='None', cmap='Greys')
        ax[t, 1].set_xlabel("y = x+eta", size=20)

        x_post = np.copy(x)      # OR maybe it's suppose to be a new 'create_random_sample(n=100)' ??
        for i in range(50):
            single_sweep(x_post, temp, y)
        ax[t, 2].imshow(x_post, interpolation='None', cmap='Greys', vmin=-1, vmax=1)
        ax[t, 2].set_xlabel("x ~ p(x|y)", size=20)

        icm = np.copy(x)
        single_sweep(icm, temp, y, by_most_likely=True)
        ax[t, 3].imshow(icm, interpolation='None', cmap='Greys', vmin=-1, vmax=1)
        ax[t, 3].set_xlabel("argmax p(Xs | sX,y)", size=20)

        y2 = np.copy(y)
        single_sweep(y2, temp, x, by_most_likely=True)
        ax[t, 4].imshow(y2, interpolation='None', cmap='Greys')
        ax[t, 4].set_xlabel("argmax p(y|x)", size=20)

    fig.set_size_inches(25, 10)
    fig.tight_layout()
    fig.suptitle(f"Exercise 10:", fontsize=32)
    plt.show()


def Z_temp(temp, ex):
    """
    :param temp: The desired temperature
    :param ex: Number of exercise (3,4,5,6)
    :return: The computation of Z_temp by method from exercise ex
    """
    values = [-1, 1]
    if ex == 3:
        return sum(np.exp(1 / temp * (x1*x2 + x1*x3 + x2*x4 + x3*x4))
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


if __name__ == '__main__':
    start = time.perf_counter()
    print("Exercise 3 (2x2 lattice):", end="")
    print(*[f"\n\tZ(temp={i:<3})  =  {Z_temp(temp=i, ex=3)}" for i in [1, 1.5, 2]])

    print("\n\nExercise 4 (3x3 lattice):", end="")
    print(*[f'\n\tZ(temp={i:<3})  =  {Z_temp(temp=i, ex=4)}' for i in [1, 1.5, 2]])

    print("\n\nExercise 5 (2x2 lattice):", end="")
    print(*[f'\n\tZ(temp={i:<3})  =  {Z_temp(temp=i, ex=5)}' for i in [1, 1.5, 2]])

    print("\n\nExercise 6: (3x3 lattice)", end="")
    print(*[f'\n\tZ(temp={i:<3})  =  {Z_temp(temp=i, ex=6)}' for i in [1, 1.5, 2]])

    print("\n\nExercise 7: Printing images... (8x8 lattice)")
    temps, P_all = ex7(temps=[1, 1.5, 2], imgPerTemp=10, dim=8)
    print("\tPrinted successfully!")

    print("\n\nExercise 8: Calculating empirical expectations (exact sampling)... ")
    ex8(P_all, temps=temps)

    print("\n\nExercise 9:")
    ex9(temps=[1, 1.5, 2])

    print("\n\nExercise 10: Printing images...")
    ex10(temps=[1, 1.5, 2])
    print("\tPrinted successfully!")

    total = int(time.perf_counter() - start)
    print(f"\n\n\nTotal runtime: {total//60} minutes and {total%60} seconds.\n")


