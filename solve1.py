import numpy as np
import matplotlib.pyplot as plt


def get_input_tuple():
    input_x = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    input_y = [1.20, 1.50, 1.70, 2.00, 2.24, 2.40, 2.75, 3.00]
    input_weight = [ 1, 1, 50, 1, 1, 1, 1, 1]
    return input_x, input_y, input_weight


def calc_inner_product(func1, func2, input_x, input_weight):
    assert len(input_x) == len(input_weight)

    res = 0
    for x, w in zip(input_x, input_weight):
        res += func1(x) * func2(x) * w

    return res


def assumed_actual_func(input_x, input_y):

    def func(x):
        for i in range(len(input_x)):
            if x == input_x[i]:
                return input_y[i]
        raise Exception('x not exist in input_x')

    return func


def get_base_func(sub_num):
    if sub_num == 0:
        return lambda x : 1
    if sub_num == 1:
        return lambda x : x
    if sub_num == 2:
        return lambda x : pow(x, 2)
    
    raise Exception('undefined base func for sub_num: %d' % sub_num)


def solve():
    input_x, input_y, input_weight = get_input_tuple()

    assert len(input_x) == len(input_y) and len(input_y) == len(input_weight)

    assumed_func = assumed_actual_func(input_x, input_y)

    base_funcs = [get_base_func(i) for i in range(2 + 1)]

    # Ax = y, A_{n*n}, x_{n*1}, y_{n*1}
    A = np.zeros((2+1, 2+1))
    for i in range(2 + 1):
        for j in range(2 + 1):
            A[i, j] = calc_inner_product(base_funcs[i], base_funcs[j], input_x, input_weight)
        
    y = np.zeros((2+1, 1))
    for i in range(2+1):
        y[i][0] = calc_inner_product(base_funcs[i], assumed_func, input_x, input_weight)
    
    x = np.linalg.solve(A, y)       # a_0, a_1, a_2

    def fit_func(param):
        res = 0
        for i in range(2+1):
            res += x[i] * base_funcs[i](param)
        return res

    draw(input_x, input_y, fit_func, './out.png')

    print(x)
    print('function:', '%.3fx^2 + %.3fx + %.3f' % (x[2], x[1], x[0]))

def draw(input_x, input_y, func, out_path):
    fig, ax = plt.subplots()
    ax.scatter(input_x, input_y)

    fit_x = np.linspace(0, 1, 1000)
    fit_y = np.array([func(x) for x in fit_x])
    ax.plot(fit_x, fit_y)
    plt.savefig(out_path)


def main():
    solve()


if __name__ == '__main__':
    main()
