from least_square_fit import calc_inner_product
from least_square_fit import get_input_tuple
from least_square_fit import assumed_actual_func
from least_square_fit import draw

def another_sample_tuple():
    input_x = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    input_y = [1.00, 1.75, 1.96, 2.19, 2.44, 2.71, 3.00]
    input_weight = [1, 1, 1, 1, 1, 1, 1]
    return input_x, input_y, input_weight


class orthogonal_func:
    def __init__(self, coefficient=[]):
        # sample: 3*x^2 + 4*x + 2, self.coefficient=[2, 4, 3]
        self.coefficient = coefficient

        self._remove_useless_zeros()

    def _remove_useless_zeros(self):
        i = len(self.coefficient) - 1
        while i > 0:
            if self.coefficient[i] != 0:
                break
            i -= 1
        self.coefficient = self.coefficient[0:i+1]

    def multi_k(self, k):
        return orthogonal_func([k * x for x in self.coefficient])

    def multi_x(self):
        '''
        move all coefficient to right and insert 0 at front.
        this is multiple by x.
        '''
        coefficient = self.coefficient.copy()
        coefficient = [0] + coefficient
        return orthogonal_func(coefficient)
    
    def negate(self):
        return orthogonal_func([-x for x in self.coefficient])
    
    def add(self, other):
        max_len = max(len(self.coefficient), len(other.coefficient))
        coefficient = [0] * max_len

        for i in range(len(self.coefficient)):
            coefficient[i] = self.coefficient[i]

        for i in range(len(other.coefficient)):
            coefficient[i] += other.coefficient[i]
        
        return orthogonal_func(coefficient)
    
    def minus(self, other):
        return self.add(other.negate())

    def __call__(self, x):
        res = 0
        tmp_x = 1
        for i in range(len(self.coefficient)):
            res += tmp_x * self.coefficient[i]
            tmp_x *= x
        return res

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = ''
        for i in range(len(self.coefficient)-1, -1, -1):
            if i == 0:
                tmp = '%.3f' % self.coefficient[i]
            elif i == 1:
                tmp = '%.3fx' % self.coefficient[i]
            else:
                tmp = '%.3fx^%d' % (self.coefficient[i], i)
            
            s += ' + %s' % tmp
        return s.strip(' +')


class orthogonal_generator:
    r'''
    $$
    \begin{cases}
        P_0(x) = 1 \\
        P_1(x) = x - \alpha_0 \\

        P_{k+1}(x) = (x - \alpha_k)P_k(x) - \beta_k P_{k-1}(x) \\
    \end{cases} \\

    \begin{cases}

        \alpha_k = \frac{(x P_k, P_k)}{(P_k, P_k)} \\

        \beta_k = \frac{(P_k, P_k)}{(P_{k-1}, P_{k-1})} \\

    \end{cases}
    $$
    '''

    def __init__(self, input_x, input_weight):
        self.P = {}
        self.alpha = {}
        self.beta = {}

        self.input_x = input_x
        self.input_weight = input_weight

        self.cached = -1
        self._pre_generate()

    def _pre_generate(self):
        self.P[0] = orthogonal_func([1])
        self.alpha[0] = self._calc_alpha(self.P[0])
        self.P[1] = orthogonal_func([-self.alpha[0], 1])
        self.alpha[1] = self._calc_alpha(self.P[1])

        self.beta[0] = 0         # use beta_0 = 0 for comfortable
        self.beta[1] = self._calc_beta(self.P[0], self.P[1])

        self.cached = 1

    def _calc_alpha(self, P_k):
        '''
        need P_k, return alpha_k
        '''
        x_P_k = P_k.multi_x()
        
        tmp1 = calc_inner_product(x_P_k, P_k, self.input_x, self.input_weight)
        tmp2 = calc_inner_product(P_k, P_k, self.input_x, self.input_weight)

        return tmp1 / tmp2

    def _calc_beta(self, P_k_minus_1, P_k):
        '''
        need P_{k-1} and P_k, return beta_k
        '''
        tmp1 = calc_inner_product(P_k, P_k, self.input_x, self.input_weight)
        tmp2 = calc_inner_product(P_k_minus_1, P_k_minus_1, self.input_x, self.input_weight)

        return tmp1 / tmp2

    def generate(self, k):
        if k <= self.cached:
            return self.P[k]

        for i in range(self.cached, k):
            r'''
            P_{k+1}(x) = (x - \alpha_k)P_k(x) - \beta_k P_{k-1}(x) \\
                       = x P_k(x) - \alpha_k P_k(x) - \beta_k P_{k-1}(x)
                          tmp1          tmp2              tmp3
            '''

            tmp1 = self.P[i].multi_x()
            tmp2 = self.P[i].multi_k(self.alpha[i])
            tmp3 = self.P[i-1].multi_k(self.beta[i])
            
            self.P[i+1] = tmp1.minus(tmp2).minus(tmp3)
            self.alpha[i+1] = self._calc_alpha(self.P[i+1])
            self.beta[i+1] = self._calc_beta(self.P[i], self.P[i+1])

        self.cached = k
        return self.P[k]


def test1():
    f1 = orthogonal_func([1, 2, 3])     # 1 + 2*x + 3x^2
    f2 = f1.multi_k(2)                  # 2 + 4*x + 6*x^2
    f3 = f1.multi_x()                   # 0 + x + 2*x^2 + 3*x^3
    f4 = f2.add(f3)
    f5 = f4.minus(f3)
    f6 = orthogonal_func([1, 0, 0])
    f7 = orthogonal_func([0, 0, 0])
    f8 = f7.add(f5)

    print(f1(1))
    print(f2(1))
    print(f3(1))
    print(f4(1))
    print(f5(1))

    print(f1(2))
    print(f2(2))
    print(f3(2))
    print(f4(2))
    print(f5(2))

    print(f1)
    print(f2)
    print(f3)
    print(f4)
    print(f5)
    print(f6)
    print(f7)
    print(f8)

def test2():
    input_x, input_y, input_weight = get_input_tuple()
    g = orthogonal_generator(input_x, input_weight)

    print(g.generate(0))
    print(g.generate(1))
    print(g.generate(2))
    print(g.generate(3))
    print(g.generate(4))
    print(g.generate(5))

def main():
    K = 2

    P = []
    a = []

    input_x, input_y, input_weight = get_input_tuple()
    og = orthogonal_generator(input_x, input_weight)

    for i in range(K+1):
        P.append(og.generate(i))
    
    actual_func = assumed_actual_func(input_x, input_y)
    for i in range(K+1):
        tmp1 = calc_inner_product(P[i], actual_func, input_x, input_weight)
        tmp2 = calc_inner_product(P[i], P[i], input_x, input_weight)
        a.append(tmp1 / tmp2)
    
    fit_func = orthogonal_func([0])
    for i in range(K+1):
        fit_func = fit_func.add(P[i].multi_k(a[i]))

    print('P:', P)
    print('a:', a)
    print('alpha:', og.alpha)
    print('beta:', og.beta)
    print('function:', fit_func)

    draw(input_x, input_y, fit_func, './out2.png')


if __name__ == '__main__':
    main()
