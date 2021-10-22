from solve1 import calc_inner_product


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

    def __str__(self):
        s = ''
        for i in range(len(self.coefficient)-1, -1, -1):
            s += ' %d*x^%d' % (self.coefficient[i], i)
        return s.strip()


def calc_alpha(P_k, input_x, input_weight):
    '''
    need P_k, return alpha_k
    '''
    x_P_k = P_k.multi_x()
    
    tmp1 = calc_inner_product(x_P_k, P_k, input_x, input_weight)
    tmp2 = calc_inner_product(P_k, P_k, input_x, input_weight)

    return tmp1 / tmp2


def calc_beta(P_k_minus_1, P_k, input_x, input_weight):
    '''
    need P_{k-1} and P_k, return beta_k
    '''
    tmp1 = calc_inner_product(P_k, P_k, input_x, input_weight)
    tmp2 = calc_inner_product(P_k_minus_1, P_k_minus_1, input_x, input_weight)

    return tmp1 / tmp2


def generate_orthogonal(k):
    P = {}
    alpha = {}
    beta = {}

    P[0] = orthogonal_func([1])
    alpha[0] = calc_alpha(P[0], input_x, input_weight)
    P[1] = orthogonal_func([-alpha[0], 1])

    beta[0] = 0         # use beta_0 = 0 for comfortable
    beta[1] = calc_inner_product(P[1], P[1], input_x, input_weight) / calc_inner_product(P[0], P[0], input_x, input_weight)

    if k == 0:
        return P[0]
    else:
        for i in range(2, k+1):
            alpha[i - 1] = calc_alpha(P[i - 1], input_x, input_weight)

            P[i] = orthogonal_func([alpha[i - 1], 1])
        return P[k]


def test():
    f1 = orthogonal_func([1, 2, 3])     # 1 + 2*x + 3x^2
    f2 = f1.multi_k(2)                  # 2 + 4*x + 6*x^2
    f3 = f1.multi_x()                   # 0 + x + 2*x^2 + 3*x^3
    f4 = f2.add(f3)
    #f5 = f1.negate()
    f5 = f4.minus(f3)
    f6 = orthogonal_func([1, 0, 0])
    f7 = orthogonal_func([0, 0, 0])

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


if __name__ == '__main__':
    test()