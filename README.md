这本是一次工程数学的作业, least_square_fit_orthogonal.py 那个代码写得自我感觉良好, 遂开源.

下述文本表述了一些编码时的数学逻辑, 若无法正常显示, 建议贴到 LaTeX 中再查看.

$$

取拟合函数 S^* = \sum_{i=0}^{n} \phi_i a_i \\
取基函数 \phi_i = x^i \\
取权重 \omega_i \\
取内积计算 (a(x), b(x)) = \sum_{i=0}^{m} a_i(x) b_i(x) \omega_i \\

通过以下等式可以求出拟合函数中的 a_i \\
其中 f 为假定的实际函数, 即 f(x_i) = y_i \\

\left[ \begin{matrix} 

(\phi_0, \phi_0) & (\phi_0, \phi_1) & \cdots & (\phi_0, \phi_n) \\
(\phi_1, \phi_0) & (\phi_1, \phi_1) & \cdots & (\phi_1, \phi_n) \\
\vdots & \vdots & \ddots & \vdots \\
(\phi_n, \phi_0) & (\phi_n, \phi_1) & \cdots & (\phi_n, \phi_n) \\

\end{matrix} \right]

\left[ \begin{matrix}

a_0 \\
a_1 \\
\vdots \\
a_n \\

\end{matrix} \right]

=

\left[ \begin{matrix}
(\phi_0, f) \\
(\phi_1, f) \\
\vdots \\
(\phi_n, f) \\
\end{matrix} \right]

$$

---

$$
当所有的基函数正交时, 原矩阵变为对角阵, 拟合函数中的系数 a_i = \frac{(\phi_i, f)}{(\phi_i, \phi_i)} \\

构造正交的基函数的方法如下, P_i 即最高次为 i 的基函数 \\

\begin{cases}
    P_0(x) = 1 \\
    P_1(x) = x - \alpha_0 \\

    P_{k+1}(x) = (x - \alpha_k)P_k(x) - \beta_k P_{k-1}(x) \\
\end{cases} \\

其中 \alpha 和 \beta 的值如下 \\

\begin{cases}

    \alpha_k = \frac{(x P_k, P_k)}{(P_k, P_k)} \\

    \beta_k = \frac{(P_k, P_k)}{(P_{k-1}, P_{k-1})} \\

\end{cases}
$$

# 许可

使用 [MIT](/LICENSE) 协议