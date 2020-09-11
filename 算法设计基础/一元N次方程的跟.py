# def newton(x0=0.001, *, a, b, c, d, e=1e-6):
#     x_n = x0 - ((a * x0 ** 3 + b * x0 ** 2 + c * x0 + d) /
#                 (3 * a * x0 ** 2 + 2 * b * x0 + c))
#     while abs(x_n - x0) > e:
#         x0 = x_n
#         x_n = x0 - ((a * x0 ** 3 + b * x0 ** 2 + c * x0 + d) /
#                 (3 * a * x0 ** 2 + 2 * b * x0 + c))
#     return x_n


# # 算法加速

import numpy as np
arg = [1, 2, 1]
print(np.root(args))
