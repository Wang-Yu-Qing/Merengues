import numpy as np
import time

m = 2048
n = 2048
k = 4056
flop = 2 * m * k * n # 2 is for multi and sum
a = np.random.random((m, n))
b = np.random.random((n, k))

tic = time.monotonic()
a @ b
toc = time.monotonic()

print("{:.4f} TFLOPS".format(flop / (toc - tic) * 1e-12))

"""
(n, n)
origin: n * n * n * 2 = 2 * n^3

b = 4

4 * (n / 4, n / 4)
now: (n / 4) * (n / 4) * (n / 4) * 2 * 4 = (2 / 4^2) * n^3

"""