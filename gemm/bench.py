import numpy as np
import time

# 125.0000 GFLOPS for numpy
n = 1000
flop = 2 * n * n * n # 2 is for multi and sum
a = np.random.random((n, n))
b = np.random.random((n, n))

tic = time.monotonic()
a @ b
toc = time.monotonic()

print("{:.4f} GFLOPS".format(flop / (toc - tic) * 1e-9))