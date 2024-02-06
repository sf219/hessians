import numpy as np
from scipy.linalg import eigh
from utils.utils_lpit import unifgrid

Q1 = np.random.rand(64, 64)
Q = Q1 @ Q1.T

L, _ = unifgrid(8)

E, U = eigh(L, Q, eigvals_only=False)

print(U)
print(E)
breakpoint()
