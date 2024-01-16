import numpy as np
from numba import njit
import numba
from utils.utils_lpit import matlab_round
import io
import math

@njit("uint8[:, :]()")
def get_table():
    table = np.array([
    [0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 4, 8, 4, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 5, 10, 5, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 6, 12, 6, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 7, 14, 7, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [0, 8, 18, 10, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
    [0, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], 
    [0, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1], 
    [1, 1, 5, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [1, 2, 8, 6, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [1, 3, 10, 7, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 4, 13, 9, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 5, 16, 11, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0], 
    [1, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0], 
    [1, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1], 
    [1, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0], 
    [1, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], 
    [1, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0], 
    [2, 1, 6, 5, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [2, 2, 10, 8, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [2, 3, 13, 10, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 ], 
    [2, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1], 
    [2, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0], 
    [2, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1], 
    [2, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0], 
    [2, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1], 
    [2, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0], 
    [2, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1], 
    [3, 1, 7, 6, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [3, 2, 11, 9, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
    [3, 3, 14, 11, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0 ], 
    [3, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], 
    [3, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1], 
    [3, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0], 
    [3, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1], 
    [3, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0], 
    [3, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1], 
    [3, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0], 
    [4, 1, 7, 6, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [4, 2, 12, 10, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [4, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1], 
    [4, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], 
    [4, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], 
    [4, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0], 
    [4, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1], 
    [4, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0], 
    [4, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1], 
    [4, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0], 
    [5, 1, 8, 7, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [5, 2, 12, 10, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0 ], 
    [5, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1], 
    [5, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0], 
    [5, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1], 
    [5, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0], 
    [5, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], 
    [5, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0], 
    [5, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1], 
    [5, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0], 
    [6, 1, 8, 7, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [6, 2, 13, 11, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [6, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1], 
    [6, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0], 
    [6, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1], 
    [6, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], 
    [6, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1], 
    [6, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], 
    [6, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1], 
    [6, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0], 
    [7, 1, 9, 8, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [7, 2, 13, 11, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0 ], 
    [7, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
    [7, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0], 
    [7, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1], 
    [7, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0], 
    [7, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1], 
    [7, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0], 
    [7, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], 
    [7, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0], 
    [8, 1, 9, 8, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [8, 2, 17, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
    [8, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], 
    [8, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], 
    [8, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1], 
    [8, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], 
    [8, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], 
    [8, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0], 
    [8, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], 
    [8, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], 
    [9, 1, 10,  9, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [9, 2, 18, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
    [9, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
    [9, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1], 
    [9, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0], 
    [9, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1], 
    [9, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0], 
    [9, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1], 
    [9, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0], 
    [9, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], 
    [10, 1, 10,  9, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ], 
    [10, 2, 18, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0], 
    [10, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1], 
    [10, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0], 
    [10, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1], 
    [10, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0], 
    [10, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1], 
    [10, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0], 
    [10, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], 
    [10, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0], 
    [11, 1, 10,  9, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ], 
    [11, 2, 18, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1], 
    [11, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0], 
    [11, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1], 
    [11, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0], 
    [11, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1], 
    [11, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0], 
    [11, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1], 
    [11, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0], 
    [11, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1], 
    [12, 1, 11, 10, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 ], 
    [12, 2, 18, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0], 
    [12, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], 
    [12, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0], 
    [12, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], 
    [12, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], 
    [12, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], 
    [12, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
    [12, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1], 
    [12, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0], 
    [13, 1, 12, 11, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0 ], 
    [13, 2, 18, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1], 
    [13, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], 
    [13, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1], 
    [13, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0], 
    [13, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 
    [13, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0], 
    [13, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], 
    [13, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0], 
    [13, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], 
    [14, 1, 13, 12, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], 
    [14, 2, 18, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0], 
    [14, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], 
    [14, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0], 
    [14, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], 
    [14, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 
    [14, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1], 
    [14, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0], 
    [14, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1], 
    [14, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0], 
    [15, 1, 17, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1], 
    [15, 2, 18, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0], 
    [15, 3, 19, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
    [15, 4, 20, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 
    [15, 5, 21, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1], 
    [15, 6, 22, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], 
    [15, 7, 23, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], 
    [15, 8, 24, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 
    [15, 9, 25, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
    [15, 10, 26, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]).astype(np.uint8)
    #[0, 0, 4, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    #[15, 0, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]]).astype(np.uint8)
    return table

huff_tab = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [3, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [4, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [5, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [6, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [7, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [8, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [9, 1, 1, 1, 1, 1, 1, 1, 1, 0]]).astype(int)

@numba.njit("uint8[:](int64)")
def int2bin(x):
    n = int(np.floor(np.log2(np.abs(x)))+1)
    b = np.zeros(n+1).astype(np.uint8)
    b[0] = -(np.sign(x)-1)*0.5*(x!=0)
    x = np.abs(x)
    idx = 2**(n-1)
    for j in range(n):
        tmp = x - np.sign(x)*idx*(x!=0)
        nn_zero_tmp = tmp>=0
        nn_x = x!=0
        logicals = (nn_zero_tmp and nn_x)
        #print(logicals)
        b[j+1] = int(logicals)
        x = tmp
        idx = (idx>>1)
    return b

@numba.njit("int64(uint8[:])")
def bin2int(b):
    #if(b.size == 0):
    #    return b
    out = 0
    n = len(b)
    mult = 2**(n-1)
    for j in range(n):
        out += mult*b[j]
        mult = (mult>>1)
    return out

def dpcm(x, a):
    m, nx = x.shape
    if hasattr(a, "__len__"):
        p, na = a.shape
    else:
        p = 1
        na = 1
    r = np.zeros_like(x)
    xtilde = np.zeros_like(x)
    r[0:p] = matlab_round(x[0:p])
    xtilde[0:p] = r[0:p]
    for t in range(p, m, 1):
        xhat = np.sum(a*np.flip(xtilde[t-p:t]))
        r[t] = matlab_round(x[t] - xhat)
        xtilde[t] = r[t] + xhat 
    return r, xtilde   

# numbers, divisor = N, M
def rice_golomb_encode(numbers: list[int], divisor: int) -> str:
    bin_len = int(np.floor(np.log2(divisor)))
    inv_divisor = 2 ** (bin_len + 1) - divisor

    code = ''
    for dividend in numbers:
        quotient = dividend // divisor
        code += '1' * quotient
        code += '0'

        reminder = dividend % divisor
        if reminder < inv_divisor:
            code += format(reminder, f'0{bin_len}b')
        else:
            bin_len += 1
            code += format(reminder + inv_divisor, f'0{bin_len}b')
    return code

@njit
def add_zeros_beg(x):
    b = np.zeros(16*len(x)).astype(np.uint8)
    last = 0
    while (x[0] == 0):
        last+=2
        x = x[1:]
    return x, b, last

@njit("uint32[:](int64[:])")
def find_pos_ones(x):
    pos = (x != 0)
    out = np.arange(len(x)).astype(np.uint32)
    out = out[pos]
    return out

def jacenc(x):
    b = np.zeros(32*len(x)).astype(np.uint8)
    last = 0
    ix = find_pos_ones(x)
    Nx = len(ix)
    pencil_15 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]).astype(np.uint8)
    pencil_999 = np.array([1, 0, 1, 0]).astype(np.uint8)
    table = get_table()
    prev = -1
    for n in range(Nx):
        run = ix[n] - prev - 1
        prev = ix[n]
        while run > 15:
            prefix = pencil_15
            b[last:last+len(prefix)] = prefix
            last = last + len(prefix)
            run = run - 15
        if x[ix[n]] == 999:
            prefix = pencil_999
            b[last:last+len(prefix)] = prefix
            last = last + len(prefix)
            break
        else:
            tmp1 = int2bin(x[ix[n]])
            cat = len(tmp1)-1
            row=run*10+cat-1
            inter = table[row, 3]
            prefix = table[row, 4:inter+4]
            if tmp1[0]==1:
                tmp = negate_x(tmp1, cat+1)
            else:
                tmp=tmp1
            b[last:last+len(prefix)] = prefix
            last = last + len(prefix)
            b[last:last+cat] = tmp[1:cat+1]
            last = last + cat
    b = b[0:last]
    return b

@njit
def x_tmp_matching(x, tmp, tab):
    if (x == 1):
        tmp_1 = tab
    else:
        tmp_1 = (np.logical_not(tab)).astype(np.uint8)
        #tmp_1 = negate_x(tab, len(tab)).astype(np.uint8)
    out = tmp*tmp_1
    out_2 = np.sum(out)
    return out, out_2

@njit
def find_pos_first_one(input):
    n = len(input)
    for j in range(n):
        if(input[j] == 1):
            break
    return int(j)

@njit
def negate_x(x, cat):
    tmp = np.ones(cat) - x
    return tmp.astype(np.uint8) 

@njit
def reduce_beginning(x):
    N = len(x)
    y = 1002*np.ones((N)).astype(np.int64)
    last = 0
    while (np.sum(x[0:16] == 0)==16):
        y[last] = 0
        last += 1
        x = x[16:]
    return x, y, last

def jacdec(x):
    x = x.astype(np.uint8)
    x, y, last = reduce_beginning(x)
    p = 162
    i=0
    N = len(x)
    d=4
    base_tmp = np.ones((p,))
    tmp = base_tmp.astype(np.uint8)
    tep = i
    table = get_table()
    while i<N:
        tab = table[0:p, d]   
        tmp, sum_tmp = x_tmp_matching(x[i], tmp, tab)
        if sum_tmp == 1:
            d = 4
            row = find_pos_first_one(tmp)
            run = table[row, 0]
            cat = table[row, 1]
            pos = table[row, 3]
            i = tep + pos -1

            tmp = base_tmp
            if row==161:
                if(cat !=0 ):
                    num = bin2int(x[i+1:i+cat+1])
                else:
                    num = x[i+1:i+cat+1]
               # i = i + cat
               # i = i+1
               # tmp_1 = np.zeros((run,))    
               # y = np.hstack((y, tmp_1))
            elif row==160:
                num=999
            else:
                if(cat !=0 ):
                    num = bin2int(x[i+1:i+cat+1])
                else:
                    num = x[i+1:i+cat+1]
                if (not ( num >= (2**(cat-1)) and num < (2**cat))):
                    num = -1*bin2int(negate_x(x[i+1:i+cat+1], cat))
            #print(row, run, cat, num)
            i = i + cat
            y[last:last+run] = np.zeros(run).astype(np.int64)
            last += run
            if (cat != 0 or num == 999):
                y[last] = int(num)
                last += 1
            tep=i+1
        else:
            d = d + 1
    i = i + 1
    y = y[0:last]
    return y 

def jdcenc(x):
    if x==0:
        b = np.array([0, 0])
        return b
    else:
        c = int(np.floor(np.log2(np.abs(x)))+1)
    b = huff_tab[c, 1:huff_tab[c, 0]+1]
    tmp = int2bin(x[0])
    #tmp_2 = int2bin(x)
    #breakpoint()
    if(tmp[0]==0):
        b = np.hstack((b, tmp[1:c+1]))
    elif tmp[0]==1:
        rem = tmp[1:c+1]
        b = np.hstack((b, np.ones_like(rem)-rem))
    return b




class BitWriter:

  def __init__(self):
    self.sink = io.BytesIO()
    self.buffer = 0
    self.offset = 0
  
  def write(self, val: int, n: int) -> None:
    """Writes lower `n` bits of `val`."""
    assert 0 <= self.offset <= 32, self.offset
    assert 0 <= n <= 32, n

    room = 32 - self.offset
    if n <= room:
      val &= (1 << n) - 1
      self.buffer |= (val << self.offset)
      self.offset += n
      return

    self.write(val, room)
    val >>= room
    n -= room

    assert self.offset == 32, self.offset
    self.sink.write(int(self.buffer).to_bytes(4, "little", signed=False))  # 4 bytes.
    self.buffer = 0
    self.offset = 0

    assert 0 < n <= 32, n
    self.write(val, n)
    
  def write_run_length(self, n: int) -> None:
    assert 0 <= n <= 31
    self.write(1 << n, n + 1)

  def finalize(self) -> bytes:
    self.write(1, 1)  # End-of-sequence marker.
    n = (self.offset + 7) // 8
    self.sink.write(int(self.buffer).to_bytes(n, "little", signed=False))
    output = self.sink.getvalue()
    self.sink.close()
    return output

def rle(Input, N=8):
    L = len(Input)
    Output = []
    j = 0
    k = 0
    i = 0
    
    while i < 2 * L:
        comp = 1
        while j < L:
            if j == L - 1:
                break
            
            if(comp == N):
                break

            if Input[j] == Input[j + 1]:
                comp += 1
            else:
                break
            j += 1
        
        Output.append(comp)
        Output.append(Input[j])
        
        if j == L - 1 and Input[j - 1] == Input[j]:
            break
        
        i += 1
        k += 2
        j += 1
        
        if j == L:
            if L % 2 == 0:
                Output.append(1)
                Output.append(Input[j - 1])
            else:
                Output.append(1)
                Output.append(Input[j])
            break
    
    return Output


class BitReader:

  def __init__(self, source: bytes):
    self.source = io.BytesIO(source)
    self.buffer = 0
    self.remain = 0

  def _read_from_source(self) -> None:
    read = self.source.read(4)
    assert read, "Read past the end of the source."
    assert len(read) <= 4, read

    self.buffer = int.from_bytes(read, "little", signed=False)
    self.remain = len(read) * 8

  def read(self, n: int) -> int:
    assert 0 <= n <= 32, n
    assert 0 <= self.remain <= 32, self.remain
    if n <= self.remain:
      val = self.buffer & ((1 << n) - 1)
      self.buffer >>= n
      self.remain -= n
      return val
    
    val = self.buffer
    offset = self.remain
    n -= self.remain

    self._read_from_source()
    val |= self.read(n) << offset
    return val

  def read_run_length(self) -> int:
    # Maximum is 32.
    if self.buffer != 0:
      n = (self.buffer ^ (self.buffer - 1)).bit_length()
      assert n != 0, n
      assert n <= self.remain, (self.buffer, self.remain)
      self.buffer >>= n
      self.remain -= n
      return n - 1

    n = self.remain
    self._read_from_source()
    return n + self.read_run_length()


def rlgr(x: np.ndarray, L=6) -> bytes:
  """Encodes with Adaptive Run Length Golomb Rice coding.

  Args:
    x: An array of signed integers to be coded.

  Returns:
    A Python `bytes`.
  """
  assert x.dtype == np.int32, x.dtype
  x = np.ravel(x)
  assert np.all(x <= ((1 << 30) - 1))
  assert np.all(-(1 << 30) <= x)

  sink = BitWriter()

  # Constants.
  L = L #originally this was 4
  U0 = 3
  D0 = 1
  U1 = 2
  D1 = 1
  quotientMax = 24

  # Initialize state.
  k_P = 0
  k_RP = 10 * L

  # Preprocess data from signed to unsigned.
  z = x * 2
  z[z < 0] += 1
  z = np.abs(z)

  N = len(z)
  n = 0
  while n < N:
    k = k_P // L
    k_RP = min(k_RP, 31 * L)
    k_R = k_RP // L

    u = z[n]  # Symbol to encode.

    if k != 0:  # Encode zero runs.
      m = 1 << k  # m = 2**k = expected length of run of zeros

      # Count the run length of zeros, up to m.
      ahead = z[n:n + m]
      zero_count = np.argmax(ahead != 0)  # np.argmax returns the _first_ index.
      if ahead[zero_count] == 0:  # In case (ahead == 0).all() is true.
        zero_count = len(ahead)

      n += zero_count
      if zero_count == len(ahead):
        # Found a complete run of zeros.
        # Write a 0 to denote the run was a complete one.
        sink.write(0, 1)

        # Adapt k.
        k_P += U1
        continue

      # Found a partial run of zeros (length < m).
      # Write a 1 to denote the run was a partial one, and the decoder needs
      # to read k bits to extract the actual run length.
      sink.write(1, 1)
      sink.write(zero_count, k)

      # The next symbol is encoded as z[n] - 1 instead of z[n].
      assert z[n] != 0
      u = z[n] - 1

    # Output GR code for symbol u.
    # bits = bits + gr(u,k_R)
    assert 0 <= u, u
    quotient = u >> k_R  # `quotient` is run-length encoded.
    if quotient < quotientMax:
      sink.write_run_length(quotient)
      sink.write(u, k_R)
    else:
      assert int(u).bit_length() <= 31, (u, u.bit_length())
      sink.write_run_length(quotientMax)
      sink.write(u, 31)

    # Adapt k_R.
    if quotient == 0:
      k_RP = max(0, k_RP - 2)
    elif quotient > 1:
      k_RP += quotient + 1

    # Adapt k.
    if k == 0 and u == 0:
      k_P += U0
    else:  # k > 0 or u > 0
      k_P = max(0, k_P - D0)

    n += 1

  output = sink.finalize()
  return output


def irlgr(source: bytes, N: int) -> np.ndarray:
  """IRLGR decodes bitStream into integers using Adaptive Run Length Golomb Rice.

  Args:
    source: A Python `bytes`.
    N: Number of symbols to decode.

  Returns:
    An array of decoded signed integers.
  """
  # Constants.
  L = 4
  U0 = 3
  D0 = 1
  U1 = 2
  D1 = 1
  quotientMax = 24

  source = BitReader(source)

  # Initialize state.
  k_P = 0
  k_RP = 10 * L

  # Allocate space for decoded unsigned integers.
  output = np.zeros(N, np.int32)

  # Process data one sample at a time (time consuming in Matlab).
  n = 0
  while n < N:
    k = k_P // L
    k_RP = min([k_RP, 31 * L])
    k_R = k_RP // L

    if k != 0:
      is_complete = (source.read(1) == 0)
      if is_complete:
        zero_count = 1 << k  # 2**k = expected length of run of zeros
        output[n:n + zero_count] = 0
        n += zero_count

        # Adapt k.
        k_P += U1
        continue

      # A partial run was encoded.
      zero_count = source.read(k)
      output[n:n + zero_count] = 0
      n += zero_count

    quotient = source.read_run_length()
    if quotient < quotientMax:
      u = (quotient << k_R) + source.read(k_R)
    else:
      u = source.read(31)
      quotient = u >> k_R

    # Adapt k_R.
    if quotient == 0:
      k_RP = max(0, k_RP - 2)
    elif quotient > 1:
      k_RP += quotient + 1

    # Adapt k.
    if k == 0 and u == 0:
      k_P += U0
    else:  # k > 0 or u > 0
      k_P = max(0, k_P - D0)

    output[n] = u if k == 0 else u + 1
    n += 1

  # Postprocess data from unsigned to signed.
  is_negative = (output % 2 == 1)
  output = ((output + 1) // 2) * np.where(is_negative, -1, 1)
  return output


def rlgr_test(n):
  rng = np.random.default_rng()
  x = rng.laplace(scale=1e+5, size=n)
  x = np.round(x).astype(np.int32)

  xhat = irlgr(rlgr(x), n)
  np.testing.assert_array_equal(x, xhat)
