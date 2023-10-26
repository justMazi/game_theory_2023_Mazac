from typing import List, Optional
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import week1 as week1


def verify_support_one_side(matrix: np.array, support_row: List, support_col: List) -> Optional[List]:
    """Tries to see whether the column player can mix their strategies in the support so that the values of the row player are best-responding"""
    submatrix = matrix[support_row][:, support_col]
    result = VerifyMatrix(submatrix)
    if result.success:
        return result.x[1:]
    return None
    num_rows, num_cols = submatrix.shape

    # 1*-U_1 1*p_1 0*p_2 = 0
    # 1*-U_2 2*p_1 1*p_2 = 0
    # 0 1*p_1 1*p_2 = 1

    A_eq = np.hstack([np.array([[1] * num_rows]).T, submatrix])
    A_eq = np.vstack([A_eq, [0] + [1] * (num_cols)])

    # total utility + sum of prob[i] * utility[i] = 0, sum of prob[i] = 1
    b_eq = [0] * (num_rows) + [1]
    # max utility
    c = [1] + [0] * num_cols

    # bounds for utility and probabilities
    bounds = [(None, None)] + [(0, 1) for _ in range(num_cols)]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if result.success:
        return result.x[1:]

    return None


def VerifyMatrix(submatrix: np.array):
    num_rows, num_cols = submatrix.shape
    A_eq = np.hstack([np.array([[1]*num_rows]).T, submatrix])
    A_eq = np.vstack([A_eq, [0]+[1]*(num_cols)])

    b_eq = [0] * (num_rows) + [1]

    c = [1] + [0] * num_cols

    bounds = [(None, None)] + [(0,1) for _ in range(num_cols)]

    return linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

def GetAllSupports(n_actions: int, m_actions: int):
    actions1 = list(range(n_actions))
    actions2 = list(range(m_actions))
    subset_pairs = []
    for i in range(1,1 << len(actions1)):
        subset1 = [actions1[j] for j in range(len(actions1)) if (i & (1 << j)) > 0]

        for j in range(1,1 << len(actions2)):
            subset2 = [actions2[k] for k in range(len(actions2)) if (j & (1 << k)) > 0]

            subset_pairs.append((subset1, subset2))
    return subset_pairs


def PlotGraph(matrix: np.array, step_size: float):
    # steps = np.linspace(0, 1, n)
    steps = np.arange(0, 1 + step_size, step_size)
    vals = []
    for step in steps:
        new_strat = np.array([[step, 1 - step]])
        br = week1.BestResponse_Pure_Strategy_To_Row(matrix, new_strat)
        # TODO: weird graph
        p1_val, p2_val = week1.ComputeValuesFor_TwoPlayer_NonZeroSumGame(matrix, -matrix, new_strat, br)
        vals.append(p1_val)

    plt.scatter(steps, vals, s=10)
    plt.show()



matrix = np.array([[2, 0, 0.8],
                   [-1, 1, -0.5]])
strat_col = np.array([[0.2, 0.3, 0.5]])
strat_row = np.array([[0.4, 0.6]])

PlotGraph(matrix, step_size=0.01)