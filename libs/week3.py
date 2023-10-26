import numpy
import numpy as np
import libs.week1 as week1


def compute_deltas(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> np.array:
    """Computer how much the players could improve if they were to switch to the best response"""

    utility_row, utility_col = week1.ComputeValuesFor_TwoPlayer_NonZeroSumGame(matrix, -matrix, row_strategy, column_strategy)

    best_col_utility = -week1.BestResponse_Value_To_Row(matrix, row_strategy)

    # same as for the column player
    best_row_utility = -week1.BestResponse_Value_To_Column(matrix, column_strategy)

    return np.array([best_row_utility - utility_row, best_col_utility - utility_col])
def nash_conv(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    return sum(compute_deltas(matrix, row_strategy, column_strategy))

def compute_exploitability(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Compute exploitability for a zero-sum game"""
    return nash_conv(matrix, row_strategy, column_strategy)/2

def compute_epsilon(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Computes epsilon as defined for epsilon-Nash equilibrium"""
    return numpy.max(compute_deltas(matrix, row_strategy, column_strategy))
