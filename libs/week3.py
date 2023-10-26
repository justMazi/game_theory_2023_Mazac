import numpy
import numpy as np
import libs.week1 as week1

#?????
def compute_deltas(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> np.array:
    """Computer how much the players could improve if they were to switch to the best response"""
    br_col_strat = week1.BestResponseStrategyAgainstRowPlayer(matrix, opponents_row_strategy=row_strategy)
    br_row_strat = week1.BestResponseStrategyAgainstColumnPlayer(matrix, opponents_column_strategy=column_strategy)

    rowPlayerValue, columnPlayerValue = week1.ComputeValuesFor_TwoPlayer_NonZeroSumGame(matrix, -matrix, row_strategy, column_strategy)

    brRowValue, brColValue = week1.ComputeValuesFor_TwoPlayer_NonZeroSumGame(matrix, -matrix, br_col_strat, br_row_strat)

    return np.array([ brRowValue - rowPlayerValue, brColValue - columnPlayerValue ])
def nash_conv(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    return sum(compute_deltas(matrix, row_strategy, column_strategy))

def compute_exploitability(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Compute exploitability for a zero-sum game"""
    return nash_conv(matrix, row_strategy, column_strategy)/2

def compute_epsilon(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Computes epsilon as defined for epsilon-Nash equilibrium"""
    return numpy.max(compute_deltas(matrix, row_strategy, column_strategy))


def Play(matrix: np.array):
    convergences = []
    actions1 = []
    actions2 = []
    actions1.append(1)
    actions2.append(2)

    numOfIterations = 10

    for i in range(numOfIterations):

        newAction1 = week1.BestResponseActionIndexAgainstColumnPlayer(matrix, actions2[-1])
        newAction2 = week1.CreatePureStrategyVector(3, actions1[-1])

        actions1.append(newAction2)
        actions2.append(newAction1)

    print(actions1)
    print(actions2)


matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
Play(matrix)