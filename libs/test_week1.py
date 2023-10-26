import libs.week1 as week1
import numpy as np
import pytest


def test_best_response_tests():
    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])

    column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()

    best_response_row_strategy = week1.BestResponseStrategyAgainstColumnPlayer(matrix, column_strategy)

    print(best_response_row_strategy)

def test_iterated():
    game_matrix = np.array([[3, 2, 5], [1, 4, 6]])
    simplified_matrix = week1.iterated_removal(game_matrix)
    assert simplified_matrix[0] == 2
    assert simplified_matrix[1] == 4

def test_week1():
    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    row_strategy = np.array([[0.1, 0.2, 0.7]])
    column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()

    rowValue, columnValue = week1.ComputeValuesFor_TwoPlayer_NonZeroSumGame(-matrix, matrix, row_strategy, column_strategy)
    assert rowValue == pytest.approx(0.08)

    br_value_row = week1.BestResponseValueAgainstRowPlayer(matrix=matrix, opponents_row_strategy=row_strategy)
    br_value_column = week1.BestResponseValueAgainstColumnPlayer(matrix=matrix, opponents_column_strategy=column_strategy)
    assert br_value_row == pytest.approx(-0.6)
    assert br_value_column == pytest.approx(-0.2)