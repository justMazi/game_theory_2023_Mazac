import numpy as np


def iterated_removal(matrix):
    num_players, num_strategies = matrix.shape

    while True:
        player_dominated = [False] * num_players
        for player in range(num_players):
            for strategy in range(num_strategies):
                if not player_dominated[player]:
                    for other_strategy in range(num_strategies):
                        if other_strategy != strategy and all(
                                matrix[player, i] >= matrix[player, other_strategy] for i in range(num_players)):
                            player_dominated[player] = True
                            matrix = np.delete(matrix, strategy, axis=1)
                            num_strategies -= 1
                            break
            if player_dominated[player]:
                break

        if not any(player_dominated):
            break
    return matrix


# my super duper reusable methods for future assignments

def ComputeValuesFor_TwoPlayer_NonZeroSumGame(row_player_utility_matrix, column_player_utility_matrix, row_strategy,column_strategy):
    """
    :param row_player_utility_matrix: Reward matrix for the row player.
    :param column_player_utility_matrix: Reward matrix for the column player.
    :param row_strategy: Probability distribution over actions of row player.
    :param column_strategy: Probability distribution over actions of column player.
    :return: Values for row and column players.
    """
    # probabilities of all action combinations
    prob_matrix = (column_strategy @ row_strategy).T

    # utilities of all combinations of actions multiplied by its probability
    row_utility_matrix = row_player_utility_matrix * prob_matrix
    column_utility_matrix = column_player_utility_matrix * prob_matrix

    # sum up to get value utilities
    row_value = np.sum(row_utility_matrix)
    column_value = np.sum(column_utility_matrix)

    return row_value, column_value

def BestResponse_Pure_Strategy_To_Row(matrix: np.array, row_strat: np.array) -> np.array:
    expected_payoffs = row_strat @ matrix
    max_payoff = np.argmax(expected_payoffs, axis=1)
    return CreatePureStrategy(len=matrix.shape[1], index=max_payoff).T

def BestResponse_Pure_Strategy_To_Column(matrix: np.array, column_strategy: np.array) -> np.array:
    expected_payoffs = matrix @ column_strategy
    max_payoff = np.argmax(expected_payoffs, axis=0)
    return CreatePureStrategy(len=matrix.shape[0], index=max_payoff)

def CreatePureStrategy(len, index):
    response = np.array([[1 if i==index else 0 for i in range(len)]])
    return response


def BestResponse_Value_To_Row(matrix: np.array, row_strategy: np.array) -> float:
    """Value of the row player when facing a best-responding column player in a zero-sum game"""
    bestResponseToRow = BestResponse_Pure_Strategy_To_Row(-matrix, row_strategy)
    p1_val, p2_val = ComputeValuesFor_TwoPlayer_NonZeroSumGame(matrix, -matrix, row_strategy, bestResponseToRow)
    return p1_val


def BestResponse_Value_To_Column(matrix: np.array, column_strategy: np.array) -> float:
    """Value of the column player when facing a best-responding row player in a zero-sum game"""
    bestResponseToCol = BestResponse_Pure_Strategy_To_Column(matrix, column_strategy)
    p1_val, p2_val = ComputeValuesFor_TwoPlayer_NonZeroSumGame(matrix, -matrix, bestResponseToCol, column_strategy)
    return p2_val
