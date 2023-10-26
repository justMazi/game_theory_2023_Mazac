import numpy


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
                            matrix = numpy.delete(matrix, strategy, axis=1)
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
    prob_matrix = column_strategy @ row_strategy

    # utilities of all combinations of actions multiplied by its probability
    row_utility_matrix = row_player_utility_matrix * prob_matrix
    column_utility_matrix = column_player_utility_matrix * prob_matrix

    # sum up to get value utilities
    row_value = numpy.sum(row_utility_matrix)
    column_value = numpy.sum(column_utility_matrix)

    return row_value, column_value


# best response in a zero-sum game, I am minimizing utility of row player, hence maximizing value of column player
def BestResponseValueAgainstRowPlayer(matrix, opponents_row_strategy):
    return (opponents_row_strategy @ matrix).min()

# best response in a zero-sum game, I am minimizing utility of column player, hence maximizing value of row player
def BestResponseValueAgainstColumnPlayer(matrix, opponents_column_strategy):
    return (opponents_column_strategy.T @ matrix).min()




def BestResponseStrategyAgainstColumnPlayer(matrix, opponents_column_strategy):
    index = BestResponseActionIndexAgainstColumnPlayer(matrix, opponents_column_strategy)
    vectorLength = len(opponents_column_strategy)
    return CreatePureStrategyVector(vectorLength, index)

def BestResponseStrategyAgainstRowPlayer(matrix, opponents_row_strategy):
    index = BestResponseActionIndexAgainstRowPlayer(matrix, opponents_row_strategy)
    vectorLength = len(opponents_row_strategy.T)
    return CreatePureStrategyVector(vectorLength, index)

def CreatePureStrategyVector(vectorLength, index):
    return numpy.array([[1 if i == index else 0 for i in range(vectorLength)]])




def BestResponseActionIndexAgainstColumnPlayer(matrix, opponents_column_strategy):
    """
    :return: Index of the best response action
    """
    return numpy.argmin(matrix @ opponents_column_strategy)

def BestResponseActionIndexAgainstRowPlayer(matrix, opponents_row_strategy):
    """
    :return: Index of the best response action
    """
    res = matrix @ opponents_row_strategy.T
    return numpy.argmin(res)