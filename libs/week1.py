import numpy

def evaluate(matrix, row_strategy, column_strategy):
    result = 0
    for i in range(row_strategy.size):
        for j in range(column_strategy.size):
            result += matrix[i][j] * row_strategy[0][i] * column_strategy[j][0]
    return result

def best_response_value_row(matrix, row_strategy):
    return (row_strategy @ matrix).min()

def best_response_value_column(matrix, column_strategy):
    return (column_strategy.T @ matrix).min()

def evaluateForZeroSumGames():
    matrix1 = numpy.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    matrix2 = matrix1 * -1

    row_strategy = numpy.array([[0.1, 0.2, 0.7]])
    column_strategy = numpy.array([[0.3, 0.2, 0.5]]).transpose()

    player1Value = evaluate(matrix1, row_strategy, column_strategy)
    player2Value = evaluate(matrix2, row_strategy, column_strategy)
    return


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
