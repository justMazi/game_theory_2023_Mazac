import numpy as np
import numpy.typing as npt
import pytest

class NormalFormGameCalculator:

    # non zero sum game constructor
    def __init__(self,
                 row_player_utility_matrix: npt.NDArray[np.float64],
                 column_player_utility_matrix: npt.NDArray[np.float64]) -> None:

        self.row_player_utility_matrix = row_player_utility_matrix
        self.column_player_utility_matrix = column_player_utility_matrix

        # if col player utility is not provided, we consider this a zero sum game
        if(column_player_utility_matrix is None):
            self.column_player_utility_matrix = -row_player_utility_matrix


    # calculation section

    def calculate_utilities(self,
                            row_player_strategy: npt.NDArray[np.float64],
                            column_player_strategy: npt.NDArray[np.float64]) -> [np.float64, np.float64]:


        action_probabilities = row_player_strategy @ column_player_strategy
        assert action_probabilities.sum() == pytest.approx(1)

        row_player_utility = action_probabilities * self.row_player_utility_matrix
        column_player_utility = action_probabilities * self.column_player_utility_matrix

        return row_player_utility.sum(), column_player_utility.sum()

    def get_best_response_strategy_against_row_player(self,
                                                  row_player_strategy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        assert row_player_strategy.sum() == pytest.approx(1)
        length = self.column_player_utility_matrix.shape[1]
        array_of_zeros = np.zeros(length)
        utilities =  self.column_player_utility_matrix.T @ row_player_strategy
        index = np.argmax(utilities, axis=0)
        array_of_zeros[index] = 1
        reshaped = np.reshape(array_of_zeros, (1, length))
        return (reshaped)
    
    def get_best_response_strategy_against_column_player(self,
                                                  column_player_strategy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        
        assert column_player_strategy.sum() == pytest.approx(1)
        length = self.row_player_utility_matrix.shape[0]
        array_of_zeros = np.zeros(length)
        utilities = column_player_strategy @ self.row_player_utility_matrix.T
        index = np.argmax(utilities, axis=1)
        array_of_zeros[index] = 1

        reshaped = np.reshape(array_of_zeros, (length, 1))
        return (reshaped)

    # jak moc si hrac prilepsi kdyz switchne na best response
    def incentive_to_deviate(self,
                            row_player_strategy: npt.NDArray[np.float64],
                            column_player_strategy: npt.NDArray[np.float64]) -> [np.float64, np.float64]:
        
        row_utility, column_utility = self.calculate_utilities(row_player_strategy, column_player_strategy)

        best_row = self.get_best_response_strategy_against_column_player(column_player_strategy)
        best_column = self.get_best_response_strategy_against_row_player(row_player_strategy)

        _, best_column_utility = self.calculate_utilities(row_player_strategy, best_column)
        best_row_utility, _ = self.calculate_utilities(best_row, column_player_strategy)

        return (best_column_utility - column_utility, best_row_utility - row_utility)

    def nash_conv(self,
                    row_player_strategy: npt.NDArray[np.float64],
                    column_player_strategy: npt.NDArray[np.float64]) -> np.float64:
        return np.sum(self.incentive_to_deviate(row_player_strategy, column_player_strategy))

    def get_exploitability(self,
                    row_player_strategy: npt.NDArray[np.float64],
                    column_player_strategy: npt.NDArray[np.float64]) -> np.float64:
        return self.nash_conv(row_player_strategy, column_player_strategy) / 2

    def get_epsilon_for_minimum_nash_equilibrium(self,
                            row_player_strategy: npt.NDArray[np.float64],
                            column_player_strategy: npt.NDArray[np.float64]) -> np.float64:
        return np.max(self.incentive_to_deviate(row_player_strategy, column_player_strategy))




    def play(self, num_of_iterations: int, best_respond_to_averaged_strat: bool):

        row_length = self.row_player_utility_matrix.shape[0]
        row_strategy = np.zeros(row_length)
        row_strategy[0] = 1
        row_strategy = np.reshape(row_strategy, (row_length, 1))

        
        column_length = self.column_player_utility_matrix.shape[1]
        column_strategy = np.zeros(column_length)
        column_strategy[0] = 1
        column_strategy = np.reshape(column_strategy, (1, column_length))

        row_player_actions = []
        col_player_actions = []

        exploitabilities = []

        row_player_actions.append(row_strategy)
        col_player_actions.append(column_strategy)

        for i in range(num_of_iterations):

            col_strat, row_strat = None, None

            col_strat = col_player_actions[-1]
            row_strat = row_player_actions[-1]

            if best_respond_to_averaged_strat:
                
                col_strat = np.average(col_player_actions, axis=0)
                row_strat = np.average(row_player_actions, axis=0)

            best_row_strat = self.get_best_response_strategy_against_column_player(col_strat)
            best_col_strat = self.get_best_response_strategy_against_row_player(row_strat)

            average_row = np.mean(row_player_actions, axis=0)
            average_col = np.mean(col_player_actions, axis=0)

            row_player_actions.append(best_row_strat)
            col_player_actions.append(best_col_strat)

            exploitability = self.get_exploitability(average_row, average_col)
            exploitabilities.append(exploitability)



        return exploitabilities, row_player_actions, col_player_actions, np.average(col_player_actions, axis=0), np.average(row_player_actions, axis=0)



    def regret_minimization(self, iterations: int, use_average_strat_exploitation: bool):
        # Prepare arrays to store regrets
        regrets_row = np.zeros(self.row_player_utility_matrix.shape[0])
        regrets_col = np.zeros(self.column_player_utility_matrix.shape[1])
        
        row_strategies, row_reward_vectors = [], []
        col_strategies, col_reward_vectors = [], []
        row_regrets, col_regrets = [], []
        
        cumulative_reward_row, cumulative_reward_col = 0, 0

        exploitabilities = []

        for i in range(iterations):
            # prepare new strategies 
            new_row_strat = self.match_regrets(regrets_row)
            new_col_strat = self.match_regrets(regrets_col)
            
            row_strategies.append(new_row_strat)
            col_strategies.append(new_col_strat)

            exploitability = float

            row_length = self.row_player_utility_matrix.shape[0]
            col_length = self.column_player_utility_matrix.shape[1]

            if use_average_strat_exploitation:

                a = np.average(row_strategies, axis=0)
                average_row = np.reshape(a, (row_length, 1))

                b = np.average(col_strategies, axis=0)
                average_col = np.reshape(b, (1, col_length))
                exploitability = self.get_exploitability(average_row, average_col)

            else:
                row = np.reshape(new_row_strat, (row_length, 1))
                col = np.reshape(new_col_strat, (1, col_length))
                exploitability = self.get_exploitability(row, col)


            exploitabilities.append(exploitability)


            # receive reward vector
            reward_row = self.reward_vector_row(self.row_player_utility_matrix, new_col_strat)
            row_reward_vectors.append(reward_row)
            reward_col = self.reward_vector_col(self.column_player_utility_matrix, new_row_strat)
            col_reward_vectors.append(reward_col)
            
            
            # update cumulative reward
            current_reward_row = np.sum(new_row_strat * reward_row)
            cumulative_reward_row += current_reward_row
            current_reward_col = np.sum(new_col_strat * reward_col)
            cumulative_reward_col += current_reward_col
            
            # update regrets
            self.update_regrets(regrets_row, reward_row, current_reward_row)
            self.update_regrets(regrets_col, reward_col, current_reward_col)
            

            row_regrets.append(reward_row - cumulative_reward_row)
            col_regrets.append(reward_col - cumulative_reward_col)
            
        return exploitabilities, np.average(row_strategies, axis=0), np.average(col_strategies, axis=0)
            

    def reward_vector_row(self, matrix: np.array, col_strategy: np.array) -> np.array:
        return np.dot(matrix, col_strategy)

    def reward_vector_col(self, matrix: np.array, row_strat: np.array) -> np.array:
        return np.dot(row_strat, matrix)

    def match_regrets(self, regrets: np.array):
        positive_regrets = np.maximum(regrets, 0)
        sum_of_positive = np.sum(positive_regrets)
        
        if sum_of_positive == 0:
            length = len(regrets)
            return np.array([1/length for _ in range(length)])
        
        return positive_regrets / sum_of_positive

    def update_regrets(self, regrets: np.array, rewards: np.array, current_strat_reward: float):
        regrets += rewards - current_strat_reward