import numpy as np
import matplotlib.pyplot as plt
import copy

class Node:
    def __init__(self, player, information_set, children=None, history=None):
        """
        Initializes a new instance of the Node class.

        Args:
            player (int): The player associated with the node.
            information_set (str): The information set associated with the node.
            children (dict, optional): The children nodes of the node. Defaults to None.
            history (str, optional): The history of the node. Defaults to None.

        Returns:
            None
        """
        self.player = player
        self.information_set = information_set 
        self.history = history or information_set
        self.children = children or {} 
    
    def get_actions(self):
        """
        Get the actions available from this node (keys of the children dictionary).

        Returns:
            list: A list of actions available from this node.
        """
        return self.children.keys()

    def is_terminal(self):
        """
        Check whether the node is a terminal one (meaning it has no children).

        Returns:
            bool: True if the node has no children, False otherwise.
        """
        return not self.children
    


class ExtensiveFormGameCalculator:
    def __init__(self, players: list, chance: list, root: Node, matrix: dict):
        """
        Initializes an instance of the ExtensiveFormGame class with the given parameters.

        Args:
            players (list): A list of players in the game.
            chance (list): A list of players representing chance events.
            root (Node): The root node of the game tree.
            matrix (dict): A dictionary representing the matrix of payoffs.

        Returns:
            None
        """
        self.players = players
        self.chance = chance
        self.root = root
        self.matrix = matrix
        self.infosets = self.prep_infosets()
        self.actions_in_infoset = self.prep_actions_for_all_infosets()

    def get_infosets(self) -> dict:
        """
        Returns a dictionary of information sets.

        :return: A dictionary where the keys are information sets and the values are sets of nodes.
        :rtype: dict
        """
        return self.infosets
    
    def get_nodes_in_infoset(self, infoset, player=None) -> set:
        """
        Args:
            infoset (str): The name of the information set.
            player (str, optional): The name of the player. Defaults to None.

        Returns:
            set: A set of nodes in the information set for the specified player, or all nodes in the information set if no player is specified.
        """
        if player is not None:
            return set(n for n in self.infosets[infoset] if n.player == player)
        return self.infosets[infoset] 
        
        
    def prep_infosets(self) -> dict:
        """
        Prepares and returns a dictionary of information sets
        """
        infosets = {}
        
        self.traverse_game_tree(self.root, infosets)
        return infosets
        
    def traverse_game_tree(self, node: Node, infosets: dict):
        """
        Args:
            node (Node): The starting node for the search.
            infosets (dict): The dictionary to populate with information sets.

        Returns:
            None
        """
        infosets.setdefault(node.information_set, set())
        infosets[node.information_set].add(node)
        for k, child in node.children.items():
            self.traverse_game_tree(child, infosets)
    
    def prep_actions_for_all_infosets(self):
        """
        Prepares and returns a dictionary of actions in each information set.

        Returns:
            dict: A dictionary where the keys are the information sets and the values are the actions in each information set.
        """
        actions_in_infoset = {}
        for infoset in self.infosets:
            actions_in_infoset[infoset] = next(iter(self.get_nodes_in_infoset(infoset))).actions()
        return actions_in_infoset
        
    def calculate_node_reach_probability(self, strategies, history):
        node = self.root
        prob = 1
        
        for a in history:
            if node.player in strategies:
                prob *= strategies[node.player][node.information_set][a]
            node = node.children[a]
        
        return prob
    
    def calculate_infoset_reach_probability(self, strategies, infoset, player=None):
        prob = 0
        for node in self.get_nodes_in_infoset(infoset, player):
            prob += self.calculate_node_reach_probability(strategies, node.history)
        return prob
    
    def get_node_by_history(self, history):
        node = self.root
        for a in history:
            node = node.children[a]
        return node
    
    def calculate_player_values(self, strategies, node: Node = None):
        # Calculate the player values for a given node in the game tree based on player strategies
        node = node or self.root
        
        return self.get_value_from_history(strategies, node)
        
    def get_value_from_history(self, strategies, node: Node):
        # Calculate the values for all players at a given node in the game tree based on player strategies
        if node.is_terminal():
            return self.matrix[node.history]
        
        player = node.player
        information_set = node.information_set
        
        values = {p: 0 for p in self.players}

        for action in node.get_actions():
            a_values = self.get_value_from_history(strategies, node.children[action])
            for p in self.players:
                values[p] += a_values[p] * strategies[player][information_set][action]

        return values


    def average_strategy(self, strategies):
        # Calculate the average strategy over all players and information sets
        player = next(iter(strategies[0]))
        average_strategy = {player: {}}
        for information_set in strategies[0][player].keys():
            average_strategy[player][information_set] = {}
            for action in strategies[0][player][information_set].keys():
                average_strategy[player][information_set][action] = 0
        
        # Use DFS to calculate average strategy
        self.average_strategy_traverse_dfs(self.root, strategies, average_strategy, player)
        
        # Normalize the average strategy
        for infoset in average_strategy[player].keys():
            total_sum = sum(average_strategy[player][infoset].values())
            number_of_actions = len(average_strategy[player][infoset].keys())
            for action in average_strategy[player][infoset]:
                if total_sum == 0:
                    average_strategy[player][infoset][action] = 1/number_of_actions
                else:
                    average_strategy[player][infoset][action] /= total_sum
        
        return average_strategy
    
    def average_strategy_traverse_dfs(self, node: Node, strategies, average_strategy, player):
        # Depth-first search to calculate average strategy
        if node.is_terminal():
            return
        
        # Skip nodes where the player does not play
        if node.player != player:
            for action in node.get_actions():
                self.average_strategy_traverse_dfs(node.children[action], strategies, average_strategy, player)
            return
        
        information_set = node.information_set
        # reach_prob = self.calculate_reach_probability(avg_strat, node.history)
        strat_reach_prob = [self.calculate_node_reach_probability(s, node.history) for s in strategies]
        
        for action in node.get_actions():
            for i, strat in enumerate(strategies):
                # Sum up (reach probability * probability of action) for each strategy
                average_strategy[player][information_set][action] += strat_reach_prob[i] * strat[player][information_set][action]

            self.average_strategy_traverse_dfs(node.children[action], strategies, average_strategy, player)
                
    def best_response(self, strategies, node: Node = None):
        # Find the best response strategy against given strategies
        assert len(strategies.keys()) == len(self.players) + len(self.chance) - 1
        
        player = next(p for p in self.players if p not in strategies)
        node = node or self.root
        
        best_response = {player: {}}
        self.get_best_response(strategies, node, best_response, player)
        
        return best_response
    
    def get_best_response(self, strategies, node: Node, best_response: dict, player, information_set_values: dict = {}):
        if node.is_terminal():
            return self.matrix[node.history][player]
        
        action_values = {}

        # Player's turn
        if node.player == player:
            information_set = node.information_set
            best_response[player][information_set] = {}
            
            nodes_in_information_set = self.get_nodes_in_infoset(information_set, player)
        
            for action in node.get_actions():
                action_values[action] = 0
                best_response[player][information_set][action] = 0
                for possible_node in nodes_in_information_set:
                    action_val = \
                        self.calculate_node_reach_probability(strategies, possible_node.history) *\
                        self.get_best_response(strategies, possible_node.children[action], best_response, player, information_set_values)
                    action_values[action] += action_val
            best_response[player][information_set][max(action_values, key=action_values.get)] = 1
            return max(action_values.values())
        
        # Not this player's turn
        else:
            for action in node.get_actions():
                action_values[action] = \
                    strategies[node.player][node.information_set][action] * \
                    self.get_best_response(strategies, node.children[action], best_response, player, information_set_values)

            return sum(action_values.values())
    
    def calculate_deltas(self, strategies):
        # Calculate the deltas for each player between best response utility and game values
        game_values = self.calculate_player_values(strategies)
        
        best_response_utility = {}
        for p in self.players:
            # Calculate br(pi_{-i}) = best response to -i
            without_p = {key: val for key, val in strategies.items() if key != p}
            br_p = self.best_response(without_p)
            without_p[p] = br_p[p]
            
            # Calculate u_i using br with fixed -i
            best_response_utility[p] = self.calculate_player_values(without_p)[p]
        
        deltas = {p: best_response_utility[p] - game_values[p] for p in self.players}
        return deltas
    
    def calculate_nash_conv(self, strategies) -> float:
        """
        Calculate the Nash convergence for a given set of strategies.

        Returns:
            float: The sum of deltas representing the Nash convergence.

        """
        return sum(self.calculate_deltas(strategies).values())
    
    def calculate_exploitability(self, strategies) -> float:
        return self.calculate_nash_conv(strategies) / len(self.players)
    
    def create_uniform_strategy_for_player(self, player):
        strat = {player: {}}
        all_infosets = self.get_infosets()
        for infoset, nodes in all_infosets.items():
            nodes = [n for n in nodes if n.player == player]
            if len(nodes) == 0:
                continue
            
            node = nodes[0]
            actions = node.actions()
            
            count = len(actions)
            strat[player][infoset] = {a: 1 / count for a in actions}
        return strat
    
    def get_uniform_strat_for_players(self):
        p1 = self.create_uniform_strategy_for_player(self.players[0])
        p2 = self.create_uniform_strategy_for_player(self.players[1])
        return p1, p2

    def self_play(self, chance_player_strategies=None, iterations=50):
        # Initialize chance player strategies with a uniform strategy if not provided
        if chance_player_strategies is None:
            chance_player_strategies = {
                chance_player: self.create_uniform_strategy_for_player(chance_player)[chance_player]
                for chance_player in self.chance
            }

        # Initialize uniform strategies for both players
        player1_strategies = [self.create_uniform_strategy_for_player(self.players[0])]
        player2_strategies = [self.create_uniform_strategy_for_player(self.players[1])]

        # List to store exploitabilities at each iteration
        exploitability_over_time = []

        # Run the self-play iterations
        for _ in range(iterations):
            # Get the latest strategies and incorporate chance player strategies
            latest_strategy_player1 = player1_strategies[-1]
            latest_strategy_player2 = player2_strategies[-1]
            latest_strategy_player1.update(chance_player_strategies)
            latest_strategy_player2.update(chance_player_strategies)

            # Find the best response strategies against the latest strategies
            best_response_to_player1 = self.best_response(latest_strategy_player1)
            best_response_to_player2 = self.best_response(latest_strategy_player2)
            best_response_to_player1.update(chance_player_strategies)
            best_response_to_player2.update(chance_player_strategies)

            # Append the best response strategies to the strategy lists
            player1_strategies.append(best_response_to_player2)
            player2_strategies.append(best_response_to_player1)

            # Calculate the average strategies over all iterations
            average_strategy_player1 = self.average_strategy(player1_strategies)[self.players[0]]
            average_strategy_player2 = self.average_strategy(player2_strategies)[self.players[1]]
            combined_strategies = {
                self.players[0]: average_strategy_player1,
                self.players[1]: average_strategy_player2,
            }
            combined_strategies.update(chance_player_strategies)

            # Calculate and store the exploitability for the current iteration
            current_exploitability = self.calculate_exploitability(combined_strategies)
            exploitability_over_time.append(current_exploitability)

        return player1_strategies, player2_strategies, exploitability_over_time





    def probability_of_reaching_state(self, strategies, start_history, target_history):
        # Calculate the probability of reaching target_history starting from start_history using given strategies
        current_node = self.get_node_by_history(start_history)

        if not target_history.startswith(start_history):
            # target_history cannot be reached from start_history
            return 0

        # Get the path from start_history to target_history
        path = target_history[len(start_history):]
        probability = 1

        for action in path:
            if current_node.player in strategies:
                player_strategy = strategies[current_node.player]
                information_set = current_node.information_set
                probability *= player_strategy[information_set][action]
            current_node = current_node.children[action]

        return probability


        
    def get_utility_of_infoset(self, strategies, infoset, player):
        # Calculate the utility in a specific information set for a given player using given strategies
        infoset_reach_prob = self.calculate_infoset_reach_probability(strategies, infoset)
        
        if infoset_reach_prob == 0:
            # Infoset cannot be reached
            return 0
        
        # Exclude the player's strategy to calculate the utility
        strategies_without_player = {key: val for key, val in strategies.items() if key != player}
        
        utility = 0
        terminal_nodes = [self.get_node_by_history(history) for history in self.matrix.keys()]
        
        # Iterate over all nodes in the information set
        for infoset_node in self.get_nodes_in_infoset(infoset, player):
            # Iterate over all terminal nodes in the tree
            for terminal_node in terminal_nodes:
                # Calculate the utility contribution for each pair of nodes
                utility += (
                    self.calculate_node_reach_probability(strategies_without_player, infoset_node.history) *
                    self.probability_of_reaching_state(strategies, infoset_node.history, terminal_node.history) *
                    self.matrix[terminal_node.history][player]
                )
        
        # Normalize by the probability of reaching the infoset
        utility /= infoset_reach_prob
        
        return utility


                

    ############# CFR SECTION ##############

    def counterfactual_regret_minimization(self, chance_strategies=None, iterations=50):
        # Initialize chance strategies for all chance players using a uniform strategy if not provided
        chance_strategies = chance_strategies or {c: self.create_uniform_strategy_for_player(c)[c] for c in self.chance}
        
        # Initialize regrets dictionary for each player
        regrets = {player: self.initialize_regrets(player) for player in self.players}
        
        # Initialize empty strategy lists for each player
        player_strategies = {player: [] for player in self.players}
        
        # List to store exploitabilities at each iteration
        exploitabilities = []
        
        # Run the CFR iterations
        for _ in range(iterations):
            # Update player strategies using regret matching
            for player in self.players:
                player_strategies[player].append(self.regret_matching(regrets[player], player))
            
            # Calculate exploitability from average strategies
            current_strategies = {}
            for player in self.players:
                current_strategies.update(self.average_strategy(player_strategies[player]))
            current_strategies.update(chance_strategies)
            
            exploitabilities.append(self.calculate_exploitability(current_strategies))
            
            # Update regrets using current strategies
            for player in self.players:
                self.update_regrets(regrets[player], current_strategies, player)
        
        return player_strategies, exploitabilities



    def initialize_regrets(self, player):
        regrets = {}
        for infoset in self.infosets:
            nodes_in_infoset = self.get_nodes_in_infoset(infoset, player)
            if not nodes_in_infoset:
                continue
            regrets[infoset] = {action: 0 for action in self.actions_in_infoset[infoset]}
        return regrets

    
        
    def regret_matching(self, regrets, player):
        """
        Computes a strategy for the specified player using regret matching.

        Args:
        - regrets (dict): Dictionary storing regrets for each information set and action pair.
        - player (str): The player for whom the strategy is being computed.

        Returns:
        - dict: A strategy dictionary for the player, where strategy[player][infoset][action] 
                represents the probability of choosing 'action' in 'infoset'.
        """
        strategy = {player: {}}
        
        # Iterate over each information set that the player has encountered
        for infoset in self.infosets:
            nodes_in_infoset = self.get_nodes_in_infoset(infoset, player)
            
            # Skip information sets where the player has no nodes
            if not nodes_in_infoset:
                continue
            
            strategy[player][infoset] = {}
            
            # Calculate R+ (non-negative regret) and total positive regret sum
            regrets_plus = {action: max(regrets[infoset][action], 0) for action in self.actions_in_infoset[infoset]}
            total_positive_regret = sum(regrets_plus.values())
            
            # Compute strategy probabilities
            for action in self.actions_in_infoset[infoset]:
                if total_positive_regret > 0:
                    strategy[player][infoset][action] = regrets_plus[action] / total_positive_regret
                else:
                    strategy[player][infoset][action] = 1 / len(self.actions_in_infoset[infoset])
        
        return strategy



    def update_regrets(self, regrets, strategies, player):
        """
        Updates regrets for the specified player based on current strategies.

        Args:
        - regrets (dict): Dictionary storing regrets for each information set and action pair.
        - strategies (dict): Dictionary containing strategies for all players.
        - player (str): The player whose regrets are being updated.

        Returns:
        - None
        """
        # Calculate pi_-i (strategies of all other players)
        without_player_strategies = {key: val for key, val in strategies.items() if key != player}

        # Iterate over each information set that the player has encountered
        for infoset in self.get_infosets():
            nodes_in_infoset = self.get_nodes_in_infoset(infoset, player)
            
            # Skip information sets where the player has no nodes
            if not nodes_in_infoset:
                continue
            
            # Iterate over each action in the current information set
            for action in self.actions_in_infoset[infoset]:
                # Create a copy of strategies where the player always plays action 'a' in infoset 'I'
                strategies_where_action_in_infoset = copy.deepcopy(strategies)
                for other_action in self.actions_in_infoset[infoset]:
                    strategies_where_action_in_infoset[player][infoset][other_action] = 0
                strategies_where_action_in_infoset[player][infoset][action] = 1
                
                # Calculate the regret R(I,a)
                regrets[infoset][action] += \
                    self.calculate_infoset_reach_probability(without_player_strategies, infoset) * \
                    ( \
                        self.get_utility_of_infoset(strategies_where_action_in_infoset, infoset, player) - \
                        self.get_utility_of_infoset(strategies, infoset, player) \
                    )
