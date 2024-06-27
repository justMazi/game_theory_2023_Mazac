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
    
    def is_terminal(self):
        """
        Check whether the node is a terminal one (meaning it has no children).

        Returns:
            bool: True if the node has no children, False otherwise.
        """
        return not self.children
    
    def actions(self):
        """
        Get the actions available from this node (keys of the children dictionary).

        Returns:
            list: A list of actions available from this node.
        """
        return self.children.keys()

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

        for action in node.actions():
            a_values = self.get_value_from_history(strategies, node.children[action])
            for p in self.players:
                values[p] += a_values[p] * strategies[player][information_set][action]

        return values


    def average_strategy(self, strategies):
        # Calculate the average strategy over all players and information sets
        player = next(iter(strategies[0]))
        avg_strat = {player: {}}
        for information_set in strategies[0][player].keys():
            avg_strat[player][information_set] = {}
            for action in strategies[0][player][information_set].keys():
                avg_strat[player][information_set][action] = 0
        
        # Use DFS to calculate average strategy
        self.average_strategy_traverse_dfs(self.root, strategies, avg_strat, player)
        
        # Normalize the average strategy
        for infoset in avg_strat[player].keys():
            total_sum = sum(avg_strat[player][infoset].values())
            number_of_actions = len(avg_strat[player][infoset].keys())
            for action in avg_strat[player][infoset]:
                if total_sum == 0:
                    avg_strat[player][infoset][action] = 1/number_of_actions
                else:
                    avg_strat[player][infoset][action] /= total_sum
        
        return avg_strat
    
    def average_strategy_traverse_dfs(self, node: Node, strategies, avg_strat, player):
        # Depth-first search to calculate average strategy
        if node.is_terminal():
            return
        
        # Skip nodes where the player does not play
        if node.player != player:
            for action in node.actions():
                self.average_strategy_traverse_dfs(node.children[action], strategies, avg_strat, player)
            return
        
        information_set = node.information_set
        # reach_prob = self.calculate_reach_probability(avg_strat, node.history)
        strat_reach_prob = [self.calculate_node_reach_probability(s, node.history) for s in strategies]
        
        for action in node.actions():
            for i, strat in enumerate(strategies):
                # Sum up (reach probability * probability of action) for each strategy
                avg_strat[player][information_set][action] += strat_reach_prob[i] * strat[player][information_set][action]

            self.average_strategy_traverse_dfs(node.children[action], strategies, avg_strat, player)
                
    def best_response(self, strategies, node: Node = None):
        # Find the best response strategy against given strategies
        assert len(strategies.keys()) == len(self.players) + len(self.chance) - 1
        
        # Get the player we need
        player = next(p for p in self.players if p not in strategies)
        node = node or self.root
        
        best_response = {player: {}}
        self.get_best_response(strategies, node, best_response, player)
        
        return best_response
    
    def get_best_response(self, strategies, node: Node, best_response: dict, player, information_set_values: dict = {}):
        # Use recursive DFS to find the best response strategy
        if node.is_terminal():
            return self.matrix[node.history][player]
        
        action_values = {}

        # Player's turn
        if node.player == player:
            information_set = node.information_set
            best_response[player][information_set] = {}
            
            nodes_in_information_set = self.get_nodes_in_infoset(information_set, player)
        
            for action in node.actions():
                # Calculate the value of each action 
                action_values[action] = 0
                # Set probability to 0
                best_response[player][information_set][action] = 0
                for possible_node in nodes_in_information_set:
                    # For each possible node (history) the action value is (probability of reaching node * utility of action)
                    action_val = \
                        self.calculate_node_reach_probability(strategies, possible_node.history) *\
                        self.get_best_response(strategies, possible_node.children[action], best_response, player, information_set_values)
                    # For action value sum over every possible node
                    action_values[action] += action_val
            # Best response is then the action with max action value
            best_response[player][information_set][max(action_values, key=action_values.get)] = 1
            return max(action_values.values())
        
        # Not this player's turn
        else:
            for action in node.actions():
                action_values[action] = \
                    strategies[node.player][node.information_set][action] * \
                    self.get_best_response(strategies, node.children[action], best_response, player, information_set_values)

            # return (utility of action * probability of action) for each possible action
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
        # Create a uniform strategy for a given player
        strat = {player: {}}
        all_infosets = self.get_infosets()
        for infoset, nodes in all_infosets.items():
            # Get the infosets where it is passed in player's turn
            nodes = [n for n in nodes if n.player == player]
            if len(nodes) == 0:
                continue
            
            # Get actions
            node = nodes[0]
            actions = node.actions()
            
            # Probability of playing a is 1 / |A|
            count = len(actions)
            strat[player][infoset] = {a: 1 / count for a in actions}
        return strat
    
    def get_uniform_strat_for_players(self):
        p1 = self.create_uniform_strategy_for_player(self.players[0])
        p2 = self.create_uniform_strategy_for_player(self.players[1])
        return p1, p2

    def self_play(self, chance_strategies=None, iterations=50):
        # Initialize chance strategies for all chance players using a uniform strategy
        chance_strategies = chance_strategies or {c: self.create_uniform_strategy_for_player(c)[c] for c in self.chance}
        
        # Initialize player strategies for both players with a uniform strategy
        p1_strategies = [self.create_uniform_strategy_for_player(self.players[0])]
        p2_strategies = [self.create_uniform_strategy_for_player(self.players[1])]
        
        # List to store exploitabilities at each iteration
        exploitabilities = []
        
        # Run the self-play iterations
        for i in range(iterations):
            # Update player 1's strategy with the latest one and incorporate chance strategies
            p1_strat = p1_strategies[-1]
            p1_strat.update(chance_strategies)
            
            # Update player 2's strategy with the latest one and incorporate chance strategies
            p2_strat = p2_strategies[-1]
            p2_strat.update(chance_strategies)
            
            # Find the best response strategy for player 1 against player 2's strategy
            br_to_p1 = self.best_response(p1_strat)
            br_to_p1.update(chance_strategies)
            
            # Find the best response strategy for player 2 against player 1's strategy
            br_to_p2 = self.best_response(p2_strat)
            br_to_p2.update(chance_strategies)
            
            # Append the new strategies to the lists
            p1_strategies.append(br_to_p2)
            p2_strategies.append(br_to_p1)
            
            # Calculate the average strategies over all iterations for both players and chance players
            strategies = {
                self.players[0]: self.average_strategy(p1_strategies)[self.players[0]],
                self.players[1]: self.average_strategy(p2_strategies)[self.players[1]],
            }
            strategies.update(chance_strategies)
            
            # Calculate and store the exploitability at each iteration
            exploitabilities.append(self.calculate_exploitability(strategies))
            
        return p1_strategies, p2_strategies, exploitabilities



    def probability_of_reaching_state(self, strategies, h1, h2):
        # Calculate the probability of reaching history h2 starting from history h1 using given strategies
        node = self.get_node_by_history(h1)
        
        if not h2.startswith(h1):
            # Cannot be reached
            return 0
        
        # Get path from h1 to h2
        history = h2[len(h1):]
        prob = 1
        
        for a in history:
            if node.player in strategies:
                prob *= strategies[node.player][node.information_set][a]
            node = node.children[a]
        return prob

    
    def get_utility_in_infoset(self, strategies, infoset, player):
        # Calculate the utility in a specific information set for a given player using given strategies
        infoset_reach_prob = self.calculate_infoset_reach_probability(strategies, infoset)
        if infoset_reach_prob == 0:
            # Infoset cannot be reached
            return 0
        without_p = {key: val for key, val in strategies.items() if key != player}
        
        utility = 0
        terminal_nodes = [self.get_node_by_history(h) for h in self.matrix.keys()]
        # For every node h1 in the information set
        for node in self.get_nodes_in_infoset(infoset, player):
            # For every terminal node h2 in the tree
            for terminal_node in terminal_nodes:
                # Utility is:
                # the probability of reaching infoset (if the player is trying to reach it)
                # multiplied by the probability of reaching h2 from h1
                # multiplied by the reward in h2
                # for every h1, h2
                utility += \
                    self.calculate_node_reach_probability(without_p, node.history) * \
                    self.probability_of_reaching_state(strategies, node.history, terminal_node.history) * \
                    self.matrix[terminal_node.history][player]
                    
        # Normalize by the probability of reaching the infoset
        utility /= infoset_reach_prob
        return utility


                

    ############# CFR SECTION ##############

    def counterfactual_regret_minimization(self, chance_strategies=None, iterations=50):
        chance_strategies = chance_strategies or {c: self.create_uniform_strategy_for_player(c)[c] for c in self.chance}
        
        # Dict to store the regrets 
        regrets = {p: self.prep_regrets(p) for p in self.players}
        
        # Lists to store used strategies 
        player_strategies = {p: [] for p in self.players}
        exploitabilities = []
        
        for _ in range(iterations):
            # Get new strat T from regrets until T-1 
            for p in self.players:
                player_strategies[p].append(self.regret_matching(regrets[p], p))
            
            # Calculate exploitability from average strategies
            current_strategies = {}
            for p in self.players:
                current_strategies.update(self.average_strategy(player_strategies[p]))
            current_strategies.update(chance_strategies)
            
            exploitabilities.append(self.calculate_exploitability(current_strategies))
            
            # Update regrets using current strategies
            for p in self.players:
                self.update_regrets(regrets[p], current_strategies, p)
        
        return player_strategies, exploitabilities


    def prep_regrets(self, player):
        regrets = {}
        for infoset in self.infosets:
            nodes = self.get_nodes_in_infoset(infoset, player)
            if len(nodes) == 0:
                continue
            regrets[infoset] = {}
            for a in self.actions_in_infoset[infoset]:
                regrets[infoset][a] = 0
        return regrets
    
    
    def regret_matching(self, regrets, player):
        strat = {
            player: {}
        }
        for infoset in self.infosets:
            nodes = self.get_nodes_in_infoset(infoset, player)
            if len(nodes) == 0:
                continue

            strat[player][infoset] = {}
            # Get R+ (non-negative regret)
            regrets_plus = {k: max(v, 0) for k,v in regrets[infoset].items()}
            reg_sum = sum(regrets_plus.values())
            for a in self.actions_in_infoset[infoset]:
                # If atleast one action has positive regret
                if reg_sum > 0:
                    strat[player][infoset][a] = regrets_plus[a] / reg_sum
                # Else uniform
                else:
                    strat[player][infoset][a] = 1 / len(self.actions_in_infoset[infoset])
        return strat


    def update_regrets(self, regrets, strategies, player):
        # pi_-i
        without_p = {key: val for key, val in strategies.items() if key != player}

        for infoset in self.get_infosets():
            nodes = self.get_nodes_in_infoset(infoset, player)
            if len(nodes) == 0:
                continue
            
            for a in self.actions_in_infoset[infoset]:
                # Regret in information set I and action a = R(I,a)
                # is calculated against a strategy that is identical to the player's strategy 
                # but in infoset I the probability of playing action a is 1
                strategies_where_a_in_infoset = copy.deepcopy(strategies)
                for a2 in self.actions_in_infoset[infoset]:
                    strategies_where_a_in_infoset[player][infoset][a2] = 0
                strategies_where_a_in_infoset[player][infoset][a] = 1
                
                # R(I,a) = 
                # probability of reaching I if the player is trying to reach it *
                # (utility with strategies but the player plays a in I - utility with strategies)
                regrets[infoset][a] += \
                    self.calculate_infoset_reach_probability(without_p, infoset) * \
                    ( \
                        self.get_utility_in_infoset(strategies_where_a_in_infoset, infoset, player) - \
                        self.get_utility_in_infoset(strategies, infoset, player) \
                    )