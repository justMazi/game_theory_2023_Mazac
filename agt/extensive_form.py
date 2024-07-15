"""Extensive-form game representation.

A game is a tree of :class:`Node` objects plus a payoff table keyed by terminal
history. Nodes that a player cannot tell apart share an information set, which
is what makes the representation capable of expressing imperfect information.

This module holds only the representation and the value of a strategy profile;
the solving algorithms live in :mod:`agt.best_response`, :mod:`agt.cfr` and
:mod:`agt.fictitious_play`.
"""


class Node:
    """A single decision point in the game tree."""

    def __init__(self, player: str, information_set: str, children: dict = None, history: str = None):
        """
        Args:
            player: The player to act, or the empty string at a terminal node.
            information_set: Identifier shared by all nodes this player cannot
                distinguish; strategies are defined per information set.
            children: Map of action to the resulting child node.
            history: The sequence of actions leading here. Defaults to the
                information set, which is only correct for singleton sets, so
                pass it explicitly whenever a set spans several nodes.
        """
        self.player = player
        self.information_set = information_set
        self.history = history if history is not None else information_set
        self.children = children or {}

    def get_actions(self) -> list:
        """The actions available here, in insertion order."""
        return list(self.children)

    def is_terminal(self) -> bool:
        """Whether the node ends the game and therefore carries a payoff."""
        return not self.children

    def __repr__(self) -> str:
        return f"Node(player={self.player!r}, history={self.history!r})"


class ExtensiveFormGameCalculator:
    """An extensive-form game together with queries over strategy profiles."""

    def __init__(self, players: list, chance: list, root: Node, matrix: dict):
        """
        Args:
            players: The strategic players, in a fixed order.
            chance: Players whose actions are drawn by nature rather than chosen.
            root: The root of the game tree.
            matrix: Payoff per player, keyed by terminal history.
        """
        self.players = players
        self.chance = chance
        self.root = root
        self.matrix = matrix
        self.infosets = self._collect_infosets()

    def get_infosets(self) -> dict:
        """Map of information set to the set of nodes it contains."""
        return self.infosets

    def get_nodes_in_infoset(self, infoset: str, player: str = None) -> set:
        """Nodes in ``infoset``, optionally restricted to those where ``player`` acts."""
        nodes = self.infosets[infoset]
        if player is None:
            return nodes
        return {node for node in nodes if node.player == player}

    def _collect_infosets(self) -> dict:
        infosets = {}
        self._traverse(self.root, infosets)
        return infosets

    def _traverse(self, node: Node, infosets: dict) -> None:
        infosets.setdefault(node.information_set, set()).add(node)
        for child in node.children.values():
            self._traverse(child, infosets)

    def get_node_by_history(self, history: str) -> Node:
        """The node reached by following ``history`` from the root."""
        node = self.root
        for action in history:
            node = node.children[action]
        return node

    def calculate_node_reach_probability(self, strategies: dict, history: str) -> float:
        """Probability that ``strategies`` produce ``history``.

        Players absent from ``strategies`` contribute a factor of one, which is
        how a counterfactual reach probability is obtained.
        """
        node = self.root
        probability = 1.0

        for action in history:
            if node.player in strategies:
                probability *= strategies[node.player][node.information_set][action]
            node = node.children[action]

        return probability

    def calculate_infoset_reach_probability(self, strategies: dict, infoset: str, player: str = None) -> float:
        """Probability of reaching ``infoset``, summed over the nodes it contains."""
        return sum(
            self.calculate_node_reach_probability(strategies, node.history)
            for node in self.get_nodes_in_infoset(infoset, player)
        )

    def calculate_player_values(self, strategies: dict, node: Node = None) -> dict:
        """Expected payoff to every player when all of them follow ``strategies``."""
        node = node or self.root

        if node.is_terminal():
            return self.matrix[node.history]

        values = {player: 0.0 for player in self.players}
        action_probabilities = strategies[node.player][node.information_set]

        for action, probability in action_probabilities.items():
            if probability == 0:
                continue
            child_values = self.calculate_player_values(strategies, node.children[action])
            for player in self.players:
                values[player] += probability * child_values[player]

        return values

    def create_uniform_strategy_for_player(self, player: str) -> dict:
        """A strategy playing every available action with equal probability."""
        strategy = {}

        for infoset in self.infosets:
            nodes = self.get_nodes_in_infoset(infoset, player)
            if not nodes:
                continue
            actions = next(iter(nodes)).get_actions()
            strategy[infoset] = {action: 1 / len(actions) for action in actions}

        return {player: strategy}
