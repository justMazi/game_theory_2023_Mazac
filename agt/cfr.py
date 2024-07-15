"""Counterfactual regret minimization over extensive-form games.

The algorithm follows Zinkevich et al. (2007). Each iteration is a single
recursive walk of the game tree that simultaneously accumulates counterfactual
regret for every information set and every player, which is what makes CFR
practical on games larger than a toy tree.
"""

from collections import defaultdict

from .best_response import exploitability
from .extensive_form import ExtensiveFormGameCalculator, Node


class CounterfactualRegretMinimizer:
    """Solves an extensive-form game by regret matching on counterfactual regret."""

    def __init__(self, game: ExtensiveFormGameCalculator):
        self.game = game
        # Keyed by (player, infoset): players may share an infoset name.
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sums = defaultdict(lambda: defaultdict(float))

    def run(self, iterations: int) -> list:
        """Run CFR and return the exploitability of the average strategy per iteration."""
        exploitabilities = []
        for _ in range(iterations):
            for player in self.game.players:
                self._walk(self.game.root, player, reach_player=1.0, reach_others=1.0)
            exploitabilities.append(
                exploitability(self.game, self.average_strategy())
            )
        return exploitabilities

    def _walk(self, node: Node, player: str, reach_player: float, reach_others: float) -> float:
        """Recursively compute ``player``'s counterfactual value at ``node``.

        ``reach_player`` is the probability that ``player``'s own strategy
        reaches this node; ``reach_others`` is the same for every other player
        including chance. Regret is weighted by ``reach_others`` alone, which is
        exactly what makes the quantity counterfactual.
        """
        if node.is_terminal():
            return self.game.matrix[node.history][player]

        strategy = self._current_strategy(node)

        if node.player != player:
            # Opponent or chance node: recurse and average over their actions.
            value = 0.0
            for action, probability in strategy.items():
                value += probability * self._walk(
                    node.children[action], player, reach_player, reach_others * probability
                )
            return value

        key = (player, node.information_set)

        action_values = {}
        node_value = 0.0
        for action, probability in strategy.items():
            action_values[action] = self._walk(
                node.children[action], player, reach_player * probability, reach_others
            )
            node_value += probability * action_values[action]

        for action, action_value in action_values.items():
            self.regrets[key][action] += reach_others * (action_value - node_value)
            self.strategy_sums[key][action] += reach_player * strategy[action]

        return node_value

    def _current_strategy(self, node: Node) -> dict:
        """Strategy at ``node``: regret matching for players, uniform for chance."""
        actions = list(node.get_actions())
        if node.player in self.game.chance:
            return {action: 1 / len(actions) for action in actions}
        return _regret_matching(self.regrets[node.player, node.information_set], actions)

    def average_strategy(self) -> dict:
        """The average strategy over all iterations, which is what converges to equilibrium.

        Information sets the walk never reached keep a uniform strategy, so the
        result is always a complete, well-defined strategy profile.
        """
        strategies = {
            player: self.game.create_uniform_strategy_for_player(player)[player]
            for player in self.game.players + self.game.chance
        }

        for (player, infoset), sums in self.strategy_sums.items():
            total = sum(sums.values())
            if total > 0:
                strategies[player][infoset] = {a: s / total for a, s in sums.items()}

        return strategies


def _regret_matching(regrets: dict, actions: list) -> dict:
    """Turn accumulated regrets into a strategy, uniform when no regret is positive."""
    positive = {action: max(regrets[action], 0.0) for action in actions}
    total = sum(positive.values())
    if total <= 0:
        return {action: 1 / len(actions) for action in actions}
    return {action: value / total for action, value in positive.items()}
