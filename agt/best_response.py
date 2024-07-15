"""Best response computation for extensive-form games.

A best response has to be chosen per information set rather than per node: a
player cannot distinguish the nodes inside one information set, so a single
action must be picked for all of them. The value of an action is therefore the
sum of its value over every node in the set, each weighted by the probability
that the opponents' strategies reach that node.
"""

from collections import defaultdict

from .extensive_form import ExtensiveFormGameCalculator, Node


class BestResponse:
    """Computes a player's best response against a fixed profile of opponent strategies."""

    def __init__(self, game: ExtensiveFormGameCalculator, player: str, opponents: dict):
        self.game = game
        self.player = player
        self.opponents = opponents
        self._action_values = defaultdict(lambda: defaultdict(float))
        self._strategy = None

    def strategy(self) -> dict:
        """The best-responding strategy, as ``{infoset: {action: probability}}``."""
        if self._strategy is None:
            self._collect_action_values(self.game.root, reach=1.0)
            self._strategy = {
                infoset: self._pure_strategy(values)
                for infoset, values in self._action_values.items()
            }
        return self._strategy

    def value(self) -> float:
        """The player's expected utility when playing the best response."""
        return self._value_of(self.game.root, self.strategy())

    def _collect_action_values(self, node: Node, reach: float) -> None:
        """Accumulate counterfactual action values for every information set.

        ``reach`` is the probability that the opponents reach ``node``; the
        responding player's own probabilities are deliberately excluded, so the
        accumulated values are comparable across the whole information set.
        """
        if node.is_terminal():
            return

        if node.player == self.player:
            for action in node.get_actions():
                child = node.children[action]
                self._action_values[node.information_set][action] += (
                    reach * self._terminal_reach_value(child, reach=1.0)
                )
                self._collect_action_values(child, reach)
            return

        for action in node.get_actions():
            probability = self.opponents[node.player][node.information_set][action]
            if probability > 0:
                self._collect_action_values(node.children[action], reach * probability)

    def _terminal_reach_value(self, node: Node, reach: float) -> float:
        """Expected payoff below ``node`` assuming the player plays to maximise it."""
        if node.is_terminal():
            return reach * self.game.matrix[node.history][self.player]

        if node.player == self.player:
            return max(
                self._terminal_reach_value(node.children[action], reach)
                for action in node.get_actions()
            )

        return sum(
            self._terminal_reach_value(
                node.children[action],
                reach * self.opponents[node.player][node.information_set][action],
            )
            for action in node.get_actions()
        )

    def _value_of(self, node: Node, strategy: dict) -> float:
        """Expected payoff of ``node`` under the responding player's ``strategy``."""
        if node.is_terminal():
            return self.game.matrix[node.history][self.player]

        if node.player == self.player:
            probabilities = strategy[node.information_set]
        else:
            probabilities = self.opponents[node.player][node.information_set]

        return sum(
            probability * self._value_of(node.children[action], strategy)
            for action, probability in probabilities.items()
            if probability > 0
        )

    @staticmethod
    def _pure_strategy(action_values: dict) -> dict:
        """Put all probability on the highest-valued action."""
        best = max(action_values, key=action_values.get)
        return {action: float(action == best) for action in action_values}


def best_response_value(game: ExtensiveFormGameCalculator, player: str, opponents: dict) -> float:
    """Utility ``player`` can guarantee against ``opponents`` by best responding."""
    return BestResponse(game, player, opponents).value()


def exploitability(game: ExtensiveFormGameCalculator, profile: dict) -> float:
    """Average gain the players could obtain by unilaterally best responding.

    Zero exactly at a Nash equilibrium, and non-negative everywhere, which makes
    it the standard convergence measure for equilibrium-finding algorithms.
    """
    missing = (set(game.players) | set(game.chance)) - set(profile)
    if missing:
        raise ValueError(f"profile is missing strategies for {sorted(missing)}")

    values = game.calculate_player_values(profile)
    total = 0.0
    for player in game.players:
        opponents = {p: s for p, s in profile.items() if p != player}
        total += best_response_value(game, player, opponents) - values[player]
    return total / len(game.players)
