"""Fictitious play over extensive-form games.

Each iteration every player best-responds to the opponents' current average
strategy, and the response is folded into that average. It is the natural
baseline for CFR: the same convergence measure applies, but the update rule
uses a full best response rather than accumulated regret.

In an extensive-form game the average has to be weighted by how often each
information set is actually reached, otherwise responses at rarely visited
information sets are over-counted and the average need not converge. This is
the realization-weighted average of Heinrich et al. (2015).
"""

from collections import defaultdict

from .best_response import BestResponse, exploitability
from .extensive_form import ExtensiveFormGameCalculator


class FictitiousPlay:
    """Solves an extensive-form game by iterated best response to the average strategy."""

    def __init__(self, game: ExtensiveFormGameCalculator):
        self.game = game
        # Keyed by (player, infoset): two players may share an infoset name.
        self.strategy_sums = defaultdict(lambda: defaultdict(float))
        self._chance_strategies = {
            chance_player: self.game.create_uniform_strategy_for_player(chance_player)[
                chance_player
            ]
            for chance_player in self.game.chance
        }

    def run(self, iterations: int) -> list:
        """Run fictitious play and return the exploitability of the average strategy per iteration."""
        exploitabilities = []
        profile = self.average_strategy()

        for iteration in range(iterations):
            responses = {
                player: BestResponse(
                    self.game, player, self._opponents_of(player, profile)
                ).strategy()
                for player in self.game.players
            }
            # Accumulated only once every response is computed, so both
            # players move against the same profile.
            for player, response in responses.items():
                self._accumulate(player, response, profile, weight=1.0)

            profile = self.average_strategy()
            exploitabilities.append(exploitability(self.game, profile))

        return exploitabilities

    def _opponents_of(self, player: str, profile: dict) -> dict:
        return {other: strategy for other, strategy in profile.items() if other != player}

    def _accumulate(self, player: str, response: dict, profile: dict, weight: float) -> None:
        """Fold a best response into the running average, weighted by reach probability.

        Weighting by the probability that ``player`` reaches each information
        set is what makes the average a valid behavioural strategy rather than
        an unweighted mixture of pure strategies.
        """
        for infoset, action_probabilities in response.items():
            reach = self.game.calculate_infoset_reach_probability(profile, infoset, player)
            for action, probability in action_probabilities.items():
                self.strategy_sums[player, infoset][action] += weight * reach * probability

    def average_strategy(self) -> dict:
        """The realization-weighted average of every best response played so far."""
        strategies = {
            player: self.game.create_uniform_strategy_for_player(player)[player]
            for player in self.game.players
        }
        strategies.update(self._chance_strategies)

        for (player, infoset), sums in self.strategy_sums.items():
            total = sum(sums.values())
            if total > 0:
                strategies[player][infoset] = {a: s / total for a, s in sums.items()}

        return strategies
