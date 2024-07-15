"""Structural tests for the benchmark game trees."""

import pytest

from agt.games import KUHN_CARDS, kuhn_poker, rock_paper_scissors


@pytest.fixture(params=[rock_paper_scissors, kuhn_poker], ids=["rps", "kuhn"])
def game(request):
    return request.param()


def test_games_are_zero_sum(game):
    for payoffs in game.matrix.values():
        assert sum(payoffs.values()) == 0


def test_kuhn_deals_distinct_cards():
    """A card cannot be dealt to both players, so no deal repeats a card."""
    game = kuhn_poker()
    deals = {history[:2] for history in game.matrix}

    assert len(deals) == len(KUHN_CARDS) * (len(KUHN_CARDS) - 1)
    assert all(first != second for first, second in deals)


def test_kuhn_information_sets_hide_the_opponent_card():
    """Every acting information set must span more than one possible deal."""
    game = kuhn_poker()
    acting_sets = {
        infoset: {node for node in nodes if node.player in game.players}
        for infoset, nodes in game.get_infosets().items()
    }

    non_trivial = {i: n for i, n in acting_sets.items() if len(n) > 1}
    assert non_trivial, "the game must contain at least one non-singleton information set"
    for nodes in non_trivial.values():
        assert len({node.history for node in nodes}) == len(nodes)


def test_rock_paper_scissors_payoffs_are_antisymmetric():
    game = rock_paper_scissors()
    for first in "RPS":
        for second in "RPS":
            assert (
                game.matrix[first + second]["Player1"]
                == -game.matrix[second + first]["Player1"]
            )
