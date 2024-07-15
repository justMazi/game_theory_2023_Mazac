"""Tests for the normal-form solution concepts."""

import numpy as np
import pytest

from agt.normal_form import NormalFormGame

ROCK_PAPER_SCISSORS = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
MATCHING_PENNIES = np.array([[1, -1], [-1, 1]])
UNIFORM_OVER_THREE = np.ones(3) / 3


def test_zero_sum_is_the_default_second_matrix():
    game = NormalFormGame(ROCK_PAPER_SCISSORS)
    assert np.array_equal(game.column_payoffs, -ROCK_PAPER_SCISSORS)


def test_mismatched_matrices_are_rejected():
    with pytest.raises(ValueError):
        NormalFormGame(np.zeros((2, 2)), np.zeros((2, 3)))


def test_strategies_must_be_distributions():
    game = NormalFormGame(MATCHING_PENNIES)
    with pytest.raises(ValueError):
        game.values(np.array([0.5, 0.2]), np.array([0.5, 0.5]))


def test_uniform_play_is_an_equilibrium_of_rock_paper_scissors():
    game = NormalFormGame(ROCK_PAPER_SCISSORS)
    assert game.exploitability(UNIFORM_OVER_THREE, UNIFORM_OVER_THREE) == pytest.approx(0)


def test_pure_play_is_exploitable_in_rock_paper_scissors():
    """Always playing rock loses a full unit to a player who answers with paper."""
    game = NormalFormGame(ROCK_PAPER_SCISSORS)
    always_rock = np.array([1.0, 0.0, 0.0])

    _, column_incentive = game.incentives_to_deviate(always_rock, UNIFORM_OVER_THREE)
    assert column_incentive == pytest.approx(1.0)


def test_best_response_picks_the_highest_paying_action():
    game = NormalFormGame(ROCK_PAPER_SCISSORS)
    always_first_action = np.array([1.0, 0.0, 0.0])

    # The third action is the only one that pays the column player against the
    # row player's first action.
    assert np.array_equal(game.best_response_to_row(always_first_action), [0.0, 0.0, 1.0])


def test_iterated_removal_solves_a_dominance_solvable_game():
    """Repeated elimination reduces this game to its unique equilibrium cell."""
    row_payoffs = np.array([[11, 1, 7], [4, 3, 6], [-1, 2, 8]])
    column_payoffs = np.array([[3, 4, 3], [1, 3, 2], [9, 8, -1]])

    reduced = NormalFormGame(row_payoffs, column_payoffs).iterated_removal_of_dominated_actions()

    assert reduced.shape == (1, 1)
    assert reduced.row_payoffs[0, 0] == 3
    assert reduced.column_payoffs[0, 0] == 3


def test_iterated_removal_leaves_games_without_dominance_untouched():
    game = NormalFormGame(ROCK_PAPER_SCISSORS)
    assert game.iterated_removal_of_dominated_actions().shape == game.shape


def test_support_enumeration_finds_the_mixed_equilibrium_of_matching_pennies():
    equilibria = NormalFormGame(MATCHING_PENNIES).equilibria_by_support_enumeration()

    assert len(equilibria) == 1
    for strategy in equilibria[0]:
        assert strategy == pytest.approx([0.5, 0.5])


def test_support_enumeration_finds_the_uniform_equilibrium_of_rock_paper_scissors():
    equilibria = NormalFormGame(ROCK_PAPER_SCISSORS).equilibria_by_support_enumeration()

    assert len(equilibria) == 1
    for strategy in equilibria[0]:
        assert strategy == pytest.approx(UNIFORM_OVER_THREE)


def test_every_enumerated_equilibrium_has_zero_nash_conv():
    """Whatever the enumeration returns must actually be an equilibrium."""
    game = NormalFormGame(np.array([[1, -2], [0, 4]]), np.array([[4, -1], [1, 1]]))

    for row_strategy, column_strategy in game.equilibria_by_support_enumeration():
        assert game.nash_conv(row_strategy, column_strategy) == pytest.approx(0, abs=1e-6)
