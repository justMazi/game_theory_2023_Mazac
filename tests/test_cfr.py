"""Convergence tests for the equilibrium-finding algorithms."""

import pytest

from agt.best_response import exploitability
from agt.cfr import CounterfactualRegretMinimizer
from agt.fictitious_play import FictitiousPlay
from agt.games import kuhn_poker, rock_paper_scissors

SOLVERS = [CounterfactualRegretMinimizer, FictitiousPlay]
SOLVER_IDS = ["cfr", "fictitious-play"]
GAMES = [rock_paper_scissors, kuhn_poker]
GAME_IDS = ["rps", "kuhn"]


@pytest.fixture(params=GAMES, ids=GAME_IDS)
def game(request):
    return request.param()


@pytest.fixture(params=SOLVERS, ids=SOLVER_IDS)
def solver_class(request):
    return request.param


def test_exploitability_decreases(game, solver_class):
    exploitabilities = solver_class(game).run(500)
    assert exploitabilities[-1] < exploitabilities[0]
    assert exploitabilities[-1] < 0.05


def test_exploitability_is_never_negative(game, solver_class):
    """Exploitability measures a gain from deviating, so it cannot be below zero."""
    assert all(value >= 0 for value in solver_class(game).run(200))


def test_average_strategy_is_a_valid_distribution(game, solver_class):
    solver = solver_class(game)
    solver.run(50)

    for player_strategy in solver.average_strategy().values():
        for action_probabilities in player_strategy.values():
            assert all(p >= 0 for p in action_probabilities.values())
            assert sum(action_probabilities.values()) == pytest.approx(1)


def test_cfr_finds_the_known_value_of_kuhn_poker():
    """Optimal play in three-card Kuhn poker is worth -1/18 to the first player."""
    game = kuhn_poker()
    solver = CounterfactualRegretMinimizer(game)
    solver.run(5000)

    value = game.calculate_player_values(solver.average_strategy())["Player1"]
    assert value == pytest.approx(-1 / 18, abs=0.005)


def test_cfr_converges_to_the_uniform_equilibrium_of_rock_paper_scissors():
    solver = CounterfactualRegretMinimizer(rock_paper_scissors())
    solver.run(1000)

    for probability in solver.average_strategy()["Player1"][""].values():
        assert probability == pytest.approx(1 / 3, abs=0.02)


def test_uniform_play_is_exploitable_in_kuhn_poker():
    """Uniform play is an equilibrium of rock-paper-scissors but not of Kuhn poker."""
    game = kuhn_poker()
    uniform = {
        player: game.create_uniform_strategy_for_player(player)[player]
        for player in game.players + game.chance
    }
    assert exploitability(game, uniform) > 0

    solver = CounterfactualRegretMinimizer(game)
    solver.run(2000)
    assert exploitability(game, solver.average_strategy()) < 0.01


def test_uniform_play_is_an_equilibrium_of_rock_paper_scissors():
    game = rock_paper_scissors()
    uniform = {
        player: game.create_uniform_strategy_for_player(player)[player]
        for player in game.players
    }
    assert exploitability(game, uniform) == pytest.approx(0)
