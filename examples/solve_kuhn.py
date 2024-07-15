"""Solve Kuhn poker with CFR and print the resulting strategy."""

from agt.cfr import CounterfactualRegretMinimizer
from agt.games import kuhn_poker

ITERATIONS = 5000


def main() -> None:
    game = kuhn_poker()
    solver = CounterfactualRegretMinimizer(game)
    exploitabilities = solver.run(ITERATIONS)
    strategy = solver.average_strategy()

    print(f"after {ITERATIONS} iterations")
    print(f"  exploitability {exploitabilities[-1]:.5f}")
    print(f"  value to Player1 {game.calculate_player_values(strategy)['Player1']:.5f}")
    print(f"  known game value {-1 / 18:.5f}")

    print("\naverage strategy")
    for player in game.players:
        print(f"  {player}")
        for infoset in sorted(strategy[player]):
            actions = strategy[player][infoset]
            formatted = "  ".join(f"{a} {p:.3f}" for a, p in sorted(actions.items()))
            print(f"    {infoset:<5} {formatted}")


if __name__ == "__main__":
    main()
