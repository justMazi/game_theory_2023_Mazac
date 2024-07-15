"""Compare CFR against fictitious play on the benchmark games.

Running this script regenerates ``docs/convergence.png``, the figure shown in
the README.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agt.cfr import CounterfactualRegretMinimizer
from agt.fictitious_play import FictitiousPlay
from agt.games import kuhn_poker, rock_paper_scissors

ITERATIONS = 500
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "convergence.png"


def main() -> None:
    games = {"Rock-paper-scissors": rock_paper_scissors, "Kuhn poker": kuhn_poker}
    figure, axes = plt.subplots(1, len(games), figsize=(11, 4.2), sharey=True)

    for axis, (title, build_game) in zip(axes, games.items()):
        cfr = CounterfactualRegretMinimizer(build_game()).run(ITERATIONS)
        fictitious_play = FictitiousPlay(build_game()).run(ITERATIONS)

        axis.plot(cfr, label="CFR")
        axis.plot(fictitious_play, label="Fictitious play")
        axis.set_yscale("log")
        axis.set_xlabel("Iteration")
        axis.set_title(title)
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Exploitability")
    axes[0].legend()
    figure.suptitle("Exploitability of the average strategy")
    figure.tight_layout()

    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
