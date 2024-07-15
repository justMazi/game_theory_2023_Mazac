"""Normal-form games: solution concepts on a pair of payoff matrices.

A two-player normal-form game is given by the row player's payoff matrix and
the column player's. Strategies are probability vectors over the players' own
actions, and the concepts here -- best response, exploitability, dominance and
support enumeration -- are the finite-game counterparts of the extensive-form
machinery in :mod:`agt.extensive_form`.
"""

from itertools import combinations

import numpy as np

TOLERANCE = 1e-9


class NormalFormGame:
    """A two-player normal-form game and the solution concepts defined on it."""

    def __init__(self, row_payoffs: np.ndarray, column_payoffs: np.ndarray = None):
        """
        Args:
            row_payoffs: Payoff to the row player for each action pair.
            column_payoffs: Payoff to the column player. Defaults to the
                negation of ``row_payoffs``, making the game zero-sum.
        """
        self.row_payoffs = np.asarray(row_payoffs, dtype=float)
        if column_payoffs is None:
            self.column_payoffs = -self.row_payoffs
        else:
            self.column_payoffs = np.asarray(column_payoffs, dtype=float)

        if self.row_payoffs.shape != self.column_payoffs.shape:
            raise ValueError("both payoff matrices must have the same shape")

    @property
    def shape(self) -> tuple:
        """Number of row actions and column actions."""
        return self.row_payoffs.shape

    def values(self, row_strategy: np.ndarray, column_strategy: np.ndarray) -> tuple:
        """Expected payoff to each player under the given mixed strategies."""
        row_strategy = _as_distribution(row_strategy)
        column_strategy = _as_distribution(column_strategy)

        outer = np.outer(row_strategy, column_strategy)
        row_value = float((outer * self.row_payoffs).sum())
        column_value = float((outer * self.column_payoffs).sum())
        return row_value, column_value

    def best_response_to_column(self, column_strategy: np.ndarray) -> np.ndarray:
        """The row player's pure best response to ``column_strategy``."""
        return _pure_best_response(self.row_payoffs @ _as_distribution(column_strategy))

    def best_response_to_row(self, row_strategy: np.ndarray) -> np.ndarray:
        """The column player's pure best response to ``row_strategy``."""
        return _pure_best_response(_as_distribution(row_strategy) @ self.column_payoffs)

    def incentives_to_deviate(self, row_strategy: np.ndarray, column_strategy: np.ndarray) -> tuple:
        """How much each player would gain by switching to a best response."""
        row_value, column_value = self.values(row_strategy, column_strategy)

        best_row, _ = self.values(
            self.best_response_to_column(column_strategy), column_strategy
        )
        _, best_column = self.values(
            row_strategy, self.best_response_to_row(row_strategy)
        )

        return best_row - row_value, best_column - column_value

    def nash_conv(self, row_strategy: np.ndarray, column_strategy: np.ndarray) -> float:
        """Total gain available to the players from deviating; zero at equilibrium."""
        return sum(self.incentives_to_deviate(row_strategy, column_strategy))

    def exploitability(self, row_strategy: np.ndarray, column_strategy: np.ndarray) -> float:
        """Average gain per player from deviating; the standard convergence measure."""
        return self.nash_conv(row_strategy, column_strategy) / 2

    def iterated_removal_of_dominated_actions(self) -> "NormalFormGame":
        """Repeatedly delete strictly dominated actions until none remain.

        An action is strictly dominated when some other action of the same
        player pays strictly more against every opponent action, so no rational
        player ever uses it and removing it preserves the equilibria.
        """
        row_payoffs, column_payoffs = self.row_payoffs, self.column_payoffs

        while True:
            # One player at a time, so each check sees the current matrix.
            surviving_rows = _undominated(row_payoffs, axis=0)
            if len(surviving_rows) < row_payoffs.shape[0]:
                row_payoffs = row_payoffs[surviving_rows, :]
                column_payoffs = column_payoffs[surviving_rows, :]
                continue

            surviving_columns = _undominated(column_payoffs, axis=1)
            if len(surviving_columns) < column_payoffs.shape[1]:
                row_payoffs = row_payoffs[:, surviving_columns]
                column_payoffs = column_payoffs[:, surviving_columns]
                continue

            return NormalFormGame(row_payoffs, column_payoffs)

    def equilibria_by_support_enumeration(self) -> list:
        """All Nash equilibria found by enumerating candidate supports.

        For a fixed pair of supports an equilibrium solves a linear system:
        every action a player actually uses must earn the same value, and no
        unused action may earn more. The number of supports is exponential, so
        this is practical only on small games.
        """
        rows, columns = self.shape
        equilibria = []

        for row_size in range(1, rows + 1):
            for row_support in combinations(range(rows), row_size):
                for column_size in range(1, columns + 1):
                    for column_support in combinations(range(columns), column_size):
                        equilibrium = self._solve_support(row_support, column_support)
                        if equilibrium is not None:
                            equilibria.append(equilibrium)

        return equilibria

    def _solve_support(self, row_support: tuple, column_support: tuple):
        """Solve for an equilibrium on a fixed pair of supports, if one exists."""
        rows, columns = list(row_support), list(column_support)

        # Each player mixes so as to make the *opponent* indifferent across the
        # opponent's own support, which is what sustains a mixed equilibrium.
        row_strategy = _indifference_strategy(
            self.column_payoffs[np.ix_(rows, columns)].T, len(rows)
        )
        column_strategy = _indifference_strategy(
            self.row_payoffs[np.ix_(rows, columns)], len(columns)
        )
        if row_strategy is None or column_strategy is None:
            return None

        full_row = np.zeros(self.shape[0])
        full_row[rows] = row_strategy
        full_column = np.zeros(self.shape[1])
        full_column[columns] = column_strategy

        if self.nash_conv(full_row, full_column) > 1e-7:
            return None
        return full_row, full_column


def _undominated(payoffs: np.ndarray, axis: int) -> list:
    """Indices along ``axis`` that no other index strictly dominates."""
    count = payoffs.shape[axis]
    surviving = []

    for candidate in range(count):
        candidate_payoffs = np.take(payoffs, candidate, axis=axis)
        dominated = any(
            np.all(np.take(payoffs, other, axis=axis) > candidate_payoffs)
            for other in range(count)
            if other != candidate
        )
        if not dominated:
            surviving.append(candidate)

    return surviving


def _indifference_strategy(payoffs: np.ndarray, size: int):
    """Distribution over ``size`` actions equalising every row of ``payoffs``.

    Returns ``None`` when the system has no valid probability vector, which is
    the usual outcome for a support that carries no equilibrium.
    """
    rows = payoffs.shape[0]

    # Unknowns are the ``size`` probabilities followed by the common value v.
    # Each row contributes ``payoffs @ p - v = 0``; the last row is sum(p) = 1.
    system = np.zeros((rows + 1, size + 1))
    system[:rows, :size] = payoffs
    system[:rows, size] = -1
    system[rows, :size] = 1

    target = np.zeros(rows + 1)
    target[rows] = 1

    try:
        solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if not np.allclose(system @ solution, target, atol=1e-8):
        return None

    strategy = solution[:size]
    if np.any(strategy < -TOLERANCE):
        return None
    return np.clip(strategy, 0, None)


def _pure_best_response(action_values: np.ndarray) -> np.ndarray:
    """Put all probability on the highest-valued action."""
    strategy = np.zeros(len(action_values))
    strategy[int(np.argmax(action_values))] = 1.0
    return strategy


def _as_distribution(strategy: np.ndarray) -> np.ndarray:
    """Validate a strategy and flatten it to a one-dimensional distribution."""
    strategy = np.asarray(strategy, dtype=float).reshape(-1)
    if np.any(strategy < -TOLERANCE):
        raise ValueError("strategy has negative probabilities")
    if abs(strategy.sum() - 1) > 1e-6:
        raise ValueError(f"strategy sums to {strategy.sum()}, expected 1")
    return strategy
