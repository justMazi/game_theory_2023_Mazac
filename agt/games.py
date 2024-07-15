"""Benchmark games in extensive form.

Each factory returns a fully constructed :class:`ExtensiveFormGameCalculator`
so that algorithms can be compared on identical inputs.
"""

from .extensive_form import ExtensiveFormGameCalculator, Node

PLAYERS = ["Player1", "Player2"]
CHANCE = ["Chance"]


def rock_paper_scissors() -> ExtensiveFormGameCalculator:
    """Rock-paper-scissors as an imperfect-information extensive-form game.

    Player 2 acts second but shares a single information set, which makes the
    game strategically simultaneous.
    """
    moves = ["R", "P", "S"]
    beats = {("R", "S"), ("P", "R"), ("S", "P")}

    root = Node(
        "Player1",
        "",
        {
            first: Node(
                "Player2",
                "",
                {second: Node("", first + second) for second in moves},
                history=first,
            )
            for first in moves
        },
    )

    matrix = {}
    for first in moves:
        for second in moves:
            if (first, second) in beats:
                payoff = 1
            elif (second, first) in beats:
                payoff = -1
            else:
                payoff = 0
            matrix[first + second] = {"Player1": payoff, "Player2": -payoff}

    return ExtensiveFormGameCalculator(PLAYERS, [], root, matrix)


KUHN_CARDS = ["J", "Q", "K"]


def kuhn_poker(cards: list = None) -> ExtensiveFormGameCalculator:
    """Kuhn poker, the standard CFR benchmark.

    Chance deals one private card to each player from a deck of distinct cards,
    then a single betting round follows. Information sets are keyed by the
    acting player's own card plus the public betting history, so a player cannot
    distinguish the deals that differ only in the opponent's card.

    The three-card default is the classic game, whose value to Player 1 is
    -1/18 under optimal play.
    """
    cards = cards or KUHN_CARDS

    def betting_round(deal: str, own_card: str) -> Node:
        """Build the betting subtree that follows a deal.

        ``deal`` is the full two-card history; ``own_card`` is the card Player 1
        holds and therefore the prefix of every information set they own.
        """
        opponent_card = deal[1]
        return Node(
            "Player1",
            own_card,
            {
                "C": Node(
                    "Player2",
                    f"{opponent_card}C",
                    {
                        "B": Node(
                            "Player1",
                            f"{own_card}CB",
                            {
                                "C": Node("", f"{deal}CBC"),
                                "F": Node("", f"{deal}CBF"),
                            },
                            f"{deal}CB",
                        ),
                        "C": Node("", f"{deal}CC"),
                    },
                    f"{deal}C",
                ),
                "B": Node(
                    "Player2",
                    f"{opponent_card}B",
                    {
                        "C": Node("", f"{deal}BC"),
                        "F": Node("", f"{deal}BF"),
                    },
                    f"{deal}B",
                ),
            },
            deal,
        )

    root = Node(
        "Chance",
        "",
        {
            first: Node(
                "Chance",
                f"deal:{first}",
                {
                    second: betting_round(first + second, first)
                    for second in cards
                    if second != first
                },
                first,
            )
            for first in cards
        },
    )

    matrix = _kuhn_payoffs(cards)
    return ExtensiveFormGameCalculator(PLAYERS, CHANCE, root, matrix)


def _kuhn_payoffs(cards: list) -> dict:
    """Terminal payoffs for Kuhn poker, from Player 1's perspective.

    Both players ante 1. A call after a bet raises the stake to 2 per player; a
    fold hands the pot to whoever bet. At showdown the higher card wins.
    """
    rank = {card: index for index, card in enumerate(cards)}
    matrix = {}

    for first in cards:
        for second in cards:
            if first == second:
                continue
            deal = first + second
            if rank[first] > rank[second]:
                showdown_winner = 1
            elif rank[first] < rank[second]:
                showdown_winner = -1
            else:
                showdown_winner = 0

            # Both check: showdown for the antes alone.
            matrix[f"{deal}CC"] = _payoff(showdown_winner)
            # Check, bet, call: showdown for ante plus bet.
            matrix[f"{deal}CBC"] = _payoff(showdown_winner * 2)
            # Check, bet, fold: Player 1 folds and loses the ante.
            matrix[f"{deal}CBF"] = _payoff(-1)
            # Bet, call: showdown for ante plus bet.
            matrix[f"{deal}BC"] = _payoff(showdown_winner * 2)
            # Bet, fold: Player 2 folds, Player 1 takes the antes.
            matrix[f"{deal}BF"] = _payoff(1)

    return matrix


def _payoff(player1_value: int) -> dict:
    """Wrap a zero-sum payoff into the per-player dictionary the tree expects."""
    return {"Player1": player1_value, "Player2": -player1_value}
