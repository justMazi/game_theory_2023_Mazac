"""Algorithmic game theory algorithms implemented from scratch.

Normal-form solution concepts, extensive-form game trees, and the solvers that
converge to equilibrium on them: best response, fictitious play and
counterfactual regret minimization.
"""

from .best_response import BestResponse, best_response_value, exploitability
from .cfr import CounterfactualRegretMinimizer
from .extensive_form import ExtensiveFormGameCalculator, Node
from .fictitious_play import FictitiousPlay
from .games import kuhn_poker, rock_paper_scissors
from .normal_form import NormalFormGame

__all__ = [
    "BestResponse",
    "CounterfactualRegretMinimizer",
    "ExtensiveFormGameCalculator",
    "FictitiousPlay",
    "NormalFormGame",
    "Node",
    "best_response_value",
    "exploitability",
    "kuhn_poker",
    "rock_paper_scissors",
]
