"""Concrete role implementations for the Phase 1 sequential org."""

from .ceo import CEORole
from .marketing import MarketingRole
from .ops import OpsRole
from .production import ProductionRole
from .qa import QARole
from .rnd import RnDRole

__all__ = [
    "CEORole",
    "RnDRole",
    "MarketingRole",
    "ProductionRole",
    "OpsRole",
    "QARole",
]


def all_roles() -> list[type]:
    """Default sequential role order: CEO -> R&D -> Marketing -> Production -> Ops -> QA."""
    return [CEORole, RnDRole, MarketingRole, ProductionRole, OpsRole, QARole]
