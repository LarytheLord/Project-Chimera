"""chimera.org -- the agent company ecosystem.

Phase 1: six-role sequential org (CEO -> R&D -> Marketing -> Production -> Ops -> QA),
WorkOrders persisted to SQLite, no external I/O. See plan in
/root/.claude/plans/before-planning-of-working-cuddly-metcalfe.md for the roadmap.
"""

from .approval import ApprovalGate, ApprovalRequired
from .org import Org
from .role import Role
from .store import WorkOrderStore
from .work_order import IllegalTransition, OrgStatus, WorkOrder

__all__ = [
    "Org",
    "Role",
    "WorkOrder",
    "OrgStatus",
    "IllegalTransition",
    "WorkOrderStore",
    "ApprovalGate",
    "ApprovalRequired",
]
