"""WorkOrder: the unit of currency passed between roles in the org."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OrgStatus(str, Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    REJECTED = "rejected"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


# Legal status transitions. Anything not in this set raises IllegalTransition.
_LEGAL_TRANSITIONS: set[tuple[OrgStatus, OrgStatus]] = {
    (OrgStatus.OPEN, OrgStatus.ASSIGNED),
    (OrgStatus.ASSIGNED, OrgStatus.IN_PROGRESS),
    (OrgStatus.IN_PROGRESS, OrgStatus.PASSED),
    (OrgStatus.PASSED, OrgStatus.ASSIGNED),
    (OrgStatus.PASSED, OrgStatus.COMPLETED),
    (OrgStatus.IN_PROGRESS, OrgStatus.REJECTED),
    (OrgStatus.REJECTED, OrgStatus.ASSIGNED),
    (OrgStatus.REJECTED, OrgStatus.FAILED),
    (OrgStatus.IN_PROGRESS, OrgStatus.AWAITING_APPROVAL),
    (OrgStatus.AWAITING_APPROVAL, OrgStatus.IN_PROGRESS),
    (OrgStatus.AWAITING_APPROVAL, OrgStatus.FAILED),
}


class IllegalTransition(ValueError):
    """Raised when a WorkOrder is asked to move between two incompatible statuses."""


@dataclass
class WorkOrder:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    assigned_role: Optional[str] = None
    status: OrgStatus = OrgStatus.OPEN
    history: list[dict] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    reject_count: int = 0
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def _transition(self, new_status: OrgStatus) -> None:
        if (self.status, new_status) not in _LEGAL_TRANSITIONS:
            raise IllegalTransition(
                f"Illegal transition: {self.status.value} -> {new_status.value}"
            )
        self.status = new_status
        self.updated_at = time.time()

    def assign_initial(self, role: str) -> None:
        """Move OPEN -> ASSIGNED. Called once by Org.submit."""
        self.assigned_role = role
        self._transition(OrgStatus.ASSIGNED)

    def begin(self) -> None:
        """ASSIGNED -> IN_PROGRESS. Called by Role.process at the start."""
        self._transition(OrgStatus.IN_PROGRESS)

    def advance(self, by_role: str, output: dict, next_role: Optional[str]) -> None:
        """IN_PROGRESS -> PASSED -> ASSIGNED(next_role) or COMPLETED."""
        self.history.append(
            {
                "role": by_role,
                "type": "advance",
                "output": output,
                "timestamp": time.time(),
            }
        )
        self._transition(OrgStatus.PASSED)
        if next_role is None:
            self.assigned_role = None
            self._transition(OrgStatus.COMPLETED)
        else:
            self.assigned_role = next_role
            self._transition(OrgStatus.ASSIGNED)

    def reject(self, by_role: str, reason: str, route_back_to: Optional[str]) -> None:
        """IN_PROGRESS -> REJECTED -> ASSIGNED(route_back_to) on first reject, FAILED on second."""
        self.history.append(
            {
                "role": by_role,
                "type": "reject",
                "output": {"verdict": "rejected", "reason": reason},
                "timestamp": time.time(),
            }
        )
        self.artifacts["qa_feedback"] = reason
        self.reject_count += 1
        self._transition(OrgStatus.REJECTED)
        if route_back_to and self.reject_count <= 1:
            self.assigned_role = route_back_to
            self._transition(OrgStatus.ASSIGNED)
        else:
            self.assigned_role = None
            self._transition(OrgStatus.FAILED)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "assigned_role": self.assigned_role,
            "status": self.status.value,
            "history": self.history,
            "artifacts": self.artifacts,
            "reject_count": self.reject_count,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkOrder":
        return cls(
            id=data["id"],
            goal=data["goal"],
            assigned_role=data.get("assigned_role"),
            status=OrgStatus(data["status"]),
            history=list(data.get("history", [])),
            artifacts=dict(data.get("artifacts", {})),
            reject_count=int(data.get("reject_count", 0)),
            parent_id=data.get("parent_id"),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )
