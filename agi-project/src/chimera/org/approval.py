"""ApprovalGate: Phase 1 is a pass-through stub. Phase 2 enforces human-in-the-loop."""

from __future__ import annotations

from typing import Any, Callable, Optional


class ApprovalRequired(Exception):
    """Raised by a gated tool when human approval is required before execution.

    The Org catches this and parks the WorkOrder in AWAITING_APPROVAL. Phase 2+ uses it;
    Phase 1 keeps the gate permissive so the demo runs end-to-end without prompts.
    """

    def __init__(self, action_description: str, payload: Any) -> None:
        super().__init__(action_description)
        self.action_description = action_description
        self.payload = payload


class ApprovalGate:
    """Wraps a "dangerous" callable. Phase 1: pass-through. Phase 2: enforce."""

    def __init__(self, approver: Optional[Callable[[str, Any], bool]] = None, enforce: bool = False):
        self.approver = approver
        self.enforce = enforce

    def guard(self, action_description: str, payload: Any, execute: Callable[[], Any]) -> Any:
        if not self.enforce:
            return execute()
        if self.approver is not None and self.approver(action_description, payload):
            return execute()
        raise ApprovalRequired(action_description, payload)
