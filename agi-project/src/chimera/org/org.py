"""Org: the dispatcher that owns roles and drives WorkOrders to completion."""

from __future__ import annotations

import os
from typing import Iterable, Literal, Optional

from ..cognitive_core.interfaces import CognitiveCore
from .role import Role
from .store import WorkOrderStore
from .work_order import OrgStatus, WorkOrder


class Org:
    """Owns the role registry, persists WorkOrders, drives the sequential flow."""

    def __init__(
        self,
        roles: Iterable[Role],
        store: WorkOrderStore,
        process: Literal["sequential"] = "sequential",
        first_role: str = "CEO",
    ):
        self.roles: dict[str, Role] = {r.name: r for r in roles}
        self.store = store
        self.process = process
        self.first_role = first_role
        if first_role not in self.roles:
            raise ValueError(
                f"first_role={first_role!r} not in roles {list(self.roles)}"
            )

    @classmethod
    def default(
        cls,
        cognitive_core: CognitiveCore,
        db_root: str,
        rlhf_oracle=None,
    ) -> "Org":
        """Construct the standard 6-role sequential org."""
        from .roles import all_roles

        os.makedirs(db_root, exist_ok=True)
        roles = [
            cls_(cognitive_core=cognitive_core, db_root=db_root, rlhf_oracle=rlhf_oracle)
            for cls_ in all_roles()
        ]
        store = WorkOrderStore(db_path=os.path.join(db_root, "org.sqlite3"))
        return cls(roles=roles, store=store, process="sequential", first_role="CEO")

    def submit(self, goal: str) -> WorkOrder:
        wo = WorkOrder(goal=goal)
        wo.assign_initial(self.first_role)
        self.store.save(wo)
        return wo

    def run_until_complete(self, wo_id: str, max_hops: int = 12) -> WorkOrder:
        wo = self.store.load(wo_id)
        if wo is None:
            raise KeyError(f"No WorkOrder with id={wo_id!r}")
        hops = 0
        while wo.status == OrgStatus.ASSIGNED and hops < max_hops:
            role_name = wo.assigned_role
            if role_name not in self.roles:
                raise KeyError(
                    f"WorkOrder {wo.id} assigned to unknown role {role_name!r}"
                )
            role = self.roles[role_name]
            wo = role.process(wo)
            self.store.save(wo)
            hops += 1
            if wo.status in (
                OrgStatus.COMPLETED,
                OrgStatus.FAILED,
                OrgStatus.AWAITING_APPROVAL,
            ):
                break
        return wo

    def resume(self, max_hops: int = 12) -> list[WorkOrder]:
        """Resume every active WorkOrder. Returns the final states."""
        active = self.store.list_active()
        final = []
        for wo in active:
            if wo.status == OrgStatus.ASSIGNED:
                final.append(self.run_until_complete(wo.id, max_hops=max_hops))
            else:
                final.append(wo)
        return final
