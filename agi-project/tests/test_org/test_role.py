"""Per-role tests: each Role.process produces the right artifact and handoff."""

import pytest

from chimera.org.roles import (
    CEORole,
    MarketingRole,
    OpsRole,
    ProductionRole,
    QARole,
    RnDRole,
)
from chimera.org.work_order import OrgStatus, WorkOrder


@pytest.fixture
def make_wo_for(happy_core, tmp_db_root):
    """Factory: returns a (role_instance, work_order_in_assigned_state) for a given role class."""

    def _make(role_cls):
        role = role_cls(cognitive_core=happy_core, db_root=tmp_db_root)
        wo = WorkOrder(goal="test goal")
        wo.assign_initial(role.name)
        return role, wo

    return _make


@pytest.mark.parametrize(
    "role_cls,expected_artifact_key,expected_next",
    [
        (CEORole, "brief", "RnD"),
        (RnDRole, "research", "Marketing"),
        (MarketingRole, "positioning", "Production"),
        (ProductionRole, "deliverable", "Ops"),
        (OpsRole, "ops_plan", "QA"),
    ],
)
def test_role_advances_with_expected_artifact(
    make_wo_for, role_cls, expected_artifact_key, expected_next
):
    role, wo = make_wo_for(role_cls)
    role.process(wo)
    assert expected_artifact_key in wo.artifacts
    assert wo.status == OrgStatus.ASSIGNED
    assert wo.assigned_role == expected_next
    assert wo.history[-1]["role"] == role.name


def test_qa_approval_completes(make_wo_for):
    role, wo = make_wo_for(QARole)
    role.process(wo)
    assert wo.status == OrgStatus.COMPLETED
    assert wo.artifacts["qa_verdict"]["verdict"] == "approved"


def test_qa_rejection_routes_to_production(happy_core, tmp_db_root):
    happy_core.set_response(
        "QA",
        {
            "output": {"verdict": "rejected", "reason": "not enough detail"},
            "next_role": None,
        },
    )
    role = QARole(cognitive_core=happy_core, db_root=tmp_db_root)
    wo = WorkOrder(goal="g")
    wo.assign_initial("QA")
    role.process(wo)
    assert wo.status == OrgStatus.ASSIGNED
    assert wo.assigned_role == "Production"
    assert wo.reject_count == 1
    assert wo.artifacts["qa_feedback"] == "not enough detail"


def test_role_handles_malformed_response(happy_core, tmp_db_root):
    """If the LLM returns non-JSON, the role should not crash -- it should record an error and advance."""

    class BrokenCore:
        def generate_response(self, inputs, **kwargs):
            return "this is not json at all"

        def load_model(self, p): pass
        def train(self, d): pass
        def get_state(self): return None

    role = CEORole(cognitive_core=BrokenCore(), db_root=tmp_db_root)
    wo = WorkOrder(goal="g")
    wo.assign_initial("CEO")
    role.process(wo)
    # Should still have advanced to next role; output captured the parsing error.
    assert wo.status == OrgStatus.ASSIGNED
    assert wo.assigned_role == "RnD"
    assert "error" in wo.artifacts["brief"]
