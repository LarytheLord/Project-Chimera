"""Unit tests for WorkOrder state machine."""

import pytest

from chimera.org.work_order import IllegalTransition, OrgStatus, WorkOrder


def test_initial_state():
    wo = WorkOrder(goal="hello")
    assert wo.status == OrgStatus.OPEN
    assert wo.assigned_role is None
    assert wo.history == []
    assert wo.reject_count == 0


def test_happy_path_transitions():
    wo = WorkOrder(goal="g")
    wo.assign_initial("CEO")
    assert wo.status == OrgStatus.ASSIGNED
    assert wo.assigned_role == "CEO"

    wo.begin()
    assert wo.status == OrgStatus.IN_PROGRESS

    wo.advance(by_role="CEO", output={"brief": "x"}, next_role="RnD")
    assert wo.status == OrgStatus.ASSIGNED
    assert wo.assigned_role == "RnD"
    assert len(wo.history) == 1
    assert wo.history[0]["role"] == "CEO"


def test_final_advance_completes():
    wo = WorkOrder(goal="g")
    wo.assign_initial("QA")
    wo.begin()
    wo.advance(by_role="QA", output={"verdict": "approved"}, next_role=None)
    assert wo.status == OrgStatus.COMPLETED
    assert wo.assigned_role is None


def test_illegal_transition_raises():
    wo = WorkOrder(goal="g")
    wo.assign_initial("CEO")
    wo.begin()
    wo.advance(by_role="CEO", output={}, next_role=None)
    assert wo.status == OrgStatus.COMPLETED
    # Cannot move out of COMPLETED.
    with pytest.raises(IllegalTransition):
        wo.begin()


def test_begin_from_open_is_illegal():
    wo = WorkOrder(goal="g")
    with pytest.raises(IllegalTransition):
        wo.begin()


def test_first_reject_routes_back():
    wo = WorkOrder(goal="g")
    wo.assign_initial("QA")
    wo.begin()
    wo.reject(by_role="QA", reason="missing detail", route_back_to="Production")
    assert wo.status == OrgStatus.ASSIGNED
    assert wo.assigned_role == "Production"
    assert wo.reject_count == 1
    assert wo.artifacts["qa_feedback"] == "missing detail"


def test_second_reject_fails():
    wo = WorkOrder(goal="g")
    wo.assign_initial("QA")
    wo.begin()
    wo.reject(by_role="QA", reason="r1", route_back_to="Production")
    # Production re-runs and we hit QA again.
    wo.begin()
    wo.reject(by_role="QA", reason="r2", route_back_to="Production")
    assert wo.status == OrgStatus.FAILED
    assert wo.reject_count == 2


def test_serialize_round_trip():
    wo = WorkOrder(goal="round trip me")
    wo.assign_initial("CEO")
    wo.begin()
    wo.advance(by_role="CEO", output={"brief": "x"}, next_role="RnD")

    payload = wo.to_dict()
    restored = WorkOrder.from_dict(payload)

    assert restored.id == wo.id
    assert restored.goal == wo.goal
    assert restored.status == wo.status
    assert restored.assigned_role == wo.assigned_role
    assert restored.history == wo.history
    assert restored.artifacts == wo.artifacts
