"""End-to-end sequential org tests: 6-role happy path and reject-and-retry."""

import os

from chimera.org.org import Org
from chimera.org.work_order import OrgStatus


def test_six_role_happy_path(happy_core, tmp_db_root):
    org = Org.default(cognitive_core=happy_core, db_root=tmp_db_root)
    wo = org.submit("Draft a 3-paragraph announcement for vector embeddings")
    final = org.run_until_complete(wo.id)

    assert final.status == OrgStatus.COMPLETED
    assert final.assigned_role is None

    # Each of the six roles touched the WorkOrder exactly once.
    roles_visited = [entry["role"] for entry in final.history]
    assert roles_visited == ["CEO", "RnD", "Marketing", "Production", "Ops", "QA"]

    # All expected artifacts are present.
    for key in ("brief", "research", "positioning", "deliverable", "ops_plan", "qa_verdict"):
        assert key in final.artifacts


def test_qa_reject_then_approve(happy_core, tmp_db_root):
    """QA rejects on first pass; Production redoes; QA approves on second pass."""
    org = Org.default(cognitive_core=happy_core, db_root=tmp_db_root)

    # State machine for the QA mock: reject first, approve second.
    call_state = {"qa_calls": 0}
    original_generate = happy_core.generate_response

    def patched_generate(inputs, **kwargs):
        prompt = inputs.get("text_data", "")
        if "[[ROLE:QA]]" in prompt:
            call_state["qa_calls"] += 1
            if call_state["qa_calls"] == 1:
                import json
                return json.dumps(
                    {
                        "output": {
                            "verdict": "rejected",
                            "reason": "Production output too short",
                        },
                        "next_role": None,
                    }
                )
        return original_generate(inputs, **kwargs)

    happy_core.generate_response = patched_generate  # type: ignore[method-assign]

    wo = org.submit("Write a launch tweet")
    final = org.run_until_complete(wo.id, max_hops=20)

    assert final.status == OrgStatus.COMPLETED
    assert call_state["qa_calls"] == 2
    # Production appeared twice (original + revision), QA twice (reject + approve).
    roles_visited = [entry["role"] for entry in final.history]
    assert roles_visited.count("Production") == 2
    assert roles_visited.count("QA") == 2


def test_double_reject_fails(happy_core, tmp_db_root):
    """QA rejects twice -> WorkOrder marked FAILED."""
    happy_core.set_response(
        "QA",
        {
            "output": {"verdict": "rejected", "reason": "still not good"},
            "next_role": None,
        },
    )
    org = Org.default(cognitive_core=happy_core, db_root=tmp_db_root)
    wo = org.submit("Goal that will fail")
    final = org.run_until_complete(wo.id, max_hops=20)

    assert final.status == OrgStatus.FAILED
    assert final.reject_count == 2


def test_org_persists_to_sqlite(happy_core, tmp_db_root):
    """After running, the SQLite store should contain the completed WorkOrder."""
    org = Org.default(cognitive_core=happy_core, db_root=tmp_db_root)
    wo = org.submit("persist me")
    final = org.run_until_complete(wo.id)

    # Reload from a brand-new store pointing at the same file.
    from chimera.org.store import WorkOrderStore

    store2 = WorkOrderStore(db_path=os.path.join(tmp_db_root, "org.sqlite3"))
    reloaded = store2.load(final.id)
    assert reloaded is not None
    assert reloaded.status == OrgStatus.COMPLETED
    assert len(reloaded.history) == 6
