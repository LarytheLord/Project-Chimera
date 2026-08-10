"""Crash-and-resume: simulate a failure mid-flow and verify Org.resume() finishes the job."""

import json
import os
import pytest

from chimera.org.org import Org
from chimera.org.work_order import OrgStatus


def test_crash_after_rnd_then_resume(happy_core, tmp_db_root):
    """Simulate the Marketing role raising on its first invocation. The persisted state
    should still show ASSIGNED to Marketing (last successful save was after R&D).
    Then on resume, allow Marketing through and verify the flow completes."""

    org = Org.default(cognitive_core=happy_core, db_root=tmp_db_root)

    original_generate = happy_core.generate_response
    crash_state = {"crashed": False}

    def patched_generate(inputs, **kwargs):
        prompt = inputs.get("text_data", "")
        if "[[ROLE:Marketing]]" in prompt and not crash_state["crashed"]:
            crash_state["crashed"] = True
            raise RuntimeError("simulated crash during Marketing")
        return original_generate(inputs, **kwargs)

    happy_core.generate_response = patched_generate  # type: ignore[method-assign]

    wo = org.submit("crash test")
    with pytest.raises(RuntimeError, match="simulated crash"):
        org.run_until_complete(wo.id)

    # After the crash, the persisted store should show the WO still ASSIGNED to Marketing
    # (the role from which the run was attempted) because the save after R&D's advance
    # already routed it there.
    reloaded = org.store.load(wo.id)
    assert reloaded is not None
    assert reloaded.status == OrgStatus.ASSIGNED
    assert reloaded.assigned_role == "Marketing"
    # CEO and R&D both ran before the crash.
    roles_so_far = [h["role"] for h in reloaded.history]
    assert roles_so_far == ["CEO", "RnD"]

    # Now resume: Marketing should run successfully on the second attempt and the flow completes.
    finals = org.resume(max_hops=20)
    assert len(finals) == 1
    final = finals[0]
    assert final.status == OrgStatus.COMPLETED
    roles_visited = [h["role"] for h in final.history]
    assert roles_visited == ["CEO", "RnD", "Marketing", "Production", "Ops", "QA"]


def test_resume_with_no_active_workorders(happy_core, tmp_db_root):
    """resume() on an empty store returns an empty list, no errors."""
    org = Org.default(cognitive_core=happy_core, db_root=tmp_db_root)
    finals = org.resume()
    assert finals == []
