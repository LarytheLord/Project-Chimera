"""Each role should only see the tools its allowed_tools list grants it."""

import pytest

from chimera.org.roles import (
    CEORole,
    MarketingRole,
    OpsRole,
    ProductionRole,
    QARole,
    RnDRole,
)


@pytest.mark.parametrize(
    "role_cls,expected_tools",
    [
        (CEORole, set()),
        (RnDRole, {"web_search", "file_system"}),
        (MarketingRole, set()),
        (ProductionRole, {"file_system"}),
        (OpsRole, set()),
        (QARole, {"file_system"}),
    ],
)
def test_role_tool_grants(happy_core, tmp_db_root, role_cls, expected_tools):
    role = role_cls(cognitive_core=happy_core, db_root=tmp_db_root)
    actual = set(role.agent.tool_registry.get_tool_names())
    assert actual == expected_tools, (
        f"{role_cls.__name__} expected tools {expected_tools}, got {actual}"
    )


def test_ceo_has_no_web_search(happy_core, tmp_db_root):
    """Regression guard: Agent.__init__ auto-registers WebSearchTool. Role must strip it."""
    role = CEORole(cognitive_core=happy_core, db_root=tmp_db_root)
    assert "web_search" not in role.agent.tool_registry.get_tool_names()


def test_rnd_keeps_web_search(happy_core, tmp_db_root):
    role = RnDRole(cognitive_core=happy_core, db_root=tmp_db_root)
    assert "web_search" in role.agent.tool_registry.get_tool_names()
