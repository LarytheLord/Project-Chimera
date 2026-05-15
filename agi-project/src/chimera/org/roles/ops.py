from ..role import Role


class OpsRole(Role):
    name = "Ops"
    allowed_tools = ()
    artifact_key = "ops_plan"
    next_role = "QA"
