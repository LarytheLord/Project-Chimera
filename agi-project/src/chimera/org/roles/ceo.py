from ..role import Role


class CEORole(Role):
    name = "CEO"
    allowed_tools = ()
    artifact_key = "brief"
    next_role = "RnD"
