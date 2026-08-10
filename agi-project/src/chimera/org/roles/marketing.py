from ..role import Role


class MarketingRole(Role):
    name = "Marketing"
    allowed_tools = ()
    artifact_key = "positioning"
    next_role = "Production"
