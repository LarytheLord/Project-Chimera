from ..role import Role


class RnDRole(Role):
    name = "RnD"
    allowed_tools = ("web_search", "file_system")
    artifact_key = "research"
    next_role = "Marketing"
