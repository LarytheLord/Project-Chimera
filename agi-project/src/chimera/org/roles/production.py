from ..role import Role


class ProductionRole(Role):
    name = "Production"
    allowed_tools = ("file_system",)
    artifact_key = "deliverable"
    next_role = "Ops"
