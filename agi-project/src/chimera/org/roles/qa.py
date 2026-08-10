from ..role import Role
from ..work_order import WorkOrder


class QARole(Role):
    name = "QA"
    allowed_tools = ("file_system",)
    artifact_key = "qa_verdict"
    next_role = None
    route_back_to = "Production"

    def _apply(self, wo: WorkOrder, parsed: dict) -> WorkOrder:
        output = parsed.get("output", {}) or {}
        verdict = str(output.get("verdict", "approved")).strip().lower()
        wo.artifacts[self.artifact_key] = output
        if verdict in ("rejected", "reject", "fail", "failed"):
            wo.reject(
                by_role=self.name,
                reason=str(output.get("reason", "no reason given")),
                route_back_to=self.route_back_to,
            )
        else:
            wo.advance(by_role=self.name, output=output, next_role=None)
        return wo
