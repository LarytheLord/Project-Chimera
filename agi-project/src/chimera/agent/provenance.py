# Provenance and audit trail system
# Records what ran, with what inputs, what outputs, and what failed

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import hashlib


@dataclass
class ExecutionRecord:
    """A single execution record for provenance tracking."""
    run_id: str
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "tool_name": self.tool_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "metadata": self.metadata
        }
    
    def compute_hash(self) -> str:
        """Compute a cryptographic hash of this execution record."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class TaskRun:
    """Tracks a complete task run with state, logs, and provenance."""
    task_id: str
    task_type: str  # e.g., "tool_execution", "digest", "risk_snapshot"
    status: str = "pending"  # pending, running, completed, failed, cancelled
    executions: List[ExecutionRecord] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_execution(self, record: ExecutionRecord):
        """Add an execution record."""
        self.executions.append(record)
    
    def add_log(self, message: str):
        """Add a log message."""
        self.logs.append(message)
    
    def update_progress(self, progress: float):
        """Update progress (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))
    
    def start(self):
        """Mark the task as started."""
        self.status = "running"
        self.started_at = datetime.now()
    
    def complete(self):
        """Mark the task as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        self.progress = 1.0
    
    def fail(self, error: str):
        """Mark the task as failed."""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.error = error
    
    def cancel(self):
        """Mark the task as cancelled."""
        self.status = "cancelled"
        self.completed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "executions": [e.to_dict() for e in self.executions],
            "logs": self.logs,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata
        }


class AuditLogger:
    """Central audit logger for provenance tracking."""
    
    def __init__(self):
        self._task_runs: Dict[str, TaskRun] = {}
        self._execution_log: List[ExecutionRecord] = []
    
    def create_task_run(self, task_id: str, task_type: str, **kwargs) -> TaskRun:
        """Create a new task run."""
        task_run = TaskRun(
            task_id=task_id,
            task_type=task_type,
            **kwargs
        )
        self._task_runs[task_id] = task_run
        return task_run
    
    def get_task_run(self, task_id: str) -> Optional[TaskRun]:
        """Get a task run by ID."""
        return self._task_runs.get(task_id)
    
    def list_task_runs(self, task_type: Optional[str] = None) -> List[TaskRun]:
        """List task runs, optionally filtered by type."""
        if task_type:
            return [tr for tr in self._task_runs.values() if tr.task_type == task_type]
        return list(self._task_runs.values())
    
    def log_execution(self, record: ExecutionRecord):
        """Log a tool execution."""
        self._execution_log.append(record)
    
    def get_execution_log(self, tool_name: Optional[str] = None) -> List[ExecutionRecord]:
        """Get execution log, optionally filtered by tool."""
        if tool_name:
            return [r for r in self._execution_log if r.tool_name == tool_name]
        return self._execution_log.copy()
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of all runs."""
        return {
            "total_task_runs": len(self._task_runs),
            "completed": len([tr for tr in self._task_runs.values() if tr.status == "completed"]),
            "failed": len([tr for tr in self._task_runs.values() if tr.status == "failed"]),
            "running": len([tr for tr in self._task_runs.values() if tr.status == "running"]),
            "total_executions": len(self._execution_log),
            "task_runs": {tid: tr.to_dict() for tid, tr in self._task_runs.items()}
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit logger state."""
        return {
            "task_runs": {tid: tr.to_dict() for tid, tr in self._task_runs.items()},
            "execution_log": [r.to_dict() for r in self._execution_log]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogger":
        """Deserialize audit logger state."""
        logger = cls()
        # Restore task runs
        for task_id, task_data in data.get("task_runs", {}).items():
            task_run = TaskRun(
                task_id=task_data["task_id"],
                task_type=task_data["task_type"],
                status=task_data["status"],
                logs=task_data.get("logs", []),
                progress=task_data.get("progress", 0.0),
                error=task_data.get("error"),
                metadata=task_data.get("metadata", {})
            )
            if task_data.get("started_at"):
                task_run.started_at = datetime.fromisoformat(task_data["started_at"])
            if task_data.get("completed_at"):
                task_run.completed_at = datetime.fromisoformat(task_data["completed_at"])
            logger._task_runs[task_id] = task_run
        
        # Restore execution log
        for exec_data in data.get("execution_log", []):
            record = ExecutionRecord(
                run_id=exec_data["run_id"],
                tool_name=exec_data["tool_name"],
                inputs=exec_data["inputs"],
                outputs=exec_data.get("outputs"),
                error=exec_data.get("error"),
                success=exec_data.get("success", False),
                metadata=exec_data.get("metadata", {})
            )
            if exec_data.get("start_time"):
                record.start_time = datetime.fromisoformat(exec_data["start_time"])
            if exec_data.get("end_time"):
                record.end_time = datetime.fromisoformat(exec_data["end_time"])
            logger._execution_log.append(record)
        
        return logger
