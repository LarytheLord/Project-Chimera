# Tests for provenance and audit trail system

import pytest
from datetime import datetime
from chimera.agent.provenance import (
    ExecutionRecord,
    TaskRun,
    AuditLogger
)


class TestExecutionRecord:
    def test_create_record(self):
        record = ExecutionRecord(
            run_id="test-123",
            tool_name="web_search",
            inputs={"query": "test"}
        )
        assert record.run_id == "test-123"
        assert record.tool_name == "web_search"
        assert record.success == False
    
    def test_successful_execution(self):
        record = ExecutionRecord(
            run_id="test-123",
            tool_name="web_search",
            inputs={"query": "test"},
            outputs="Results found",
            success=True
        )
        assert record.success == True
        assert record.outputs == "Results found"
    
    def test_failed_execution(self):
        record = ExecutionRecord(
            run_id="test-123",
            tool_name="web_search",
            inputs={"query": "test"},
            error="Network error",
            success=False
        )
        assert record.error == "Network error"
        assert record.success == False
    
    def test_compute_hash(self):
        record = ExecutionRecord(
            run_id="test-123",
            tool_name="web_search",
            inputs={"query": "test"}
        )
        hash1 = record.compute_hash()
        hash2 = record.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hash length
    
    def test_serialization(self):
        record = ExecutionRecord(
            run_id="test-123",
            tool_name="web_search",
            inputs={"query": "test"},
            outputs="Results",
            success=True
        )
        data = record.to_dict()
        assert data["run_id"] == "test-123"
        assert data["tool_name"] == "web_search"


class TestTaskRun:
    def test_create_task_run(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        assert task.status == "pending"
        assert task.progress == 0.0
    
    def test_task_lifecycle(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        task.start()
        assert task.status == "running"
        assert task.started_at is not None
        
        task.complete()
        assert task.status == "completed"
        assert task.completed_at is not None
        assert task.progress == 1.0
    
    def test_task_failure(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        task.start()
        task.fail("Something went wrong")
        assert task.status == "failed"
        assert task.error == "Something went wrong"
        assert task.completed_at is not None
    
    def test_task_cancellation(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        task.start()
        task.cancel()
        assert task.status == "cancelled"
    
    def test_progress_update(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        task.update_progress(0.5)
        assert task.progress == 0.5
        task.update_progress(1.5)  # Should cap at 1.0
        assert task.progress == 1.0
        task.update_progress(-0.5)  # Should floor at 0.0
        assert task.progress == 0.0
    
    def test_add_execution(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        record = ExecutionRecord(
            run_id="exec-1",
            tool_name="web_search",
            inputs={"query": "test"}
        )
        task.add_execution(record)
        assert len(task.executions) == 1
        assert task.executions[0].run_id == "exec-1"
    
    def test_add_log(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        task.add_log("Starting task")
        task.add_log("Processing...")
        assert len(task.logs) == 2
        assert task.logs[0] == "Starting task"
    
    def test_serialization(self):
        task = TaskRun(
            task_id="task-1",
            task_type="tool_execution"
        )
        task.start()
        task.complete()
        data = task.to_dict()
        assert data["task_id"] == "task-1"
        assert data["status"] == "completed"


class TestAuditLogger:
    def test_create_task_run(self):
        logger = AuditLogger()
        task = logger.create_task_run("task-1", "tool_execution")
        assert task.task_id == "task-1"
        assert logger.get_task_run("task-1") == task
    
    def test_log_execution(self):
        logger = AuditLogger()
        record = ExecutionRecord(
            run_id="exec-1",
            tool_name="web_search",
            inputs={"query": "test"},
            success=True
        )
        logger.log_execution(record)
        log = logger.get_execution_log()
        assert len(log) == 1
        assert log[0].run_id == "exec-1"
    
    def test_filter_by_tool(self):
        logger = AuditLogger()
        logger.log_execution(ExecutionRecord(
            run_id="exec-1",
            tool_name="web_search",
            inputs={},
            success=True
        ))
        logger.log_execution(ExecutionRecord(
            run_id="exec-2",
            tool_name="file_system",
            inputs={},
            success=True
        ))
        web_logs = logger.get_execution_log(tool_name="web_search")
        assert len(web_logs) == 1
        assert web_logs[0].tool_name == "web_search"
    
    def test_list_task_runs(self):
        logger = AuditLogger()
        logger.create_task_run("task-1", "tool_execution")
        logger.create_task_run("task-2", "digest")
        logger.create_task_run("task-3", "tool_execution")
        all_runs = logger.list_task_runs()
        assert len(all_runs) == 3
        tool_runs = logger.list_task_runs(task_type="tool_execution")
        assert len(tool_runs) == 2
    
    def test_run_summary(self):
        logger = AuditLogger()
        task1 = logger.create_task_run("task-1", "tool_execution")
        task1.start()
        task1.complete()
        
        task2 = logger.create_task_run("task-2", "tool_execution")
        task2.start()
        task2.fail("Error occurred")
        
        logger.log_execution(ExecutionRecord(
            run_id="exec-1",
            tool_name="web_search",
            inputs={},
            success=True
        ))
        
        summary = logger.get_run_summary()
        assert summary["total_task_runs"] == 2
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["total_executions"] == 1
    
    def test_serialization(self):
        logger = AuditLogger()
        task = logger.create_task_run("task-1", "tool_execution")
        task.start()
        task.complete()
        logger.log_execution(ExecutionRecord(
            run_id="exec-1",
            tool_name="web_search",
            inputs={"query": "test"},
            success=True
        ))
        
        data = logger.to_dict()
        restored = AuditLogger.from_dict(data)
        
        assert len(restored.get_task_run("task-1").executions) == 0
        assert len(restored.get_execution_log()) == 1
        assert restored.get_execution_log()[0].run_id == "exec-1"
