
import pytest
import asyncio
from fastmcp.client import Client
from mcp_server import mcp_server
from resilience_checkpoint import demonstrate_sqlite_checkpointing
from resilience_retries import (
    execute_with_retry_and_timeout,
    simulate_transient_underwriting_call,
    simulate_slow_database_lookup,
    RetryConfig
)


def test_mcp_client_tool_call():
    async def run_test():
        client = Client(mcp_server)
        async with client:
            tools = await client.list_tools()
            assert any(t.name == "check_loan_application_status" for t in tools)
            
            # Test calling tool on 2 record IDs
            res1 = await client.call_tool("check_loan_application_status", {"record_id": "CRD-APP-1001"})
            assert res1.is_error is False
            assert res1.data["found"] is True
            assert res1.data["record_id"] == "CRD-APP-1001"

            res2 = await client.call_tool("check_loan_application_status", {"record_id": "CRD-APP-1002"})
            assert res2.is_error is False
            assert res2.data["found"] is True
            assert res2.data["record_id"] == "CRD-APP-1002"

    asyncio.run(run_test())


def test_sqlite_checkpoint_resumption(tmp_path):
    test_db = str(tmp_path / "test_cp.sqlite")
    demonstrate_sqlite_checkpointing(db_path=test_db)


def test_exponential_backoff_recovery():
    async def run_test():
        import resilience_retries
        resilience_retries._transient_call_count = 0

        cfg = RetryConfig(max_attempts=4, initial_interval=0.01, backoff_factor=2.0, max_interval=0.1)
        result = await execute_with_retry_and_timeout(
            simulate_transient_underwriting_call,
            "CRD-APP-1001",
            retry_config=cfg
        )
        assert result["status"] == "success"
        assert result["resolved_on_attempt"] == 3

    asyncio.run(run_test())


def test_per_node_timeout_error():
    async def run_test():
        cfg = RetryConfig(max_attempts=1)
        with pytest.raises(TimeoutError):
            await execute_with_retry_and_timeout(
                simulate_slow_database_lookup,
                retry_config=cfg,
                node_timeout=0.10
            )

    asyncio.run(run_test())
