"""
Timeouts and Exponential Backoff Retries Demonstration
Track: Banking & FinTech (Cred)

Demonstrates:
1. Exponential Backoff Retry Policy:
   - Configured with: max_attempts=4, initial_interval=0.05s, max_interval=0.50s, backoff_factor=2.0, jitter=True.
   - Recovers from simulated transient failures (fails first 2 calls, succeeds on 3rd).
2. Per-Node Timeout:
   - Enforces a 0.20s per-node deadline on a slow external database/API call; raises clean error without hanging.
3. Global Graph Timeout:
   - Enforces a 0.35s end-to-end deadline on entire graph execution; cleanly aborts upon overrun.
"""

import time
import random
import asyncio
import inspect
from typing import Dict, Any, Callable, Optional


class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 4,
        initial_interval: float = 0.05,
        backoff_factor: float = 2.0,
        max_interval: float = 0.50,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_interval = initial_interval
        self.backoff_factor = backoff_factor
        self.max_interval = max_interval
        self.jitter = jitter


async def execute_with_retry_and_timeout(
    operation: Callable[..., Any],
    *args,
    retry_config: Optional[RetryConfig] = None,
    node_timeout: Optional[float] = None,
    **kwargs
) -> Any:
    """
    Executes an async or sync callable with exponential backoff retries and per-node timeout.
    """
    cfg = retry_config or RetryConfig()
    delay = cfg.initial_interval
    last_exception = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            print(f"    -> [Attempt {attempt}/{cfg.max_attempts}] Executing node operation...")
            
            # Per-node timeout wrapper
            if node_timeout is not None:
                if inspect.iscoroutinefunction(operation):
                    result = await asyncio.wait_for(operation(*args, **kwargs), timeout=node_timeout)
                else:
                    # Sync operation wrapped in executor with timeout
                    loop = asyncio.get_running_loop()
                    result = await asyncio.wait_for(loop.run_in_executor(None, lambda: operation(*args, **kwargs)), timeout=node_timeout)
            else:
                if inspect.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)

            print(f"    -> [Attempt {attempt}] Operation SUCCEEDED.")
            return result

        except asyncio.TimeoutError as te:
            print(f"    -> [TIMEOUT ERROR] Node execution exceeded per-node timeout of {node_timeout}s!")
            raise TimeoutError(f"Node execution timed out after {node_timeout}s") from te

        except Exception as exc:
            last_exception = exc
            print(f"    -> [TRANSIENT ERROR on Attempt {attempt}]: {type(exc).__name__}: {str(exc)}")
            if attempt == cfg.max_attempts:
                print(f"    -> Max retry attempts ({cfg.max_attempts}) exhausted.")
                raise exc

            # Calculate backoff with jitter
            sleep_time = min(delay, cfg.max_interval)
            if cfg.jitter:
                sleep_time = sleep_time * random.uniform(0.8, 1.2)
            
            print(f"    -> Backing off for {sleep_time:.3f}s before retry {attempt + 1}...")
            await asyncio.sleep(sleep_time)
            delay *= cfg.backoff_factor

    raise last_exception


# -------------------------------------------------------------
# Demonstration 1: Transient Failure Recovery
# -------------------------------------------------------------
_transient_call_count = 0

def simulate_transient_underwriting_call(record_id: str) -> Dict[str, Any]:
    global _transient_call_count
    _transient_call_count += 1
    if _transient_call_count <= 2:
        raise ConnectionResetError(f"Simulated network reset on upstream credit bureau (Call #{_transient_call_count})")
    
    return {
        "status": "success",
        "record_id": record_id,
        "credit_score": 765,
        "resolved_on_attempt": _transient_call_count
    }


async def demo_retry_recovery():
    global _transient_call_count
    _transient_call_count = 0
    print("\n" + "=" * 75)
    print("DEMO A: EXPONENTIAL BACKOFF RETRY RECOVERY")
    print("Policy: max_attempts=4, initial=0.05s, factor=2.0, max=0.50s, jitter=True")
    print("=" * 75)

    config = RetryConfig(max_attempts=4, initial_interval=0.05, backoff_factor=2.0, max_interval=0.5, jitter=True)
    result = await execute_with_retry_and_timeout(
        simulate_transient_underwriting_call,
        "CRD-APP-1001",
        retry_config=config
    )
    print("\nResult:")
    print(f"  Final Output: {result}")
    assert result["status"] == "success"
    assert result["resolved_on_attempt"] == 3
    print("Verification: Transient failure recovered cleanly on Attempt 3.")


# -------------------------------------------------------------
# Demonstration 2: Per-Node Timeout Clean Error
# -------------------------------------------------------------
def simulate_slow_database_lookup():
    time.sleep(1.0)
    return {"data": "slow_result"}


async def demo_per_node_timeout():
    print("\n" + "=" * 75)
    print("DEMO B: PER-NODE TIMEOUT CLEAN ERROR")
    print("Scenario: Node sleep duration = 1.0s, Configured Per-Node Timeout = 0.20s")
    print("=" * 75)

    config = RetryConfig(max_attempts=1)  # 1 attempt to demonstrate timeout
    try:
        await execute_with_retry_and_timeout(
            simulate_slow_database_lookup,
            retry_config=config,
            node_timeout=0.20
        )
        raise AssertionError("Expected timeout did not occur")
    except TimeoutError as te:
        print(f"\nCaught Expected Clean Error: {te}")
        print("Verification: Per-node timeout raised cleanly without process hanging.")


# -------------------------------------------------------------
# Demonstration 3: Global Graph Timeout Cancellation
# -------------------------------------------------------------
async def run_full_graph_simulation():
    # Multi-step graph simulation totaling ~0.8s
    await asyncio.sleep(0.3)  # Node 1
    await asyncio.sleep(0.3)  # Node 2
    await asyncio.sleep(0.3)  # Node 3
    return {"status": "all_nodes_completed"}


async def demo_global_graph_timeout():
    print("\n" + "=" * 75)
    print("DEMO C: GLOBAL GRAPH TIMEOUT CANCELLATION")
    print("Scenario: Total graph duration = ~0.90s, Global Graph Timeout = 0.35s")
    print("=" * 75)

    global_timeout_seconds = 0.35
    try:
        print(f"Starting whole-graph execution with global deadline {global_timeout_seconds}s...")
        await asyncio.wait_for(run_full_graph_simulation(), timeout=global_timeout_seconds)
        raise AssertionError("Expected global graph timeout did not occur")
    except asyncio.TimeoutError:
        print(f"\n[GLOBAL TIMEOUT TRIGGERED] Graph execution cancelled after exceeding {global_timeout_seconds}s global budget.")
        print("Verification: Global graph timeout cleanly aborted the multi-node workflow.")


async def main():
    print("=" * 75)
    print("RESILIENCE: TIMEOUTS & EXPONENTIAL BACKOFF RETRY SUITE")
    print("=" * 75)
    await demo_retry_recovery()
    await demo_per_node_timeout()
    await demo_global_graph_timeout()
    print("\n" + "=" * 75)
    print("ALL TIMEOUT AND RETRY DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
