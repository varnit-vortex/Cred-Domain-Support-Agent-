
import asyncio
import json
from typing import List
from fastmcp.client import Client
from mcp_server import mcp_server


async def run_mcp_client_demonstration(record_ids: List[str] = None) -> None:
    if record_ids is None:
        record_ids = ["CRD-APP-1001", "CRD-APP-1002", "CRD-APP-1004", "CRD-APP-9999"]

    print("=" * 75)
    print("STANDALONE MCP CLIENT DEMONSTRATION")
    print("=" * 75)

    # Initialize client with server instance or endpoint
    client = Client(mcp_server)

    async with client:
        # 1. Discover tools
        tools = await client.list_tools()
        print("\n[MCP Tool Discovery]")
        print(f"Connected to MCP Server: '{mcp_server.name}'")
        print(f"Available Tools: {[t.name for t in tools]}")
        for t in tools:
            print(f"  - Tool Name: {t.name}")
            print(f"    Description: {t.description.strip().splitlines()[0] if t.description else 'N/A'}")

        # 2. Invoke tool for each record ID
        print("\n[Executing Standardized MCP Tool Calls]")
        for rid in record_ids:
            print(f"\n---> Calling tool 'check_loan_application_status' with record_id='{rid}'...")
            result = await client.call_tool("check_loan_application_status", {"record_id": rid})
            
            print("Standardized MCP Tool Response:")
            print(f"  - is_error: {result.is_error}")
            print(f"  - content : {result.content}")
            print(f"  - data    : {json.dumps(result.data, indent=4)}")

    print("\n" + "=" * 75)
    print("MCP CLIENT-SERVER ROUND-TRIP COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_mcp_client_demonstration())
