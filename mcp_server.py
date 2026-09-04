
import sys
import uvicorn
from fastmcp import FastMCP
from tools import check_loan_application_status as lookup_loan_status

# Initialize FastMCP Server
mcp_server = FastMCP(
    name="CredLendingMCPServer",
    instructions="Authoritative MCP server for Cred loan underwriting, application status lookup, and escalation risk scoring."
)


@mcp_server.tool()
def check_loan_application_status(record_id: str) -> dict:
    return lookup_loan_status(record_id)


def start_server(host: str = "127.0.0.1", port: int = 8001) -> None:
    print(f"Starting FastMCP Server '{mcp_server.name}' on http://{host}:{port}/mcp ...")
    mcp_server.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    start_server(port=port)
