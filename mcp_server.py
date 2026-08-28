"""
Model Context Protocol (FastMCP) Server for Cred Lending Operations
Track: Banking & FinTech (Cred)

Exposes `check_loan_application_status` tool following the standard Model Context Protocol.
Allows external MCP clients to retrieve application status, loan category, amount, and escalation scores.
"""

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
    """
    Looks up a loan application record by ID, evaluates status, and computes escalation score.
    
    Args:
        record_id (str): Unique loan application identifier (e.g., 'CRD-APP-1001')
        
    Returns:
        dict: A dictionary containing:
            - found (bool): Whether the record was located in the database
            - record_id (str): Application ID
            - category (str): Loan type (Personal, Home, Auto, Education, Business)
            - status (str): Current status (Submitted, Under Review, Approved, Rejected, Disbursed)
            - loan_amount_inr (int): Loan amount in INR
            - days_since_created (int): Elapsed days since application submission
            - flagged_for_fraud_review (bool): Indicator if fraud review is active
            - escalation_score (float): Computed continuous escalation score in [0.0, 1.0]
            - escalation_recommended (bool): Whether score meets or exceeds 0.65 threshold
            - escalation_reason (str): Underwriting justification
    """
    return lookup_loan_status(record_id)


def start_server(host: str = "127.0.0.1", port: int = 8001) -> None:
    """
    Runs the MCP server using FastMCP HTTP transport (mounted at /mcp).
    """
    print(f"Starting FastMCP Server '{mcp_server.name}' on http://{host}:{port}/mcp ...")
    mcp_server.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    start_server(port=port)
