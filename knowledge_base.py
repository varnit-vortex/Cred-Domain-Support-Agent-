
import os
import json
from typing import List, Dict, Any

KNOWLEDGE_BASE_DOCS: List[Dict[str, Any]] = [
    {
        "doc_id": "KB-DOC-001",
        "topic": "loan_eligibility_criteria",
        "title": "Loan Eligibility Criteria by Loan Type",
        "content": (
            "Cred evaluates loan eligibility based on monthly net income, debt-to-income (DTI) ratio, and existing financial commitments. "
            "For Personal Loans, salaried applicants must earn at least ₹25,000 monthly with a minimum CIBIL score of 700. "
            "Home and Auto Loans require a minimum stable employment history of two years and a maximum allowable DTI ratio of 50%. "
            "Business Loans require at least three years of audited financial statements with an annual business turnover exceeding ₹20 lakhs."
        )
    },
    {
        "doc_id": "KB-DOC-002",
        "topic": "emi_calculation_rules",
        "title": "Equated Monthly Installment (EMI) Calculation Rules",
        "content": (
            "EMIs are computed using the standard reducing balance method based on the principal amount, tenure in months, and annual interest rate. "
            "The monthly installment is debited automatically via National Automated Clearing House (NACH) or e-mandate on the 5th of each calendar month. "
            "Any bounce or failure in EMI deduction attracts a dishonour fee of ₹500 plus applicable GST along with penal interest of 2% per month on the overdue sum."
        )
    },
    {
        "doc_id": "KB-DOC-003",
        "topic": "credit_card_fee_structure",
        "title": "Credit Card Fee Structure and Billing Schedule",
        "content": (
            "Cred credit cards come with zero joining fees for verified members, while premium tier cards incur an annual fee of ₹1,000 to ₹5,000 depending on variant perks. "
            "The annual fee is waived if the cardholder achieves annual retail spending exceeding ₹1,50,000 in the previous calendar year. "
            "Finance charges on revolving balances are levied at 3.5% per month (42% per annum), with an interest-free grace window extending up to 45 days."
        )
    },
    {
        "doc_id": "KB-DOC-004",
        "topic": "kyc_document_requirements",
        "title": "KYC Document Requirements for Verification",
        "content": (
            "Mandatory Customer Due Diligence requires a valid Permanent Account Number (PAN) along with Aadhaar-based digital biometric authentication. "
            "For address verification where current residential address differs from Aadhaar, acceptable Officially Valid Documents (OVD) include a Passport, Voter ID, or utility bill within three months. "
            "Self-employed business applicants must additionally provide Goods and Services Tax (GST) registration certificates and the previous two years' Income Tax Returns (ITR)."
        )
    },
    {
        "doc_id": "KB-DOC-005",
        "topic": "fraud_dispute_resolution_process",
        "title": "Fraud and Transaction Dispute Resolution Process",
        "content": (
            "Members identifying unauthorized transactions must report them immediately through the Cred app or customer care hotline within 72 hours to ensure zero customer liability. "
            "Upon notification, the affected payment instrument or account is locked instantly, and a provisional dispute credit is credited within five business days pending investigation. "
            "The fraud governance committee investigates merchant dispute logs, IP telemetry, and OTP trails to deliver a final binding resolution within 30 calendar days."
        )
    },
    {
        "doc_id": "KB-DOC-006",
        "topic": "account_closure_process",
        "title": "Account and Line-of-Credit Closure Process",
        "content": (
            "Account closure or cancellation of active credit lines requires total liquidation of all outstanding principal, interest dues, and unbilled charges. "
            "A formal closure request must be initiated via verified email or the in-app support portal, followed by authentication via two-factor OTP. "
            "Once settled, a digitally signed No Objection Certificate (NOC) and full No Dues Certificate (NDC) are issued to the member within seven business days."
        )
    },
    {
        "doc_id": "KB-DOC-007",
        "topic": "interest_rate_slabs",
        "title": "Interest Rate Slabs Across Credit Offerings",
        "content": (
            "Interest rates are risk-tiered based on borrower credit bureau profile, category, and collateral availability. "
            "Personal Loans carry floating annual percentage rates ranging from 10.50% to 18.00%, whereas Secured Home Loans range from 8.40% to 9.75%. "
            "Auto Loans are priced between 8.75% and 11.50%, and commercial Business Loans range from 12.00% to 16.50% based on company balance sheet strength."
        )
    },
    {
        "doc_id": "KB-DOC-008",
        "topic": "prepayment_penalty_rules",
        "title": "Prepayment Penalty and Foreclosure Rules",
        "content": (
            "In strict compliance with Reserve Bank of India (RBI) guidelines, floating-rate personal, home, and auto loans to individual borrowers attract zero prepayment or foreclosure penalties. "
            "Fixed-rate retail loans foreclosed within the first 12 months incur a foreclosure levy of 2% on the outstanding principal balance. "
            "Part-prepayments must equal or exceed an amount equivalent to two EMIs and can be initiated directly without penalty after six completed monthly installments."
        )
    },
    {
        "doc_id": "KB-DOC-009",
        "topic": "minimum_balance_requirements",
        "title": "Minimum Average Balance (MAB) Requirements",
        "content": (
            "Savings accounts maintained under Cred banking partner integrations mandate an Average Monthly Balance (AMB) of ₹10,000 for metro branches and ₹5,000 for semi-urban locations. "
            "Failure to maintain the mandated monthly balance triggers a non-maintenance penalty ranging between ₹150 and ₹400 plus GST per monthly billing cycle. "
            "Salary accounts tied to active corporate payrolls and digital student accounts are completely exempt from minimum balance requirements."
        )
    },
    {
        "doc_id": "KB-DOC-010",
        "topic": "credit_score_impact_factors",
        "title": "Credit Score Impact Factors and Bureau Reporting",
        "content": (
            "Credit scores reflect repayment timeliness (35% weight), credit utilization ratio (30% weight), credit history length (15%), credit mix (10%), and recent hard credit inquiries (10%). "
            "Maintaining credit utilization below 30% of aggregate card limits strongly optimizes positive score progression. "
            "Delinquencies exceeding 30 days are reported to CIBIL, Experian, Equifax, and CRIF High Mark within the first week of every subsequent calendar month."
        )
    },
    {
        "doc_id": "KB-DOC-011",
        "topic": "joint_account_rules",
        "title": "Joint Account Mandates and Operational Rules",
        "content": (
            "Joint accounts can be created with spouse, parents, or adult siblings under 'Either or Survivor', 'Former or Survivor', or 'Jointly Operated' operational mandates. "
            "All primary co-borrowers share equal joint-and-several financial liability for any credit facilities or overdrafts linked to the account. "
            "Mandate modifications, account conversions, or credit closure requests require unanimous written authorization and biometric/OTP approval from all registered account holders."
        )
    },
    {
        "doc_id": "KB-DOC-012",
        "topic": "nri_account_eligibility",
        "title": "Non-Resident Indian (NRI) Account Eligibility",
        "content": (
            "Non-Resident Indians (NRIs) and Persons of Indian Origin (PIOs) are eligible to open Non-Resident External (NRE) and Non-Resident Ordinary (NRO) accounts. "
            "NRE accounts permit full repatriability of both principal and interest earnings with total Indian income tax exemption on accrued interest. "
            "NRO accounts are mandatory for managing rupee income earned inside India (such as rent or pensions) and are subject to Indian withholding tax and FEMA remittance ceilings."
        )
    }
]


def export_knowledge_base_files(export_dir: str = "data/knowledge_base") -> None:
    os.makedirs(export_dir, exist_ok=True)
    
    # Export full JSON
    json_path = os.path.join(export_dir, "knowledge_base.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(KNOWLEDGE_BASE_DOCS, f, indent=2, ensure_ascii=False)
    
    # Export individual markdown files
    for doc in KNOWLEDGE_BASE_DOCS:
        md_filename = f"{doc['doc_id']}_{doc['topic']}.md"
        md_path = os.path.join(export_dir, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {doc['title']}\n\n")
            f.write(f"**Document ID:** {doc['doc_id']}\n")
            f.write(f"**Topic:** `{doc['topic']}`\n\n")
            f.write(f"{doc['content']}\n")

    print(f"Exported {len(KNOWLEDGE_BASE_DOCS)} knowledge base documents to {export_dir}/")


if __name__ == "__main__":
    export_knowledge_base_files()
