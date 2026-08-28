"""
Universal LLM Engine: Real LLMs + Local Pipelines + Deterministic Mode
Track: Banking & FinTech (Cred)

Persona:
- Highly polite, professional, empathetic, and knowledgeable Cred Banking Assistant.
- Full awareness of Cred lending policies, application tracking, safety guardrails, and general queries.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import httpx

# Optional Groq key from environment variable
DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()

GREETING_PATTERNS = re.compile(
    r'^\s*(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|howdy|sup|hola)\b',
    re.IGNORECASE
)
HELP_PATTERNS = re.compile(
    r'(what\s*(can\s*you\s*do|can\s*i\s*ask|ques|question|topics|features|options|are\s*your\s*features)|how\s*to\s*use|who\s*are\s*you|help|capabilities|what\s*should\s*i\s*ask|\bfeatures\b|\bhelp\b)',
    re.IGNORECASE
)

POLITE_SYSTEM_PROMPT = (
    "You are Cred's friendly, highly courteous, professional, and knowledgeable AI Domain Support Assistant for Cred Banking & Lending Operations.\n\n"
    "Your Core Personality & Principles:\n"
    "1. Always be exceptionally polite, respectful, and empathetic in your tone. Greet the user warmly and provide clear, structured, and helpful guidance.\n"
    "2. What you can answer and assist with:\n"
    "   • Cred Lending & Banking Policies:\n"
    "     - Loan Eligibility Criteria (Personal, Home, Auto, Education, and Business loans)\n"
    "     - EMI Calculation & Deduction Rules (Reducing balance, NACH auto-debit on 5th, ₹500 bounce fee + 2% penal interest)\n"
    "     - Credit Card Fee Structure & Waivers (Annual spends ₹1.5L waiver, 3.5% monthly revolving charges, 45-day grace)\n"
    "     - KYC Document Requirements (Mandatory PAN, Aadhaar biometric authentication, address proofs, GST/ITR)\n"
    "     - Fraud & Dispute Resolution (72-hour zero customer liability reporting window, 5-day provisional credit)\n"
    "     - Account & Credit Line Closures (Full settlement, NOC turnaround within 7 business days)\n"
    "     - Interest Rate Slabs across credit offerings (Personal: 10.5-18%, Home: 8.4-9.75%, Auto: 8.75-11.5%)\n"
    "     - Prepayment & Foreclosure Penalties (0% on floating retail loans per RBI rules, 2% on fixed retail loans)\n"
    "     - Minimum Average Balance (MAB) Requirements (Metro: ₹10,000, Semi-urban: ₹5,000, salary/student accounts exempt)\n"
    "     - Credit Score Impact Factors & Bureau Reporting (CIBIL 700+, repayment history 35%, credit utilization 30%)\n"
    "     - Joint Account Mandates (Either/Survivor, Former/Survivor mandates with joint liability)\n"
    "     - NRI Account Eligibility (NRE tax-exempt fully repatriable vs NRO domestic taxable accounts)\n"
    "   • Real-Time Application Tracking:\n"
    "     - Evaluating loan applications by record ID (e.g. CRD-APP-1001 to CRD-APP-1050)\n"
    "     - Providing real-time aging, status updates, fraud flags, and underwriting escalation scores\n"
    "   • General Banking & Knowledge Inquiries:\n"
    "     - Answering general financial, mathematical, or general knowledge questions warmly and accurately.\n"
    "3. Output Formatting:\n"
    "   - Use clean Markdown with bullet points and bold highlights for key figures.\n"
    "   - Conclude your answer with a courteous closing offer to assist further."
)


def handle_conversational_query(query: str) -> Optional[str]:
    """
    Handles standard conversational intents (greetings, help, intro, suggested questions) politely.
    """
    clean = query.strip()
    if GREETING_PATTERNS.match(clean) or clean.lower() in ["hi", "hello", "hey"]:
        return (
            "👋 **Hello! Welcome to Cred Domain Support.**\n\n"
            "I am your dedicated Cred Lending Assistant, and I would be delighted to help you today! Here is what I can assist you with:\n\n"
            "• **Lending & Banking Policies:** Inquire about eligibility criteria, EMI calculation rules, credit card fee waivers, interest rate slabs, prepayment rules, KYC verification, and NRI accounts.\n"
            "• **Live Loan Application Tracking:** Look up the real-time status, aging, and underwriting escalation risk of any application by providing its ID (e.g., `CRD-APP-1001`).\n"
            "• **General Financial Guidance:** Ask general banking, credit score, or knowledge questions.\n\n"
            "Please feel free to ask your question or share your application ID, and I will be happy to assist you!"
        )
    
    if HELP_PATTERNS.search(clean):
        return (
            "💡 **Here is everything I can help you with at Cred Support:**\n\n"
            "**1. Cred Lending Policies & Guidelines:**\n"
            "• *\"What is the penalty for bouncing an EMI payment?\"*\n"
            "• *\"What are the KYC documents required for loan verification?\"*\n"
            "• *\"How can I get my credit card annual fee waived?\"*\n"
            "• *\"What is the prepayment penalty on a floating rate home loan?\"*\n"
            "• *\"What are the interest rate slabs for personal vs auto loans?\"*\n"
            "• *\"What are the eligibility criteria for a personal loan?\"*\n"
            "• *\"What is the difference between NRE and NRO accounts?\"*\n\n"
            "**2. Real-Time Application Tracking & Underwriting Status:**\n"
            "• *\"Please check loan status for CRD-APP-1001\"* (or `CRD-APP-1005`, `CRD-APP-1012`)\n"
            "• I will evaluate application aging, fraud flags, and calculate the senior underwriter escalation score.\n\n"
            "**3. General Questions:**\n"
            "• *\"What is the capital of India?\"*\n"
            "• *\"How does compound interest work in banking?\"*\n\n"
            "Please let me know which topic you'd like to explore, and I will gladly provide all the details!"
        )

    return None


def call_groq_api_sync(prompt: str, system: str, api_key: str) -> Optional[str]:
    """
    Invokes Groq's high-speed inference API synchronously.
    """
    try:
        from groq import Groq
        active_key = api_key or DEFAULT_GROQ_KEY
        client = Groq(api_key=active_key)
        
        candidate_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "groq/compound-mini"
        ]

        last_error = None
        for model in candidate_models:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                return completion.choices[0].message.content
            except Exception as e:
                last_error = e
                continue
        
        print(f"Groq API call error across candidates: {last_error}")
        return None
    except Exception as e:
        print(f"Groq client initialization error: {e}")
        return None


def call_openai_api_sync(prompt: str, system: str, api_key: str) -> Optional[str]:
    """
    Invokes OpenAI Chat Completions API synchronously (gpt-4o-mini).
    """
    try:
        with httpx.Client(timeout=20.0) as client:
            res = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }
            )
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"OpenAI error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"OpenAI API call error: {e}")
    return None


def call_gemini_api_sync(prompt: str, system: str, api_key: str) -> Optional[str]:
    """
    Invokes Google Gemini API synchronously (gemini-1.5-flash).
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        with httpx.Client(timeout=20.0) as client:
            res = client.post(
                url,
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}
                }
            )
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Gemini error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"Gemini API call error: {e}")
    return None


def synthesize_smart_mock_answer(query: str, retrieved_chunks: List[Dict[str, Any]], sources: List[str]) -> str:
    """
    Synthesizes a clean, polite, structured policy answer under MOCK_LLM mode.
    """
    if not retrieved_chunks:
        return (
            "I apologize, but I don't have that specific information in Cred's official lending and banking policy guidelines. "
            "Please feel free to ask about our loan eligibility criteria, EMI calculations, credit card rules, or application status!"
        )

    primary_chunk = retrieved_chunks[0]
    topic = primary_chunk.get("topic", "")
    title = primary_chunk.get("title", "")
    sources_str = ", ".join(sources) if sources else primary_chunk.get("doc_id", "Cred Policy")

    # Combine text from relevant chunks
    combined_text = " ".join([c["text"] for c in retrieved_chunks if c.get("similarity", 0) >= 0.31])
    if not combined_text:
        combined_text = primary_chunk["text"]

    closing_courtesy = "\n\n*Please let me know if you need any further clarification or assistance!*"

    # Topic-specific fluent structuring
    if "emi_calculation_rules" in topic or "bounce" in query.lower() or "emi" in query.lower():
        return (
            f"**EMI Calculation & Deduction Rules** ({sources_str}):\n\n"
            f"• **Calculation Method:** EMIs are calculated using the standard reducing balance method based on principal, tenure, and interest rate.\n"
            f"• **Debit Schedule:** Auto-debited via NACH/e-mandate on the **5th of every month**.\n"
            f"• **Bounce / Dishonour Penalty:** A fee of **₹500 + GST** applies per bounce, along with **2% per month penal interest** on overdue amounts.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "kyc" in query.lower() or "document" in query.lower() or "verification" in query.lower():
        return (
            f"**Cred KYC Document Requirements** ({sources_str}):\n\n"
            f"• **Identity Proof:** Valid **Permanent Account Number (PAN)** + Aadhaar digital biometric authentication.\n"
            f"• **Address Proof (if different from Aadhaar):** Passport, Voter ID, or Utility bill issued within the last 3 months.\n"
            f"• **Self-Employed / Business Applicants:** GST registration certificate and past 2 years' Income Tax Returns (ITR).\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "eligibility" in query.lower() or "cibil" in query.lower() or "salary" in query.lower() or "personal loan" in query.lower():
        return (
            f"**Loan Eligibility Guidelines** ({sources_str}):\n\n"
            f"• **Personal Loan:** Salaried applicants must earn at least **₹25,000/month** with a minimum **CIBIL score of 700**.\n"
            f"• **Home & Auto Loans:** Minimum **2 years** stable employment history and a maximum allowable **Debt-to-Income (DTI) ratio of 50%**.\n"
            f"• **Business Loans:** Minimum **3 years** of audited financials with annual turnover exceeding **₹20 Lakhs**.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "fee" in query.lower() or "credit card" in query.lower() or "waiver" in query.lower():
        return (
            f"**Credit Card Fee Structure & Billing** ({sources_str}):\n\n"
            f"• **Joining Fees:** ₹0 for verified Cred members.\n"
            f"• **Annual Fee:** ₹1,000 to ₹5,000 (waived upon annual spending exceeding **₹1,50,000** in the previous year).\n"
            f"• **Revolving Finance Charges:** **3.5% per month (42% per annum)** with up to **45 days interest-free grace period**.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "prepayment" in query.lower() or "foreclosure" in query.lower() or "penalty" in query.lower():
        return (
            f"**Prepayment & Foreclosure Rules** ({sources_str}):\n\n"
            f"• **Floating-Rate Retail Loans:** **0% prepayment or foreclosure penalty** for individual borrowers per Reserve Bank of India (RBI) mandates.\n"
            f"• **Fixed-Rate Loans:** Foreclosures within the first 12 months incur a **2% foreclosure levy** on outstanding principal.\n"
            f"• **Part-Prepayments:** Permitted after 6 completed EMIs for amounts equivalent to at least 2 EMIs.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "fraud" in query.lower() or "dispute" in query.lower() or "unauthorized" in query.lower():
        return (
            f"**Fraud & Dispute Resolution Process** ({sources_str}):\n\n"
            f"• **Reporting Window:** Report within **72 hours** via the Cred app to guarantee **zero customer liability**.\n"
            f"• **Account Protection:** Immediate locking of compromised payment instruments.\n"
            f"• **Provisional Credit:** Issued within **5 business days** during investigation.\n"
            f"• **Final Resolution:** Binding case resolution completed within **30 calendar days**.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "rate" in query.lower() or "interest" in query.lower() or "slab" in query.lower():
        return (
            f"**Cred Interest Rate Slabs** ({sources_str}):\n\n"
            f"• **Secured Home Loans:** **8.40% – 9.75%** APR.\n"
            f"• **Auto Loans:** **8.75% – 11.50%** APR.\n"
            f"• **Personal Loans:** **10.50% – 18.00%** APR (floating, credit-tiered).\n"
            f"• **Business Loans:** **12.00% – 16.50%** APR based on balance sheet strength.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "balance" in query.lower() or "minimum" in query.lower() or "mab" in query.lower():
        return (
            f"**Minimum Average Balance (MAB) Requirements** ({sources_str}):\n\n"
            f"• **Metro Branches:** ₹10,000 Average Monthly Balance.\n"
            f"• **Semi-Urban Branches:** ₹5,000 Average Monthly Balance.\n"
            f"• **Non-Maintenance Fee:** ₹150 – ₹400 + GST per billing cycle.\n"
            f"• **Exemptions:** Corporate salary accounts and student accounts have **₹0 minimum balance** requirements.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "closure" in query.lower() or "close" in query.lower() or "noc" in query.lower():
        return (
            f"**Account & Credit Line Closure Process** ({sources_str}):\n\n"
            f"• **Prerequisites:** Complete settlement of all outstanding principal, interest, and unbilled charges.\n"
            f"• **Request Channel:** In-app portal or verified email with 2-factor OTP authorization.\n"
            f"• **NOC Delivery:** Digitally signed No Objection Certificate (NOC) and No Dues Certificate (NDC) issued within **7 business days**.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "nri" in query.lower() or "nre" in query.lower() or "nro" in query.lower():
        return (
            f"**Non-Resident Indian (NRI) Account Guidelines** ({sources_str}):\n\n"
            f"• **NRE (Non-Resident External):** Full repatriability of principal and interest with **100% Indian income tax exemption**.\n"
            f"• **NRO (Non-Resident Ordinary):** Designed for domestic rupee income (rent, pensions); subject to Indian withholding tax and FEMA ceilings.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )
    elif "score" in query.lower() or "cibil" in query.lower() or "bureau" in query.lower():
        return (
            f"**Credit Score Impact Factors & Bureau Reporting** ({sources_str}):\n\n"
            f"• **Repayment Timeliness:** 35% overall weight.\n"
            f"• **Credit Utilization:** 30% weight (keep utilization below **30%** of aggregate card limits).\n"
            f"• **Credit History Length & Mix:** 15% history length, 10% credit mix, 10% recent inquiries.\n"
            f"• **Reporting:** Delinquencies >30 days are reported monthly to CIBIL, Experian, Equifax, and CRIF High Mark.\n\n"
            f"*(Reference: {title})*"
            f"{closing_courtesy}"
        )

    return f"**According to Cred Policy ({sources_str}):**\n\n{combined_text}{closing_courtesy}"


def generate_response_sync(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    sources: List[str],
    similarity_threshold: float = 0.31,
    provider: str = "groq",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates response synchronously using Real Groq LLM (default) or specified provider with polite persona.
    """
    # 1. Handle conversational greetings / help
    conv_ans = handle_conversational_query(query)
    if conv_ans:
        return {
            "answer": conv_ans,
            "is_grounded": True,
            "fallback_triggered": False,
            "top_similarity": 1.0,
            "sources": ["Cred-Assistant"],
            "model_used": "Conversational-Router"
        }

    # 2. Determine active key (prioritize Groq key)
    env_groq = os.environ.get("GROQ_API_KEY", "").strip()
    env_openai = os.environ.get("OPENAI_API_KEY", "").strip()
    env_gemini = os.environ.get("GEMINI_API_KEY", "").strip()

    active_key = (api_key or "").strip() or env_groq or DEFAULT_GROQ_KEY or env_openai or env_gemini

    # If provider is not explicitly set to 'force_mock', use Real LLM
    if provider != "force_mock" and active_key:
        context_text = "\n".join([f"[{c.get('doc_id')}] {c.get('text')}" for c in retrieved_chunks]) if retrieved_chunks else "No specific policy documents found."
        user_prompt = f"Knowledge Base Context:\n{context_text}\n\nUser Question: {query}\n\nPlease provide a polite, helpful, well-structured answer:"

        real_answer = None
        model_tag = "Groq (Live LLM)"

        if provider in ["groq", "auto", "mock"] and (active_key.startswith("gsk_") or env_groq or DEFAULT_GROQ_KEY):
            real_answer = call_groq_api_sync(user_prompt, POLITE_SYSTEM_PROMPT, active_key)
            model_tag = "Groq (Live LLM)"
        elif provider == "openai" and (active_key.startswith("sk-") or env_openai):
            real_answer = call_openai_api_sync(user_prompt, POLITE_SYSTEM_PROMPT, active_key)
            model_tag = "OpenAI (GPT-4o-mini)"
        elif provider == "gemini" and (active_key.startswith("AIza") or env_gemini):
            real_answer = call_gemini_api_sync(user_prompt, POLITE_SYSTEM_PROMPT, active_key)
            model_tag = "Google Gemini (1.5-Flash)"

        if real_answer:
            top_sim = retrieved_chunks[0].get("similarity", 0.0) if retrieved_chunks else 0.0
            return {
                "answer": real_answer,
                "is_grounded": True,
                "fallback_triggered": False,
                "top_similarity": top_sim,
                "sources": sources,
                "model_used": model_tag
            }

    # 3. Fallback threshold check for purely local deterministic MOCK_LLM mode
    top_sim = retrieved_chunks[0].get("similarity", 0.0) if retrieved_chunks else 0.0
    if not retrieved_chunks or top_sim < similarity_threshold:
        return {
            "answer": "I apologize, but the requested information is not covered in Cred's official lending and banking policy guidelines. Please feel free to ask about our loan eligibility criteria, EMI calculation rules, or application status!",
            "is_grounded": False,
            "fallback_triggered": True,
            "top_similarity": top_sim,
            "sources": [],
            "model_used": "Guardrail-Threshold-Fallback"
        }

    # 4. Default to Smart Deterministic MOCK_LLM
    smart_mock = synthesize_smart_mock_answer(query, retrieved_chunks, sources)
    return {
        "answer": smart_mock,
        "is_grounded": True,
        "fallback_triggered": False,
        "top_similarity": top_sim,
        "sources": sources,
        "model_used": "MOCK_LLM (Deterministic Rule Engine)"
    }
