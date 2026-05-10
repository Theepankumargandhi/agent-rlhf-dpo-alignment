import hashlib
import time

PRIORITY_KEYWORDS = {
    "urgent": "High",
    "asap": "High",
    "immediately": "High",
    "broken": "High",
    "not working": "High",
    "damaged": "High",
    "missing": "Medium",
    "wrong": "Medium",
    "incorrect": "Medium",
    "delayed": "Medium",
    "complaint": "Medium",
}

DEFAULT_PRIORITY = "Low"
DEFAULT_DEPARTMENT = "General Support"

DEPARTMENT_KEYWORDS = {
    "billing": "Billing",
    "payment": "Billing",
    "charge": "Billing",
    "refund": "Billing",
    "technical": "Technical Support",
    "app": "Technical Support",
    "website": "Technical Support",
    "login": "Technical Support",
    "account": "Technical Support",
    "shipping": "Logistics",
    "delivery": "Logistics",
    "tracking": "Logistics",
}


def raise_ticket(query: str) -> str:
    """Create a mock support ticket for the given customer query.

    Assigns a deterministic ticket ID derived from the query content and
    current timestamp, infers priority and routing department from keywords,
    and returns a confirmation message.

    Args:
        query: The customer's issue description in natural language.

    Returns:
        A string confirming ticket creation with ticket ID, inferred priority,
        assigned department, and expected response time.
    """
    raw = f"{query}{time.time()}".encode()
    ticket_id = "TKT-" + hashlib.md5(raw).hexdigest()[:8].upper()

    query_lower = query.lower()

    priority = DEFAULT_PRIORITY
    for keyword, level in PRIORITY_KEYWORDS.items():
        if keyword in query_lower:
            priority = level
            break

    department = DEFAULT_DEPARTMENT
    for keyword, dept in DEPARTMENT_KEYWORDS.items():
        if keyword in query_lower:
            department = dept
            break

    response_time = {"High": "2 hours", "Medium": "24 hours", "Low": "48 hours"}[priority]

    return (
        f"Support ticket created successfully!\n"
        f"  Ticket ID  : {ticket_id}\n"
        f"  Priority   : {priority}\n"
        f"  Department : {department}\n"
        f"  Issue      : {query[:120]}{'...' if len(query) > 120 else ''}\n"
        f"  Status     : Open\n"
        f"  Expected response within {response_time}.\n"
        f"You will receive a confirmation email shortly."
    )
