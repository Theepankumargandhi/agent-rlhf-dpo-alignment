REFUND_POLICY_TEXT = """
Refund Policy — Example Store

ELIGIBILITY
Items are eligible for a full refund within 30 days of delivery, provided they are:
  - Unused and in original packaging
  - Accompanied by proof of purchase (order number or receipt)

NON-REFUNDABLE ITEMS
  - Digital downloads and software licenses
  - Gift cards and store credit
  - Perishable goods (food, flowers)
  - Items marked "Final Sale"

PROCESS
  1. Initiate a return request at returns.example.com or through your account dashboard.
  2. Print the prepaid shipping label we email you.
  3. Drop off the package at any authorised carrier location.
  4. Once we receive and inspect the item, refunds are issued within 5–7 business days
     to the original payment method.

EXCHANGES
We offer free exchanges for defective or incorrect items. Contact support within
48 hours of delivery with a photo of the issue.

PARTIAL REFUNDS
Partial refunds may be granted for items returned in used or damaged condition at
our discretion.

For questions, email refunds@example.com or call 1-800-555-0199 (Mon–Fri, 9 AM–6 PM EST).
""".strip()


def refund_policy() -> str:
    """Return the full hardcoded refund policy text.

    No arguments are required; the policy is static store-wide content.

    Returns:
        A multi-line string containing the complete refund policy.
    """
    return REFUND_POLICY_TEXT
