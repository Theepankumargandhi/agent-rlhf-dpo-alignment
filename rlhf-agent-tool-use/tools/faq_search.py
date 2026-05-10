FAQ_DATABASE = {
    "business hours": "Our support team is available Monday to Friday, 9 AM – 6 PM EST.",
    "contact": "You can reach us at support@example.com or call 1-800-555-0199.",
    "shipping time": "Standard shipping takes 5–7 business days. Express shipping takes 1–2 business days.",
    "payment methods": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
    "account creation": "Visit our website and click 'Sign Up'. Fill in your details and verify your email.",
    "password reset": "Click 'Forgot Password' on the login page and follow the instructions sent to your email.",
    "international shipping": "We ship to over 50 countries. International delivery takes 10–15 business days.",
    "product warranty": "All products come with a 1-year manufacturer warranty.",
    "loyalty program": "Our rewards program gives 1 point per dollar spent. Redeem 100 points for $5 off.",
    "gift cards": "Gift cards are available in denominations of $10, $25, $50, and $100.",
    "default": (
        "Thank you for reaching out! For detailed assistance, please visit our Help Center at "
        "help.example.com or contact our support team at support@example.com."
    ),
}


def faq_search(query: str) -> str:
    """Search the FAQ database for an answer matching the user query.

    Performs a simple keyword match against known FAQ topics and returns
    the most relevant hardcoded answer.

    Args:
        query: The customer's natural-language question.

    Returns:
        A string containing the FAQ answer, or a default fallback message
        if no matching topic is found.
    """
    query_lower = query.lower()
    for keyword, answer in FAQ_DATABASE.items():
        if keyword in query_lower:
            return answer
    return FAQ_DATABASE["default"]
