import random

MOCK_ORDER_STATUSES = [
    "shipped",
    "processing",
    "out_for_delivery",
    "delivered",
    "pending_payment",
]

CARRIER_NAMES = ["UPS", "FedEx", "USPS", "DHL"]


def order_status(order_id: str) -> str:
    """Return a mock status report for the given order ID.

    Deterministically seeds randomness from the order ID so the same
    order always returns the same mock status within a session.

    Args:
        order_id: The alphanumeric order identifier extracted from the query
                  (e.g. "1234", "ORD-5678").

    Returns:
        A human-readable string describing the current mock status of the order,
        including estimated delivery or last-known location where applicable.
    """
    seed = sum(ord(c) for c in order_id)
    random.seed(seed)

    status = random.choice(MOCK_ORDER_STATUSES)
    carrier = random.choice(CARRIER_NAMES)
    tracking = f"TRK{seed % 90000 + 10000}"

    if status == "shipped":
        return (
            f"Order #{order_id} has been shipped via {carrier}. "
            f"Tracking number: {tracking}. Estimated delivery: 2–3 business days."
        )
    elif status == "processing":
        return (
            f"Order #{order_id} is currently being processed at our warehouse. "
            "It will be dispatched within 24 hours."
        )
    elif status == "out_for_delivery":
        return (
            f"Order #{order_id} is out for delivery today via {carrier}. "
            "Expect it by end of business day."
        )
    elif status == "delivered":
        return (
            f"Order #{order_id} was delivered successfully. "
            "If you haven't received it, please contact support."
        )
    else:  # pending_payment
        return (
            f"Order #{order_id} is on hold pending payment confirmation. "
            "Please check your payment method and try again."
        )
