"""
Approximate per-1M-token pricing in USD. These numbers WILL go stale —
check ai.google.dev/pricing before trusting them for billing.
"""
PRICING = {
    "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":   {"input": 1.25,  "output": 5.00},
}


def compute_cost(model: str, usage: dict) -> float:
    """usage = {"input_tokens": int, "output_tokens": int} -> cost in USD."""
    rates = PRICING.get(model)
    if rates is None:
        return 0.0  # unknown model: don't guess, just report zero
    input_cost  = (usage["input_tokens"]  / 1_000_000) * rates["input"]
    output_cost = (usage["output_tokens"] / 1_000_000) * rates["output"]
    return round(input_cost + output_cost, 6)
