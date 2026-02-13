"""
main.py — Demo Runner for the Customer Support Ticket Routing Agent
====================================================================

This script demonstrates the agent by processing a set of sample
support tickets that cover different scenarios:

    - **TICKET-001**: Billing issue (duplicate charge / refund request).
    - **TICKET-002**: Critical outage (production server down).
    - **TICKET-003**: Sales inquiry (bulk educational licensing).

For each ticket the full 4-step pipeline is executed and the results
are pretty-printed to the console:

    1. Analysis   — intent, sentiment, urgency, summary.
    2. Classification — department, priority, reason.
    3. Entity Extraction — customer name, order ID, contact info, etc.
    4. Generated Response — customer reply, internal note, escalation flag.

Usage::

    # Make sure GROQ_API_KEY is set in your environment
    python main.py
"""

import json
from agent import process_ticket, _safe_json

{
        "id": "TICKET-001",
        "text": (
            "Hi, I'm John Smith (order #ORD-98432). I was charged twice for my "
            "subscription on Jan 5 2026. The amount of $49.99 appeared two times "
            "on my credit card ending 4821. Please refund the duplicate charge ASAP. "
            "My email is john.smith@email.com."
        ),
    },
    {
        "id": "TICKET-002",
        "text": (
            "Our entire production server has been down since 3 AM. None of our "
            "500+ employees can access the platform. We are losing thousands of "
            "dollars every hour. This is absolutely unacceptable! We need immediate "
            "help. Account ID: ENT-20050. Contact: ops-team@bigcorp.com"
        ),
    },
    {
        "id": "TICKET-003",
        "text": (
            "Hello, I'd like to know if you offer any discounts for educational "
            "institutions. We're a university looking to purchase around 200 licenses. "
            "Could you send us a quote? Thanks! – Prof. Maria Garcia"
        ),
    },
]


def print_section(title: str, content: str):
    """Print a labelled, boxed section with pretty-printed JSON content.

    Args:
        title:   Section heading (e.g. '1. ANALYSIS').
        content: Raw string (typically JSON) to display.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(_safe_json(content))


def main():
    """Iterate over sample tickets, process each through the agent, and display results."""
    for ticket in SAMPLE_TICKETS:
        print(f"\n{'#'*60}")
        print(f"  PROCESSING: {ticket['id']}")
        print(f"{'#'*60}")
        print(f"\nTicket Text:\n{ticket['text']}\n")

        results = process_ticket(ticket["text"])

        if "analyze_ticket" in results:
            print_section("1. ANALYSIS", results["analyze_ticket"])
        if "classify_ticket" in results:
            print_section("2. CLASSIFICATION", results["classify_ticket"])
        if "extract_entities" in results:
            print_section("3. EXTRACTED ENTITIES", results["extract_entities"])
        if "generate_response" in results:
            print_section("4. GENERATED RESPONSE", results["generate_response"])
        if "final_summary" in results:
            print(f"\n{'='*60}")
            print("  AGENT SUMMARY")
            print(f"{'='*60}")
            print(results["final_summary"])

        print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    main()
