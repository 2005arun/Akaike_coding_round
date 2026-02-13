"""
agent.py — Customer Support Ticket Routing Agent
=============================================

This module implements an AI-powered customer support ticket processing
pipeline using the Groq LLM API with tool-calling (function-calling)
capabilities.

Architecture Overview:
    The agent follows a sequential 4-step pipeline to process each ticket:

    1. **analyze_ticket**    → Determines intent, sentiment, urgency, and summary.
    2. **classify_ticket**   → Routes to a department and assigns priority (P1–P4).
    3. **extract_entities**  → Pulls out customer name, order ID, dates, etc.
    4. **generate_response** → Crafts a customer-facing reply and internal notes.

    The LLM autonomously decides which tool to call next via the Groq
    function-calling API. Results from earlier steps are fed into later
    steps so the agent builds up context progressively.

Key Components:
    - ``TOOLS``: OpenAI-compatible tool/function schema definitions.
    - ``TOOL_MAP``: Maps tool names to their Python implementations.
    - ``AGENT_SYSTEM``: System prompt that instructs the LLM on the workflow.
    - ``process_ticket()``: Main entry point — runs the full agent loop.

Usage:
    >>> from agent import process_ticket
    >>> results = process_ticket("I was double-charged for order #123.")
    >>> print(results.keys())
    dict_keys(['analyze_ticket', 'classify_ticket', 'extract_entities',
               'generate_response', 'final_summary'])
"""

import json
import re
from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

# ──────────────────────────────────────────────────────────────
# Groq client initialisation
# ──────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────────────────────
# Tool / Function Schemas (OpenAI-compatible format)
# ──────────────────────────────────────────────────────────────
# Each schema tells the LLM *what* a tool does and *which*
# parameters it accepts so it can decide when to call it.
# ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_ticket",
            "description": "Analyze a customer support ticket to understand intent, sentiment, and urgency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {
                        "type": "string",
                        "description": "The raw text of the customer support ticket.",
                    }
                },
                "required": ["ticket_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_ticket",
            "description": "Classify the ticket into a department and assign a priority level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {
                        "type": "string",
                        "description": "The raw text of the customer support ticket.",
                    },
                    "analysis": {
                        "type": "string",
                        "description": "Previous analysis summary of the ticket.",
                    },
                },
                "required": ["ticket_text", "analysis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract key entities such as product names, order IDs, dates, and customer info from the ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {
                        "type": "string",
                        "description": "The raw text of the customer support ticket.",
                    }
                },
                "required": ["ticket_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_response",
            "description": "Generate an appropriate customer-facing reply or escalation note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {
                        "type": "string",
                        "description": "The raw text of the customer support ticket.",
                    },
                    "analysis": {
                        "type": "string",
                        "description": "Analysis summary.",
                    },
                    "classification": {
                        "type": "string",
                        "description": "Classification result (department + priority).",
                    },
                    "entities": {
                        "type": "string",
                        "description": "Extracted entities from the ticket.",
                    },
                },
                "required": ["ticket_text", "analysis", "classification", "entities"],
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────
# LLM Helper
# ──────────────────────────────────────────────────────────────

def _llm(system: str, user: str) -> str:
    """Send a one-shot request to the Groq LLM and return the text response.

    This is the low-level helper used by every tool function.  It creates
    a simple two-message conversation (system + user) and returns the
    assistant's reply as a plain string.

    Args:
        system: The system prompt that defines the LLM's persona and
                output format (e.g. "return ONLY valid JSON").
        user:   The user-facing content — typically the ticket text,
                optionally enriched with prior analysis results.

    Returns:
        The stripped text content of the LLM's response.
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,   # Low temperature for consistent, factual output
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────
# Tool Implementations
# ──────────────────────────────────────────────────────────────

def analyze_ticket(ticket_text: str) -> str:
    """Step 1 — Analyse intent, sentiment, urgency, and provide a summary.

    Sends the raw ticket to the LLM with a prompt that asks for a JSON
    object containing:
        - ``intent``   : What the customer wants (e.g. refund, info).
        - ``sentiment``: positive / neutral / negative / angry.
        - ``urgency``  : low / medium / high / critical.
        - ``summary``  : A one-line summary of the ticket.

    Args:
        ticket_text: The raw customer support ticket string.

    Returns:
        A JSON-formatted string with the analysis results.
    """
    system = (
        "You are a support ticket analyst. Analyze the ticket and return a JSON object with:\n"
        '- "intent": what the customer wants\n'
        '- "sentiment": positive / neutral / negative / angry\n'
        '- "urgency": low / medium / high / critical\n'
        '- "summary": one-line summary\n'
        "Return ONLY valid JSON."
    )
    return _llm(system, ticket_text)


def classify_ticket(ticket_text: str, analysis: str) -> str:
    """Step 2 — Route the ticket to a department and assign priority.

    Uses both the original ticket and the analysis from Step 1 to
    determine the appropriate department and priority level.

    Departments:
        Billing | Technical Support | Sales | Account Management |
        Product Feedback | General Inquiry

    Priority levels:
        P1-Critical | P2-High | P3-Medium | P4-Low

    Args:
        ticket_text: The raw customer support ticket string.
        analysis:    JSON string output from ``analyze_ticket()``.

    Returns:
        A JSON-formatted string with ``department``, ``priority``,
        and ``reason`` fields.
    """
    system = (
        "You are a ticket routing specialist. Based on the ticket and its analysis, return a JSON object with:\n"
        '- "department": one of [Billing, Technical Support, Sales, Account Management, Product Feedback, General Inquiry]\n'
        '- "priority": one of [P1-Critical, P2-High, P3-Medium, P4-Low]\n'
        '- "reason": brief justification\n'
        "Return ONLY valid JSON."
    )
    user = f"Ticket:\n{ticket_text}\n\nAnalysis:\n{analysis}"
    return _llm(system, user)


def extract_entities(ticket_text: str) -> str:
    """Step 3 — Extract structured entities from the ticket text.

    Identifies and extracts key information such as:
        - ``customer_name`` : The customer's name (if mentioned).
        - ``order_id``      : Order or reference numbers.
        - ``product``       : Product or service mentioned.
        - ``dates``         : Any dates referenced in the ticket.
        - ``contact_info``  : Email addresses or phone numbers.
        - ``issue_keywords``: Key terms describing the issue.

    Fields not found in the ticket are set to ``null``.

    Args:
        ticket_text: The raw customer support ticket string.

    Returns:
        A JSON-formatted string with the extracted entities.
    """
    system = (
        "You are an entity extraction specialist. Extract key entities from the ticket and return a JSON object with:\n"
        '- "customer_name": if mentioned\n'
        '- "order_id": any order/reference numbers\n'
        '- "product": product or service mentioned\n'
        '- "dates": any dates mentioned\n'
        '- "contact_info": email/phone if present\n'
        '- "issue_keywords": list of key issue terms\n'
        "Use null for fields not found. Return ONLY valid JSON."
    )
    return _llm(system, ticket_text)


def generate_response(
    ticket_text: str, analysis: str, classification: str, entities: str
) -> str:
    """Step 4 — Generate a customer-facing reply and internal notes.

    Combines all prior context (analysis, classification, entities) to
    craft an empathetic, actionable response.  If the ticket is
    classified as P1-Critical or P2-High, an escalation note is
    automatically included.

    Args:
        ticket_text:    The raw customer support ticket string.
        analysis:       JSON string from ``analyze_ticket()``.
        classification: JSON string from ``classify_ticket()``.
        entities:       JSON string from ``extract_entities()``.

    Returns:
        A JSON-formatted string containing:
            - ``response``          : The customer-facing reply.
            - ``internal_note``     : A note for the support team.
            - ``escalation_needed`` : Boolean flag.
    """
    system = (
        "You are a professional customer support agent. Generate a helpful, empathetic response.\n"
        "If priority is P1-Critical or P2-High, include an escalation note.\n"
        "Return a JSON object with:\n"
        '- "response": the customer-facing reply\n'
        '- "internal_note": note for the support team\n'
        '- "escalation_needed": true/false\n'
        "Return ONLY valid JSON."
    )
    user = (
        f"Ticket:\n{ticket_text}\n\n"
        f"Analysis:\n{analysis}\n\n"
        f"Classification:\n{classification}\n\n"
        f"Entities:\n{entities}"
    )
    return _llm(system, user)


# ──────────────────────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────────────────────
# Maps tool names (as referenced by the LLM) to their Python
# callable implementations so the agent loop can dispatch calls.
# ──────────────────────────────────────────────────────────────
TOOL_MAP = {
    "analyze_ticket": analyze_ticket,
    "classify_ticket": classify_ticket,
    "extract_entities": extract_entities,
    "generate_response": generate_response,
}


# ──────────────────────────────────────────────────────────────
# Agent System Prompt
# ──────────────────────────────────────────────────────────────
AGENT_SYSTEM = """\
You are a Customer Support Ticket Routing Agent.
You process tickets step-by-step using these tools in order:

1. **analyze_ticket** – understand intent, sentiment, urgency
2. **classify_ticket** – assign department & priority (needs the analysis)
3. **extract_entities** – pull out key entities
4. **generate_response** – craft reply (needs analysis, classification, entities)

Call ONE tool at a time. After all four tools have been called, respond with
a final summary that includes the results from every step.
"""


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _safe_json(text: str) -> str:
    """Attempt to pretty-print a JSON string.

    If ``text`` is valid JSON it is re-serialised with 2-space
    indentation for readability.  Otherwise the original string is
    returned unchanged — this prevents crashes when the LLM returns
    slightly malformed output.

    Args:
        text: A string that *may* be valid JSON.

    Returns:
        Pretty-printed JSON or the original string.
    """
    try:
        return json.dumps(json.loads(text), indent=2)
    except Exception:
        return text


# ──────────────────────────────────────────────────────────────
# Main Agent Loop
# ──────────────────────────────────────────────────────────────

def process_ticket(ticket_text: str) -> dict:
    """Process a customer support ticket through the full agent pipeline.

    This is the **main entry point** of the module.  It starts a
    conversation with the Groq LLM, which autonomously decides which
    tool to invoke at each step using the function-calling API.

    Workflow:
        1. The LLM receives the ticket and the ``AGENT_SYSTEM`` prompt.
        2. It issues a ``tool_call`` (e.g. ``analyze_ticket``).
        3. The corresponding Python function is executed locally.
        4. The result is appended to the conversation as a ``tool``
           message, and the loop continues.
        5. After all four tools have been called (or ``max_iterations``
           is reached), the LLM returns a natural-language summary.

    Args:
        ticket_text: The raw text of the support ticket to process.

    Returns:
        A dictionary whose keys are the tool names that were called
        (``analyze_ticket``, ``classify_ticket``, ``extract_entities``,
        ``generate_response``) mapped to their JSON-string results,
        plus a ``final_summary`` key with the LLM's closing summary.

    Example::

        results = process_ticket("I need a refund for order #456.")
        print(json.loads(results["classify_ticket"]))
        # {'department': 'Billing', 'priority': 'P3-Medium', ...}
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": f"Process this support ticket:\n\n{ticket_text}"},
    ]

    results = {}  # Accumulates outputs from each tool call
    max_iterations = 10  # Safety cap to prevent infinite loops

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1500,
        )

        msg = response.choices[0].message

        # If the LLM did NOT request a tool call, it has finished
        # and is returning a natural-language summary.
        if not msg.tool_calls:
            results["final_summary"] = msg.content.strip() if msg.content else ""
            break

        # Append the assistant message (including tool_calls metadata)
        # so the conversation history stays consistent.
        messages.append(msg)

        # Execute each requested tool call and feed the result back.
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            fn = TOOL_MAP.get(fn_name)
            if fn is None:
                tool_result = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                tool_result = fn(**args)

            # Store the result keyed by tool name for easy access.
            results[fn_name] = tool_result

            # Feed the tool output back into the conversation so the
            # LLM can use it when deciding the next step.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    return results
