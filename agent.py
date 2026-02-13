import json
import re
from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)

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

def _llm(system: str, user: str) -> str:
    """Helper to call Groq LLM."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def analyze_ticket(ticket_text: str) -> str:
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


def generate_response(ticket_text: str, analysis: str, classification: str, entities: str) -> str:
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


TOOL_MAP = {
    "analyze_ticket": analyze_ticket,
    "classify_ticket": classify_ticket,
    "extract_entities": extract_entities,
    "generate_response": generate_response,
}


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


def _safe_json(text: str) -> str:
    """Try to pretty-print JSON; return as-is if not valid JSON."""
    try:
        return json.dumps(json.loads(text), indent=2)
    except Exception:
        return text


def process_ticket(ticket_text: str) -> dict:
    """
    Run the agent loop: let the LLM decide which tool to call,
    execute it, feed results back, repeat until done.
    Returns a dict with all step results.
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": f"Process this support ticket:\n\n{ticket_text}"},
    ]

    results = {}
    max_iterations = 10  

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

        if not msg.tool_calls:
            results["final_summary"] = msg.content.strip() if msg.content else ""
            break

        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            fn = TOOL_MAP.get(fn_name)
            if fn is None:
                tool_result = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                tool_result = fn(**args)

            results[fn_name] = tool_result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    return results
