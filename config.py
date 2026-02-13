"""
config.py — Application Configuration
======================================

Centralises all configuration values for the Customer Support Ticket
Routing Agent.  Sensitive credentials are loaded from environment
variables so they are never hard-coded in source control.

Environment Variables:
    GROQ_API_KEY : str
        Your Groq API key.  Obtain one from https://console.groq.com/.
        The application will not work without a valid key.

Constants:
    MODEL_NAME : str
        The Groq-hosted LLM model identifier used for all inference
        calls.  Default: ``llama-3.1-8b-instant``.
"""

import os

# Groq API key — MUST be set as an environment variable before running.
# Example (PowerShell):  $env:GROQ_API_KEY = "gsk_..."
# Example (Bash/Zsh) :  export GROQ_API_KEY="gsk_..."
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# The LLM model to use for all Groq API calls.
# See https://console.groq.com/docs/models for available models.
MODEL_NAME = "llama-3.1-8b-instant"
