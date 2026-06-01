# Production LangGraph API

A robust, production-grade conversational AI API built with Python, LangGraph, and OpenAI. This project implements advanced agentic routing, built-in security filtering, and response caching for high-performance AI deployments.

## Features

* **Agentic Fallback Routing:** Intelligent state machine that automatically reroutes failed requests or timeouts from the primary LLM to a high-availability fallback model.
* **Security Pipeline:** Custom middleware that sanitizes inputs and blocks prompt injection before requests hit the LLM.
* **Response Caching:** In-memory caching with TTL to eliminate redundant LLM calls, reduce latency, and protect API rate limits.
* **Structured Observability:** JSON-formatted logging and automated metrics collection for seamless integration with enterprise monitoring dashboards.

## Prerequisites

To run this project, you will need:
* Python 3.12+
* [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)
* An OpenAI API Key (or Azure OpenAI credentials)

## Installation & Setup

**1. Clone the repository and navigate to the project folder**
```bash
git clone https://github.com/dhruvsinghal028/langgraphFastApi-production-api.git
cd langgraphFastApi-production-api

**2. RUN UV **

uv sync

**3. create .env  **
# Create a .env file in the root directory. And set the below varibles in it 

# Code snippet
OPENAI_API_KEY=your_api_key_here
PRIMARY_MODEL=gpt-4o-mini
FALLBACK_MODEL=gpt-3.5-turbo
RATE_LIMIT=20/minute
CACHE_TTL_SECONDS=300

**4. Running the API  **
Start the local development server:

uv run uvicorn app.main:app --reload
