"""
LangGraph Agent with Production Error Handling and Cloud Persistence Support.
Retry logic, model fallback, structured state management, and async checkpointer integration.
"""

from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from typing_extensions import Annotated, TypedDict

from app.config import get_settings

# === Agent State ===


class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


# === Agent Builder ===


class ProductionAgent:
    """
    Production LangGraph agent with:
    - Neon Postgres Stateful Memory Support
    - Retry on failure (model fallback)
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self, checkpointer=None):
        settings = get_settings()

        self.primary_llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,  # Handled via LangGraph conditional edges
            api_key=settings.openai_api_key,
        )
        self.fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key,
        )
        self.max_retries = settings.max_retries

        # Pass the checkpointer down during graph building
        self.graph = self._build_graph(checkpointer=checkpointer)

    def _build_graph(self, checkpointer=None):
        """Build and compile the LangGraph state machine."""

        system_prompt = """You are an intelligent, factual AI assistant specializing in learning, content creation, and brainstorming.

Core Directives:

Proactive Engagement: When assisting with learning or brainstorming, be highly interactive. Ask thoughtful, guiding follow-up questions to help the user refine their ideas and deepen their understanding.

Strict Fact-Checking: You are bound to objective truth. If a user asks you to remember, confirm, or store a historical, scientific, or objective fact that you know is FALSE, you MUST refuse. You must immediately state that the premise is incorrect, provide the actual fact, and offer a brief explanation."""

        # --- ASYNC NODE 1: Primary Processing ---
        async def process_message(state: AgentState) -> dict:
            """Try to process the message with the primary model asynchronously."""
            messages_for_llm = [SystemMessage(content=system_prompt)] + state[
                "messages"
            ]
            try:
                # Switched to ainvoke for asynchronous performance
                response = await self.primary_llm.ainvoke(messages_for_llm)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state.get("retry_count", 0) + 1,
                    "model_used": "",
                }

        # --- ASYNC NODE 2: Fallback Processing ---
        async def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model asynchronously."""
            messages_for_llm = [SystemMessage(content=system_prompt)] + state[
                "messages"
            ]
            try:
                # Switched to ainvoke
                response = await self.fallback_llm.ainvoke(messages_for_llm)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        # --- ASYNC NODE 3: Error Handler ---
        async def handle_error(state: AgentState) -> dict:
            """Return a graceful error message."""
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I'm sorry, I'm having trouble processing your request "
                            "right now. Please try again in a moment."
                        )
                    )
                ],
                "model_used": "error_handler",
            }

        # --- Routing Logic (Remains synchronous) ---
        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt."""
            if state.get("error") is None:
                return "done"
            elif state.get("retry_count", 0) < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback attempt."""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # Build the graph structure
        workflow = StateGraph(AgentState)

        workflow.add_node("process", process_message)
        workflow.add_node("fallback", try_fallback)
        workflow.add_node("error", handle_error)

        workflow.add_edge(START, "process")
        workflow.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        workflow.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        workflow.add_edge("error", END)

        # Compile the workflow while attaching the external cloud checkpointer
        return workflow.compile(checkpointer=checkpointer)

    # Shifting the entrypoint to ainvoke to accommodate async checkpointer requirements
    @traceable(name="production_agent_invoke")
    async def ainvoke(self, message: str, config: Optional[dict] = None) -> dict:
        """
        Asynchronously invoke the agent with a user message and state configuration.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """
        # We pass the input message and setup baseline state values.
        # LangGraph's checkpointer handles loading existing thread history from Neon automatically.
        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "error": None,
                "retry_count": 0,
                "model_used": "",
            },
            config=config,
        )

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }
