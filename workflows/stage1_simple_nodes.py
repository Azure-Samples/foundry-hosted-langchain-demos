"""
Simple workflow example: UpperCase → ReverseText.

Demonstrates LangGraph's StateGraph for chaining pure functions (nodes):
    - upper_case converts input text to uppercase
    - reverse_text reverses the string

No LLM calls — pure data transformation to illustrate workflow mechanics.

Run:
    python workflows/stage1_simple_nodes.py
"""

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    text: str


def upper_case(state: WorkflowState) -> WorkflowState:
    """Convert input text to uppercase."""
    return {"text": state["text"].upper()}


def reverse_text(state: WorkflowState) -> WorkflowState:
    """Reverse the string."""
    return {"text": state["text"][::-1]}


async def main():
    """Build and run the UpperCase → ReverseText workflow."""
    graph = StateGraph(WorkflowState)
    graph.add_node("upper_case", upper_case)
    graph.add_node("reverse_text", reverse_text)
    graph.add_edge(START, "upper_case")
    graph.add_edge("upper_case", "reverse_text")
    graph.add_edge("reverse_text", END)

    workflow = graph.compile()
    result = await workflow.ainvoke({"text": "hello world"})

    print("Input:  hello world")
    print(f"Output: {result['text']}")


if __name__ == "__main__":
    asyncio.run(main())
