"""
Simple workflow example: UpperCase → ReverseText.

Demonstrates LangGraph's Graph API for chaining pure functions as nodes:
    - upper_case converts input text to uppercase
    - reverse_text reverses the string

No LLM calls — pure data transformation to illustrate workflow mechanics.

Run:
    uv run python workflows/stage1_simple_nodes.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TextState(TypedDict):
    text: str


def upper_case(state: TextState) -> dict:
    """Convert input text to uppercase."""
    return {"text": state["text"].upper()}


def reverse_text(state: TextState) -> dict:
    """Reverse the string."""
    return {"text": state["text"][::-1]}


graph = (
    StateGraph(TextState)
    .add_node(upper_case)
    .add_node(reverse_text)
    .add_edge(START, "upper_case")
    .add_edge("upper_case", "reverse_text")
    .add_edge("reverse_text", END)
    .compile()
)


if __name__ == "__main__":
    result = graph.invoke({"text": "hello world"})
    print("Input:  hello world")
    print(f"Output: {result['text']}")
