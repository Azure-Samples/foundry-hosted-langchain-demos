"""
Simple workflow example: UpperCase → ReverseText.

Demonstrates LangGraph's Functional API for chaining pure functions (tasks):
    - upper_case converts input text to uppercase
    - reverse_text reverses the string

No LLM calls — pure data transformation to illustrate workflow mechanics.

Run:
    python workflows/stage1_simple_nodes.py
"""

from langgraph.func import entrypoint, task


@task
def upper_case(text: str) -> str:
    """Convert input text to uppercase."""
    return text.upper()


@task
def reverse_text(text: str) -> str:
    """Reverse the string."""
    return text[::-1]


@entrypoint()
def workflow(text: str) -> str:
    """Chain: UpperCase → ReverseText."""
    return reverse_text(upper_case(text).result()).result()


if __name__ == "__main__":
    result = workflow.invoke("hello world")
    print("Input:  hello world")
    print(f"Output: {result}")
