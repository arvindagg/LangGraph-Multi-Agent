import operator
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the multi-agent system.
    This state is passed between nodes (agents) in the LangGraph.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    """
    A list of all messages in the conversation history.
    'Annotated' with 'operator.add' indicates that when multiple nodes
    modify this field, their changes should be appended (like extending a list).
    """
    next: str
    """
    A string indicating the name of the next agent node to execute.
    This field is typically set by the supervisor agent to route requests.
    """