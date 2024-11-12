from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_openai import OpenAIEmbeddings
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from dotenv import load_dotenv
from typing import Annotated
import json

load_dotenv()

tool = TavilySearchResults(max_results=2)
tools = [tool]

memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
embeddings = OpenAIEmbeddings()
config = {"configurable": {"thread_id": "1"}}

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool]) 
graph_builder.add_node("tools", tool_node)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"])


user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()