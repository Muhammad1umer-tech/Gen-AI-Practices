from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_openai import OpenAIEmbeddings
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.tools import tool
from anthropic import BaseModel
from dotenv import load_dotenv
from typing import Annotated
import json
from tools import  get_job, get_resume

load_dotenv()

memory = MemorySaver()
tools = [get_job, get_resume]
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
llm_bind_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str

def Export(state):
    system_message = """You are a resume export. You are tasked with improving the user resume based on a 
    job description. You can access the resume and job data using provided tools
    
    You must Never provide information that the user does not have.
    This includes, skills or experiences that are not in the resume, Do not make things up."""

    messages = state["messages"]
    response = llm_bind_tools.invoke([system_message] + messages)
    print("response ", response)    
    return {"messages": [response]}

tool_node = ToolNode(tools)

def ask_human(state):
    user_question = input("ask the question: ")
    return {"messages": user_question}
    # return add_messages(user_question)

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    print("last_message ", last_message.tool_calls)
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "ask_human"

def human_in_a_loop(state):
    messages = state["messages"]
    print("state-->", state)
    last_message = messages[-1]
    if last_message == 'no':
        return "end"
    else:
        return "tools"
    
graph_builder = StateGraph(State)

graph_builder.add_node("Export", Export)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ask_human", ask_human)

graph_builder.add_edge(START, "Export")
graph_builder.add_conditional_edges(
    "Export",
    should_continue,
    {
        "ask_human": "ask_human",
        "end": END,
    },
)
graph_builder.add_conditional_edges(
     "ask_human",
     human_in_a_loop,
    {
        "end": END,
        "tools": "tools",
    },
)
graph_builder.add_edge("tools", "Export")

graph = graph_builder.compile(checkpointer=memory)


def save_graph():
    graph = graph_builder.compile()
    try:
        graph_png_data = graph.get_graph().draw_mermaid_png()
        with open("./langgraph_output.png", "wb") as f:
            f.write(graph_png_data)
    except Exception:
        pass

save_graph()

config = {"configurable": {"thread_id": "5"}}    

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
    print("events ", events)
    for event in events:
        event["messages"][-1].pretty_print()


