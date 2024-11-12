from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from dotenv import load_dotenv
from typing import Annotated
import json


load_dotenv()
checkpointer = MemorySaver() 
config = {"configurable": {"thread_id": "1"}}

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def node1(state: State):
    print("\nnode1 ", state)
    return {"messages": [llm.invoke(state["messages"])]}

def node2(state: State):
    print("\nnode2 ", state)
    return {"messages": [llm.invoke(state["messages"][-1].content)]}


def node3(state: State):
    print("\nnode3  ", state)
    return {"messages": [llm.invoke(state["messages"][-1].content)]}

graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)

graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", "node3")
graph_builder.add_edge("node3", END)

graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["node2"])

def save_graph():
    try:
        graph_png_data = graph.get_graph().draw_mermaid_png()
        with open("./langgraph_output.png", "wb") as f:
            f.write(graph_png_data)
    except Exception:
        pass


def run():
    while(True):
        config = {"configurable": {"thread_id": "1"}}
        user_input = input("User: ")
        first_arg_message = {"messages": [("user", user_input)]}
        
        snapshot = graph.get_state(config)
        print("snapshot.next ", snapshot.next)
            
        if(len(snapshot.next) != 0 and snapshot.next[0] == 'node2'):
            user_input = input("Ask a human in a loop ")
            existing_message = snapshot.values["messages"][0]
            
            print("Message ID", existing_message.id)
            
            new_message = AIMessage(
                content=user_input,
                id=existing_message.id)
            first_arg_message = None
            graph.update_state(config, {"messages": [new_message]})
            
        events = graph.stream(first_arg_message, config, stream_mode="values")

        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()


run()