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
    # state['messages'][0].content = 'Tell me about yourself'
    return {"messages": [llm.invoke(state["messages"])]}

def node2(state: State):
    print("\nnode2 ", state)
    last_message_id = state["messages"][-1].id
    print("last message id ", last_message_id)
    new_message = AIMessage(content="Tell me about X men", id=last_message_id)

    # Update the graph's state with the new message
    graph.update_state(config, {"messages": [new_message]})
    
    print("\n node2update ", state)
        
    # Retrieve and print the updated state
    updated_state = graph.get_state(config)  # Hypothetical method
    print("\nUpdated state after update_state:", updated_state)
    
    # Ensure state is updated before invoking the LLM
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

graph = graph_builder.compile(checkpointer=checkpointer)

def save_graph():
    try:
        graph_png_data = graph.get_graph().draw_mermaid_png()
        with open("./langgraph_output.png", "wb") as f:
            f.write(graph_png_data)
    except Exception:
        pass

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nGoodbye!")
        break
    
    events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()



# user_input = "Hi"
# events = graph.stream({"messages": [("user", user_input)]}, config)
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()
        
        
# snapshot = graph.get_state(config)
# existing_message = snapshot.values["messages"][0]
# existing_message.pretty_print()


# answer = (
#     "LangGraph is a library for building stateful, multi-actor applications with LLMs."
# )
# last_message_id = existing_message.id
    
# new_messages = [
#     HumanMessage(content=answer, id=last_message_id),
# ]

# new_messages[-1].pretty_print()
# graph.update_state(
#     config,
#     {"messages": new_messages},
# )

# print("\n\nLast 2 messages;")
# print(graph.get_state(config).values["messages"])





# node3  = {'messages': [HumanMessage(content='Hi', additional_kwargs={}, response_metadata={}, 
#                         id='63636ae7-45fc-4200-9ed9-50b1d5941743'), 
#                     AIMessage(content='Hello! How can I assist you today?',
#                         additional_kwargs={'refusal': None}, 
#                         response_metadata={'token_usage': {'completion_tokens': 9, 
#                                                             'prompt_tokens': 8, 
#                                                             'total_tokens': 17, 
#                                                             'completion_tokens_details': {
#                                                                 'audio_tokens': 0, 
#                                                                 'reasoning_tokens': 0, 
#                                                                 'accepted_prediction_tokens': 0, 
#                                                                 'rejected_prediction_tokens': 0}, 
#                                                             'prompt_tokens_details': {'audio_tokens': 0, 
#                                                                                       'cached_tokens': 0}},
#                                            'model_name': 'gpt-4o-2024-08-06', 
#                                            'system_fingerprint': 'fp_45cf54deae', 
#                                            'finish_reason': 'stop', 
#                                            'logprobs': None}, 
#                         id='run-f43a1147-319c-4d08-b7ed-ee165f9634fb-0', 
#                         usage_metadata={'input_tokens': 8, 
#                                         'output_tokens': 9, 
#                                         'total_tokens': 17,
#                                         'input_token_details': 
#                                             {'audio': 0, 
#                                              'cache_read': 0}, 
#                                             'output_token_details': 
#                                                 {'audio': 0, 'reasoning': 0}}), ]}
