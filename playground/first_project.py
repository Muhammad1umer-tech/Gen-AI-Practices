from anthropic import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_openai import OpenAIEmbeddings
from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Annotated
import json


load_dotenv()

memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
config = {"configurable": {"thread_id": "1"}}

class State(TypedDict):
    messages: Annotated[list, add_messages]

class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str

@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return f"I looked up: {query}. Result: It's sunny in San Francisco, but you better look out if you're a Gemini 😈."

tools = [search]
tool_node = ToolNode(tools)
model = llm.bind_tools(tools + [AskHuman])
llm_with_tools = llm.bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    # For example, send a slack message, etc
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We define a fake node to ask the human
def ask_human(state):
    user_question = input("ask")
    return "haha"


graph_builder = StateGraph(State)


# Define the three nodes we will cycle between
graph_builder.add_node("agent", call_model)
graph_builder.add_node("action", tool_node)
graph_builder.add_node("ask_human", ask_human)



# Set the entrypoint as `agent`
# This means that this node is the first one called
graph_builder.add_edge(START, "agent")


# We now add a conditional edge
graph_builder.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # We may ask the human
        "ask_human": "ask_human",
        # Otherwise we finish.
        "end": END,
    },
)


# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
graph_builder.add_edge("action", "agent")

# After we get back the human response, we go back to the agent
graph_builder.add_edge("ask_human", "agent")



# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
# We add a breakpoint BEFORE the `ask_human` node so it never executes
app = graph_builder.compile(checkpointer=memory, interrupt_before=["ask_human"])



config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(
    content="Use the search tool to ask the user where they are, then look up the weather there"
)
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


tool_call_id = app.get_state(config).values["messages"][-1].tool_calls[0]["id"]

# We now create the tool call with the id and the response we want
tool_message = [
    {"tool_call_id": tool_call_id, "type": "tool", "content": "san francisco"}
]

# We now update the state
# Notice that we are also specifying `as_node="ask_human"`
# This will apply this update as this node,
# which will make it so that afterwards it continues as normal
app.update_state(config, {"messages": tool_message}, as_node="ask_human")


for event in app.stream(None, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


