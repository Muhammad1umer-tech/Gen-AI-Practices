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


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        print("__init__ tools_by_name: ",self.tools_by_name)

    def __call__(self, inputs: dict):
        print("__call__1", inputs) 
        #{'messages': [HumanMessage(content='tell me about crewAI', 
        # additional_kwargs={}, response_metadata={}, id='ff49e5c1-ba56-46f0-a766-4997180f995c'), 
        # AIMessage(content='', additional_kwargs={'tool_calls': 
        # [{'id': 'call_U0MFXC13VYD2nsr9YDasakJA', 'function': {'arguments': '{"query":"crewAI"}', 
        # 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, 
        # response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 84, 
        # 'total_tokens': 103, 'completion_tokens_details': {'audio_tokens': None, 
        # 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None,
        #  'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 
        # 'fp_90354628f2', 'finish_reason': 'tool_calls', 'logprobs': None},
        #  id='run-ccebee6a-7712-45dd-a56f-ccd14d03391c-0', tool_calls=[{'name': 
        # 'tavily_search_results_json', 'args': {'query': 'crewAI'}, 'id': 
        # 'call_U0MFXC13VYD2nsr9YDasakJA', 'type': 'tool_call'}], 
        # usage_metadata={'input_tokens': 84, 'output_tokens': 19, 
        # 'total_tokens': 103, 'input_token_details': {'cache_read': 0}, 'output_token_details': 
        # {'reasoning': 0}})]}
        if messages := inputs.get("messages", []):
            print("messages: ", messages)
            message = messages[-1]
            print("message: ", message)

        else:
            raise ValueError("No message found in input")
        outputs = []
        print("__call__2", message.tool_calls)
        for tool_call in message.tool_calls:
            print(f"__call__tool_call['name']: {tool_call['name']}")
            print(f"__call__tool_call['args']: {tool_call['args']}")
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            print(f"__call__tool_tool_result: {tool_result}")
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

tool_node = BasicToolNode(tools=[tool]) 
# tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END}
)
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
    )

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
        for value in event:
            value["messages"][-1].pretty_print()

def save_graph():
    graph = graph_builder.compile()
    try:
        graph_png_data = graph.get_graph().draw_mermaid_png()
        with open("./langgraph_output.png", "wb") as f:
            f.write(graph_png_data)
    except Exception:
        pass

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        response = graph.invoke({"messages": [("user", user_input)]}, config, stream_mode="values")
        print(response["message"[-1].content])
    except:
        # fallback if input() is not available
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break