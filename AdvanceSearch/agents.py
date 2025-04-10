from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import initialize_agent, Tool
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
from anthropic import BaseModel
from dotenv import load_dotenv
from typing import Annotated
import json

from tools import  InternetSearchTool, InsightResearcher, DocumentWriterTool

# InternetSearchAnalyst (go thru different urls and create summary and give to InsightRearcher)
# InsightResearcher (indept search on each topic and will give use detailed report)
# duckduckgo search and beautiful soupP
#Tools (search the internet, and read) duckduckgo_search, beautiful soup

load_dotenv()


llm = ChatOpenAI(model="gpt-4o")
llm_scrap_tool = llm.bind_tools([InternetSearchTool])
llm_Insight_tool = llm.bind_tools([InsightResearcher])
llm_DocumentWriter = llm.bind_tools([DocumentWriterTool])

   
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    
nodes = ["WebSearcher", "InsightAnalyst", "DocumentWriter"]

def Supervisor(state):
    system_prompt = (
    "You are a supervisor assigning tasks sequentially to {nodes}.\n\n"
    
    "Task Order:\n"
    "  1. 'WebSearcher' - First, this agent scrapes URLs based on the user query.\n"
    "  2. 'InsightAnalyst' - Next, this agent retrieves data from the URLs provided by WebSearcher.\n"
    "  3. 'DocumentWriter' - Finally, this agent summarizes the retrieved data.\n\n"
    
    "Each agent completes its specific task and reports findings back to you. "
    "Once all tasks are complete, respond with 'END'.\n\n"
    
    "Respond in JSON format with:\n"
    "  - 'next': the agent assigned to act next\n"
    "  - 'query': instructions specific to the assigned agent, if applicable\n\n"
    
    "Agent Instructions:\n"
    "  - If 'next' is 'WebSearcher':\n"
    "      - 'query' should contain a refined user query for URL scraping.\n"
    "  - If 'next' is 'InsightAnalyst' or 'next' is 'DocumentWriter':\n"
    "      - No input is needed in 'query'—these agents operate automatically based on prior outputs.\n"
)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(nodes=", ".join(nodes))

    chain = prompt | llm | JsonOutputParser()

    last_message = state["messages"][-1]
    print("\nmessages Supervisor ", last_message)    
    
    response = chain.invoke({"messages": state["messages"]})
    
    if (response['next'] == 'InsightAnalyst' or response['next'] == 'DocumentWriter'):        
        tool_message = HumanMessage(content=json.dumps(last_message.content))
        response['query'] = tool_message 
    
    print("\nresponse Supervisor ", response)    
    return {"next": response['next'], 'messages': response['query']}

def WebSearcher(state):
    system_prompt = (
    "You are a web searcher. "
    " You can use the tool to retrieve the relevent title and links related to provided query"
    " You must Never provide information that the user does not have."
    )

    messages = state["messages"][-1]
    print("\nmessages WebSearcher", messages)   
    
    response = llm_scrap_tool.invoke([system_prompt] + [HumanMessage(messages.content)])
    print("\nresponse websearcher", response)
    return {"messages": [response]}


def InsightAnalyst(state):
    system_prompt = (
     """You are a Insight Researcher. Do step by step. 
        Based on the provided content.
        Use Tool to extract/scrap the relvent content.""")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")])

    chain = prompt | llm_Insight_tool

    messages = state["messages"][-1]
    print("\nmessages InsightAnalyst ", messages)
    response = chain.invoke({"messages": [messages]})
    print("\nresponse InsightAnalyst ", response)
    return {"messages": [response]}


def should_continue(state):
    last_message = state["next"]
    print("\nshould_continue function --->", last_message)

    return last_message
 
def DocumentWriter(state):
    system_prompt = (
        """Use the DocumentWriter tool to create a document.
        and sent the relevent summary of the content.
        Use of tool is must        
        """)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")])

    chain = prompt | llm_DocumentWriter

    messages = state["messages"][-1]
    print("\nmessages DocumentWriter ", messages)
    response = chain.invoke({"messages": [HumanMessage(content=messages.content)]})

    print("\nresponse DocumentWriter ", response)
    return {"messages": [response]}


scrap_tool = ToolNode([InternetSearchTool])
Insight_tool = ToolNode([InsightResearcher])
writer_tool = ToolNode([DocumentWriterTool])
workflow = StateGraph(State)

workflow.add_node("Supervisor", Supervisor)
workflow.add_node("WebSearcher", WebSearcher)
workflow.add_node("InsightAnalyst", InsightAnalyst)
workflow.add_node("DocumentWriter", DocumentWriter)

workflow.add_node("scrap_tool", scrap_tool)
workflow.add_node("Insight_tool", Insight_tool)
workflow.add_node("writer_tool", writer_tool)


workflow.add_edge(START, "Supervisor")
workflow.add_conditional_edges(
    "Supervisor",
    should_continue,
    {
        "WebSearcher": "WebSearcher",
        "InsightAnalyst": "InsightAnalyst",
        "DocumentWriter": "DocumentWriter",
        "END": END
    }
)

workflow.add_edge("WebSearcher", "scrap_tool")
workflow.add_edge("InsightAnalyst", "Insight_tool")

workflow.add_edge("scrap_tool", "Supervisor")
workflow.add_edge("Insight_tool", "Supervisor")
workflow.add_edge("DocumentWriter", "writer_tool")
workflow.add_edge("writer_tool", END)


graph = workflow.compile()



def save_graph():
    graph_png_data = graph.get_graph().draw_mermaid_png()
    with open("./langgraph_output.png", "wb") as f:
        f.write(graph_png_data)


def run():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        events = graph.stream({"messages": [("user", user_input)]})
        print("\nevents ", events)
        for event in events:
            for value in event.values():
                print("\nAssistant:", value)

run()


