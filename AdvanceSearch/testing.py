from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from selenium.webdriver.support import expected_conditions as EC
from langchain_core.output_parsers import JsonOutputParser
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from selenium import webdriver
from dotenv import load_dotenv
from typing import Annotated
import time
import json
import os

from tools import  InternetSearchTool, InsightResearcher, DocumentWriterTool

# InternetSearchAnalyst (go thru different urls and create summary and give to InsightRearcher)
# InsightResearcher (indept search on each topic and will give use detailed report)
# duckduckgo search and beautiful soup
#Tools (search the internet, and read) duckduckgo_search, beautiful soup

class OutputState(TypedDict):
    status_code: int
    response: str
    
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
llm_scrap_tool = llm.bind_tools([InternetSearchTool])
llm_Insight_tool = llm.bind_tools([InsightResearcher])
llm_DocumentWriter = llm.bind_tools([DocumentWriterTool])

nodes = ["WebSearcher", "InsightAnalyst", "DocumentWriter" ,"HumanInALoop"]

def custom_reducer(left: list | None, right: list | None) -> list:
    
    if right in nodes:
        return right
    
    return "Error in custom Node"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: Annotated[str, custom_reducer]
    
def Supervisor(state):
    system_prompt = (
        "You are a supervisor, a ReACT agent managing tasks for {nodes} in an Advanced Search process.\n\n"
        "Your role is to oversee tasks without directly answering user queries.\n\n"
        
        "Task Order:\n"
        "  1. 'WebSearcher' - Scrapes URLs based on the user query.\n"
        "  2. 'InsightAnalyst' - Analyzes data from URLs provided by WebSearcher.\n"
        "  3. 'DocumentWriter' - Summarizes the retrieved data.\n\n"

        "Respond in JSON format with:\n"
        "  - 'next': the next agent to act\n"
        "  - 'query': instructions specific to the agent, if needed.\n\n"

        "Agent Instructions:\n"
        "  - If the user query is irrelevant (e.g., 'Hi', 'Who are you?'):\n"
        "      - Set 'next' to 'HumanInALoop' and 'query' to 'Please provide a query for advanced search.'\n"
        
        "  - If the query is a word or you think users want to do AdvanceSearch on it. Start the process.."
        
        "  - If 'next' is 'WebSearcher':\n"
        "      - Set 'query' to 'Scrape URLs for: user query'.\n"
        "      - If results are unclear, refine the query and assign 'WebSearcher' again.\n"
        "  - If 'next' is 'InsightAnalyst' or 'DocumentWriter':\n"
        "      - No input needed in 'query'.\n"
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


def HumanInALoop(state):
    last_message = state["messages"][-1]
    print("\nmessages HumanInALoop", last_message.content)   

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
 
def DocumentWriter(state) -> OutputState:
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
    return {'messages': [response]}

def CustomStateOutputResponse(state) -> OutputState:
    last_tool_message = json.loads(state['messages'][-1].content)
    
    print("last_tool_message ", last_tool_message)
    return {"status_code": last_tool_message['status'], "response": last_tool_message['response']}

def whatsappAutomation(state: OutputState):    
    
    print("\n state ", state)
    last_message = state
    if last_message['status_code'] == 500:
        print("Error in whatsappAutomation")
        return

    # Path to a custom user profile directory for Chrome
    chrome_options = webdriver.ChromeOptions()
    user_data_dir = os.path.expanduser("~") + "/.whatsapp-session"
    chrome_options.add_argument(f"--user-data-dir={user_data_dir}")  # Use persistent session data

    # Initialize Chrome WebDriver with the custom user profile
    driver = webdriver.Chrome(options=chrome_options)

    # Open WhatsApp Web
    driver.get('https://web.whatsapp.com')

    # Wait for the user to scan the QR code
    time.sleep(10)

    # Define the recipient's phone number or name
    contact_name = "Discipline"  # Name as it appears on WhatsApp
    document_path = "/home/arsal/Desktop/learning/Gen-AI/langGraph/AdvanceSearch/Document.docx"  # Path to the document on your local system

    try:
        # Search for the contact
        search_box = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]'))
        )
        search_box.click()
        search_box.send_keys(contact_name)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for chat to open and then click the attachment icon
        plus_icon = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//span[@data-icon="plus"]'))
        )
        plus_icon.click()
        
        # Upload the document
        document_icon = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//input[@accept="*"]'))
        )
        document_icon.send_keys(document_path)
        
        # Wait for the document to upload and then click the send button
        send_button = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//span[@data-icon="send"]'))
        )
        send_button.click()

        print("Document sent successfully!")

    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close the browser after a delay to see the result
        time.sleep(5)
        driver.quit()


scrap_tool = ToolNode([InternetSearchTool])
Insight_tool = ToolNode([InsightResearcher])
writer_tool = ToolNode([DocumentWriterTool])
workflow = StateGraph(State)

workflow.add_node("Supervisor", Supervisor)
workflow.add_node("HumanInALoop", HumanInALoop)
workflow.add_node("WebSearcher", WebSearcher)
workflow.add_node("InsightAnalyst", InsightAnalyst)
workflow.add_node("DocumentWriter", DocumentWriter)
workflow.add_node("whatsappAutomation", whatsappAutomation)
workflow.add_node("CustomStateOutputResponse", CustomStateOutputResponse)

workflow.add_node("scrap_tool", scrap_tool)
workflow.add_node("Insight_tool", Insight_tool)
workflow.add_node("writer_tool", writer_tool)


workflow.add_edge(START, "Supervisor")
workflow.add_conditional_edges(
    "Supervisor",
    should_continue,
    {
        "HumanInALoop": "HumanInALoop",
        "WebSearcher": "WebSearcher",
        "InsightAnalyst": "InsightAnalyst",
        "DocumentWriter": "DocumentWriter",
        "END": END
    }
)
workflow.add_edge("HumanInALoop", "Supervisor")
workflow.add_edge("WebSearcher", "scrap_tool")
workflow.add_edge("InsightAnalyst", "Insight_tool")

workflow.add_edge("scrap_tool", "Supervisor")
workflow.add_edge("Insight_tool", "Supervisor")
workflow.add_edge("DocumentWriter", "writer_tool")
workflow.add_edge("writer_tool", "CustomStateOutputResponse")
workflow.add_edge("CustomStateOutputResponse", "whatsappAutomation")
workflow.add_edge("whatsappAutomation", END)

checkpointer = MemorySaver() 
graph = workflow.compile(checkpointer=checkpointer, interrupt_after=["HumanInALoop"])


def save_graph():
    graph_png_data = graph.get_graph().draw_mermaid_png()
    with open("./langgraph_output.png", "wb") as f:
        f.write(graph_png_data)


# def run():
#     while True:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("\nGoodbye!")
#             break

#         events = graph.stream({"messages": [("user", user_input)]})
#         print("\nevents ", events)
#         for event in events:
#             for value in event.values():
#                 print("\nAssistant:", value)

def run():
    while(True):
        config = {"configurable": {"thread_id": "1"}}
        user_input = ""
        
        snapshot = graph.get_state(config)
        print("snapshot.next ", snapshot.next)
            
        if(len(snapshot.next) != 0 and snapshot.next[0] == 'Supervisor'):
            user_input = input("Ask a human in a loop: ")
            existing_message = snapshot.values["messages"][0]
            
            print("Message ID", existing_message.id)
            
            new_message = HumanMessage(
                content=user_input,
                id=existing_message.id)
            first_arg_message = None
            graph.update_state(config, {"messages": [new_message]})
        
        else: 
            user_input = input("User: ")
        
        first_arg_message = {"messages": [("user", user_input)]}    
        events = graph.stream(first_arg_message, config)

        for event in events:
            for value in event.values():
                print("\nAssistant:", value)


run()


