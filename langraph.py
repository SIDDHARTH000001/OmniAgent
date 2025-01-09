import boto3
import requests
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, ToolMessage,HumanMessage, RemoveMessage,AIMessage
from langgraph.graph import END
from typing import List,Literal
from langchain_aws import ChatBedrock
import tools
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", message=".*resource module not available on Windows.*")


class KnowledgeBase:
    def __init__(self):
        self.bm25_retriever = None
        self.bm25_corpus = None
        self.bm25_stemmer = None
        self.qdrant = None
        self.session_id = None


from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    get_type_hints,
)

class State(MessagesState):
    summary: str
    session_id: str
    search_method: str
    web_search: str
    docx_search: str
    db_search: str

from API.getllm import get_llm
# boto3.setup_default_session(
#     **requests.get("http://localhost:8000/awscred").json()    
# )

# client = ""
# llm = ""
# def renew():
#   global llm
#   global client
#   boto3.setup_default_session(
#       **requests.get("http://localhost:8000/awscred").json()    
#   )
#   client = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
#   llm = ChatBedrock(model_id ="anthropic.claude-3-5-sonnet-20240620-v1:0",
#                   client = client,
#                   model_kwargs={"max_tokens":5000,"temperature":0}
#                   #   guardrails={"id": "xpqvrjzg8jpl", "version": "5"}
#                   )
# renew()


def should_continue(state: State) -> Literal["summarize_conversation","__end__"] :

    """Return the next node to execute."""
    
    print("                                                                                         called...... should continue")
    messages = state["messages"]

    if len(messages) > 7 and not state["messages"][-1].tool_calls:
        return "summarize_conversation"
    
    return "__end__"

import CONSTANT

def summarize_conversation(state: State):
    try:
        print(CONSTANT.model_name,".........................................")
        llm = get_llm(CONSTANT.model_name)
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"
            
        # print("                                  ||                                          summary so far\n")
        # print("                                  ||                                          ",summary_message[:80],".....")
        
        if state["messages"][-1].tool_calls:
            messages = state["messages"][:-2] + [HumanMessage(content=summary_message)]
        else:
            messages = state["messages"] + [HumanMessage(content=summary_message)]
        
        llm_with_tool = llm.bind_tools([tools.AMO_tool,tools.fetch_AMO_screenshot,tools.stock_analysis,tools.search_web_tool,tools.from_documents])
        response = llm_with_tool.invoke(messages)
        
        till = -3 if isinstance(state["messages"][-2],ToolMessage) else -2
        print("==================================================")
        for i in state["messages"]:
            print(type(i),end=" => ")
        print("==================================================")

        # delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:till]] + state["messages"][-till:]
        delete_messages =  state["messages"][-till:]
        
        return {"summary": response.content, "messages": delete_messages}
    except Exception as e:
        print(f" ERROR while creating summary {e}")
        return {"summary": f"ERROR while creating summary {e}", "messages": delete_messages}



def master_node(state: State):
    # print(CONSTANT.model_name,".........................................")
    summary = state.get("summary", "")
    web_search = state.get("web_search",False)
    docx_search = state.get("docx_search",False)
    db_search = state.get("db_search",False)
    llm = get_llm(CONSTANT.model_name)

    if summary:
        prompt = f"You are a Master node which delegates the task and decides the flow of conversation, By calling appropriate tool if needed, Summary of conversation earlier: {summary}."
    else:
        prompt = "You are a Master node which delegates the task and decides the flow of conversation, By calling appropriate tool if needed." 
    
    
            
    prompt = [SystemMessage(content=prompt)] 

    advisor = ''
    if web_search or docx_search or db_search:
        advisor += "for this task user just enabled you with these tools ->"
    
    if web_search:
        advisor += "`search_web_tool tool`" 
        
    if docx_search:
        advisor += "`from_documents tool`"

    if db_search:
        advisor += "`db_query` and `get_db_structure tool` strictly once"
    
    prompt +=  f"""you have following tools use these accordinly:
                `search_web_tool` : to search web for latest info,
                `from_documents` : to seach the document,
                `stock_analysis` : to analyize the given stock,
                `AMO_tool` : to fill AMO time sheet,
                `fetch_AMO_screenshot`: for stock analysis,
                `ko_generation` : knowledge article generation from uploaded/given documents
                `get_db_structure` : helpful to understand the structure of database, must be used before directly querying to db  
                `db_query` :  if user asked a database related question, use this tool by converting user reuest into db executable query,
                in case if it helps today is: {tools.getdate("")}

                strcitly before answering anything just by yourself try to find answers in tools ;
                """
    # print("*****************************************")
    # prompt +=[AIMessage(advisor,name="Advisor")]
    # print("*****************************************")
    
    # prompt += [HumanMessage(advisor)]+state["messages"]
    prompt += state["messages"]
    print("in master_node................ ")

    llm_with_tool = llm.bind_tools([tools.from_documents,tools.search_web_tool,tools.AMO_tool,tools.db_query,tools.get_db_structure,
                                    tools.fetch_AMO_screenshot,tools.ko_generation,
                                    tools.stock_analysis
                                    ])
    response = llm_with_tool.invoke(prompt)

    return {"messages":[response]}


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        **state,
        "messages": state["messages"] + [
            ToolMessage(content=f"Error: {repr(error)}\n please fix your mistakes.", tool_call_id=tc["id"])
            for tc in tool_calls
        ],
    }
    
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


def _tool_condition(state: Union[list[AnyMessage], dict[str, Any], tools.BaseModel], messages_key: str = "messages") -> Literal["tools", "tools_end","__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        if ai_message.tool_calls[0]['name'] in ["fetch_AMO_screenshot","stock_analysis","ko_generation"]:
            return "tools_end"
        return "tools"
    return "__end__"
    


workflow = StateGraph(State)
workflow.add_node("master_node", master_node)
workflow.add_node(summarize_conversation)
workflow.add_node("tools",create_tool_node_with_fallback([tools.AMO_tool,tools.search_web_tool,tools.from_documents,tools.db_query,tools.get_db_structure]))
workflow.add_node("tools_end",create_tool_node_with_fallback([tools.fetch_AMO_screenshot,tools.stock_analysis,tools.ko_generation]))

workflow.add_edge(START, "master_node")
workflow.add_conditional_edges("master_node", _tool_condition)
workflow.add_edge("tools", "master_node")
workflow.add_conditional_edges("master_node", should_continue)
workflow.add_edge("summarize_conversation", END)
workflow.add_edge("tools_end", END)


memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
print("agent loaded...........")
# display(Image(graph.get_graph().draw_mermaid_png()))




# res = graph.invoke({"messages": """fetch 10 actors from actor table"""},config={"thread_id":1})
# res = graph.invoke({"messages": """in db when did the last time english language got updated"""},config={"thread_id":1})
# res = graph.invoke({"messages": """yes go ahead"""},config={"thread_id":1})

# # import json
# res["messages"][-1].content