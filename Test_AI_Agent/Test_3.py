from typing import Annotated, TypedDict, Sequence, Union
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages #reducer function
from langgraph.prebuilt import ToolNode



load_dotenv()
client = Groq()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def addition(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """this is the addtion tool, it adds two numbers together and returns the result"""
    return x + y

@tool
def substraction(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """this is the subtraction tool, it subtracts two numbers and returns the result"""
    return x - y

@tool
def multiplication(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """this is the multiplication tool, it multiplies two numbers together and returns the result"""
    return x * y

@tool
def division(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """this is the division tool, it divides two numbers and returns the result"""
    return x / y

@tool
def power(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """this is the power tool, it raises x to the power of y and returns the result"""
    return x ** y


tools = [addition, substraction, multiplication, division, power]

llm = ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)
 

def model_fn_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = "You are a helpful AI assistant , please answer to the best of your ability.")
    response = llm.invoke([system_prompt] + state['messages'])
    return{"messages": [response]}

def should_continue(state: AgentState) :
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("Agent", model_fn_call)

Tool_node = ToolNode(tools=tools)
graph.add_node("Tools", Tool_node)

graph.add_conditional_edges("Agent", should_continue, {"continue": "Tools", "end": END})
graph.add_edge("Tools", "Agent")

graph.set_entry_point( "Agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message  = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input = {"messages": [("user", "Add 5 and 10 together and also divide by 4 and then raise to the power of 2")]}
print_stream(app.stream(input, stream_mode="values"))