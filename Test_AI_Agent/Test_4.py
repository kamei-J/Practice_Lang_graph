import os
from typing import Annotated, TypedDict, Sequence, Union
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages #reducer function
from langgraph.prebuilt import ToolNode

load_dotenv()
client = Groq()

document_content = ""
class agentstate(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """"update the state with new content"""
    global document_content
    document_content = content
    return f"Document updated successfully. \n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current content to a file.
    
    Args:
        filename (str): The name of the file to save the content to.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename += ".txt"
 
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
            print(f"Content saved to {filename}")
        return f"Content saved to {filename}"
    except Exception as e:
        print(f"Error saving content: {str(e)}")

    
tools = [update, save]
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(tools)

def agent(state: agentstate) -> agentstate:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

                        
    if not state["messages"]:
        user_input = input("i am your drafting assistant. \n how can i help you with your document today? ")
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("what else would you like to do with your document? ")
        print(f"\n User: {user_input}")
        user_message = HumanMessage(content=user_input)
        
            

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)

    print(f'\nAI: {response.content}')

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n USING TOOL:{[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: agentstate) -> str:
    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(state["messages"]):
        if (isinstance(message,ToolMessage)):
            if "saved" in message.content.lower():
                return "end"
    return "continue"

def print_message(messages):
    """prints the message in more readable format"""

    if not messages:
        return
    for message in messages: 
        if isinstance(message, ToolMessage):
            print(f"\nTOOL MESSAGE: {message.content}")
    

graph = StateGraph(agentstate)

graph.add_node("Agent", agent)
graph.add_node("Tools", ToolNode(tools))
graph.set_entry_point("Agent")
graph.add_edge("Agent","Tools")
graph.add_conditional_edges(
    "Tools", 
    should_continue,
    {"continue": "Agent",
      "end": END})

app = graph.compile()

def run_document_agent():
    print(" \n====DRAFTER AGENT STARTED====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])
    print("\n ====ENDED====")

if __name__ == "__main__":
    run_document_agent()