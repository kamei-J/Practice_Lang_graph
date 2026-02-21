from typing import TypedDict, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

class agentstate(TypedDict):
    messaage: list[Union[HumanMessage, AIMessage]]
     

llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0)

def process(state: agentstate) -> agentstate:
    response = llm.invoke(state['messaage'])
    state['messaage'].append(AIMessage(content=response.content))
    print("CURRENT ENTERED MESSAGES:" , state['messaage'])
    print(f'\nAI: {response.content}')
    return state

graph = StateGraph(agentstate)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

conversation_history = []


user_input = input("Enter your message: ")

while user_input != "exit":

    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messaage": conversation_history})
    conversion_history  = result["messaage"]
    
    user_input = input("Enter your message: ")

    if len(conversation_history) >= 5:
        conversation_history.pop(0)  # Remove the oldest message
    


with open("conversation_history.txt", "w") as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\n")

print("Conversation history saved to conversation_history.txt")