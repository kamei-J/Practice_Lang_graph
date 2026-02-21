import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()


class agentstate(TypedDict):
    messaage: list[HumanMessage]

llm = ChatGroq(
    model="llama-3.1-8b-instant")

def process(state: agentstate) -> agentstate:
    response = llm.invoke(state['messaage'])
    print(f'\nAI: {response.content}')
    return state

graph = StateGraph(agentstate)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

user_input = input("Enter your message: ")
while user_input != "exit":
    app.invoke({"messaage": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")