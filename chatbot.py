from typing import TypedDict , List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph , START , END
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI:{response.content}")
    return state



graph = StateGraph(AgentState)
graph.add_node("process" , process)
graph.add_edge(START , "process")
graph.add_edge("process" , END)

agent = graph.compile()


user_input = input("Enter:")
while user_input!= "exist":
    agent.invoke({"messages":[HumanMessage(content = user_input)]})
    user_input = input("Enter:")



