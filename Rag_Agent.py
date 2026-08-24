from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()


# =========================
# Gemini LLM
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# =========================
# Gemini Embedding Model
# =========================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# =========================
# PDF
# =========================

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"PDF file not found: {pdf_path}"
    )


pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")

except Exception as e:
    print(f"Error loading PDF: {e}")
    raise


# =========================
# Chunking
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

print(f"PDF has been split into {len(pages_split)} chunks")


# =========================
# ChromaDB
# =========================

persist_directory = "./chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    print("Created ChromaDB vector store!")

except Exception as e:

    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# =========================
# Retriever
# =========================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# =========================
# Retriever Tool
# =========================

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns information
    from the Stock Market Performance 2024 PDF.
    """

    docs = retriever.invoke(query)

    if not docs:
        return (
            "I found no relevant information "
            "in the Stock Market Performance 2024 document."
        )

    results = []

    for i, doc in enumerate(docs):

        page_number = doc.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page_number, int):
            page_number += 1

        results.append(
            f"Document {i + 1}\n"
            f"Page: {page_number}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n".join(results)


# =========================
# Tools
# =========================

tools = [retriever_tool]

llm = llm.bind_tools(tools)


# =========================
# Agent State
# =========================

class AgentState(TypedDict):

    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]


# =========================
# System Prompt
# =========================

system_prompt = """
You are an intelligent AI assistant who answers questions
about Stock Market Performance in 2024 based on the PDF
document loaded into your knowledge base.

Use the retriever tool whenever the user's question
requires information from the PDF.

You can make multiple tool calls if necessary.

Do not hallucinate or invent information.

Base your answers on the retrieved information.

Always mention the relevant page number when information
from the PDF is used.

If the information is not available in the PDF,
clearly say that it was not found in the document.
"""


# =========================
# Tools Dictionary
# =========================

tools_dict = {
    t.name: t
    for t in tools
}


# =========================
# Check Tool Calls
# =========================

def should_continue(state: AgentState):

    result = state["messages"][-1]

    return (
        hasattr(result, "tool_calls")
        and len(result.tool_calls) > 0
    )


# =========================
# LLM Node
# =========================

def call_llm(state: AgentState) -> AgentState:

    messages = list(
        state["messages"]
    )

    messages = [
        SystemMessage(
            content=system_prompt
        )
    ] + messages

    message = llm.invoke(
        messages
    )

    return {
        "messages": [message]
    }


# =========================
# Tool Node
# =========================

def take_action(state: AgentState) -> AgentState:

    tool_calls = (
        state["messages"][-1].tool_calls
    )

    results = []

    for t in tool_calls:

        tool_name = t["name"]

        query = t["args"].get(
            "query",
            ""
        )

        print(
            f"\nCalling Tool: {tool_name}"
        )

        print(
            f"Query: {query}"
        )

        if tool_name not in tools_dict:

            print(
                f"Tool {tool_name} does not exist."
            )

            result = (
                "Incorrect Tool Name. "
                "Please select a valid tool."
            )

        else:

            result = tools_dict[
                tool_name
            ].invoke(query)

            print(
                f"Result length: "
                f"{len(str(result))}"
            )

        results.append(
            ToolMessage(
                tool_call_id=t["id"],
                name=tool_name,
                content=str(result)
            )
        )

    print(
        "\nTools execution complete."
    )

    print(
        "Returning results to Gemini..."
    )

    return {
        "messages": results
    }


# =========================
# Build LangGraph
# =========================

graph = StateGraph(
    AgentState
)


graph.add_node(
    "llm",
    call_llm
)


graph.add_node(
    "retriever_agent",
    take_action
)


# If Gemini requests a tool:
# LLM -> Retriever Agent
#
# If Gemini does not request a tool:
# LLM -> END

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)


# After the tool finishes:
# Retriever Agent -> LLM

graph.add_edge(
    "retriever_agent",
    "llm"
)


graph.set_entry_point(
    "llm"
)


# =========================
# Compile Graph
# =========================

rag_agent = graph.compile()


# =========================
# Run Agent
# =========================

def running_agent():

    print("\n==============================")
    print("     GEMINI 2.5 FLASH RAG")
    print("==============================")

    while True:

        user_input = input(
            "\nWhat is your question: "
        )

        if user_input.lower() in [
            "exit",
            "quit"
        ]:
            break

        messages = [
            HumanMessage(
                content=user_input
            )
        ]

        result = rag_agent.invoke(
            {
                "messages": messages
            }
        )

        print("\n================ ANSWER ================")

        print(
            result["messages"][-1].content
        )


running_agent()