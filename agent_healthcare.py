import os
import sqlite3
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
import json
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please set it in the environment variables.")

# Database path
db_path = r"C:\Users\KIIT\Desktop\healthcare\healthcare_data.db"

# Initialize Langchain SQL Database
db = SQLDatabase.from_uri(f"sqlite:///{db_path}", include_tables=["patient_info"])

# Initialize ChatGroq LLM
llm = ChatGroq(model="llama3-70b-8192", temperature=0.1, api_key=GROQ_API_KEY)

# Create the SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    allow_modifications=True
)
def generate_sql_query(user_message):
    """Use LLM to convert user input into an SQL query."""
    sql_prompt = (
        f"Translate the user request into a highly optimized SQL SELECT query.\n"
        f"- Ensure efficient indexing by selecting only relevant columns.\n"
        f"- Apply LOWER() for case-insensitive searches.\n"
        f"- Return ONLY the SQL query—NO explanations or comments.\n\n"
        f"User Request: '{user_message}'"
    )

    sql_query = llm.invoke([HumanMessage(content=sql_prompt)])
    return sql_query.content.strip()

def ask_sql_agent(query):
    """Ask the SQL agent a question about patient_info."""
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"].strip()  # Ensure clean output
    except Exception as e:
        return f"Error: {str(e)}"

# RAG Agent for Healthcare Policies
VECTORDB_DIR = r"C:\Users\KIIT\Desktop\healthcare\vectordb\healthcare_vectordb"
K = 2

@tool
def lookup_policy(query: str) -> str:
    """Retrieves healthcare policy information based on user queries."""
    vectordb = Chroma(
        collection_name="healthcare_policies",
        persist_directory=VECTORDB_DIR,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    docs = vectordb.similarity_search(query, k=K)
    return "\n\n".join([doc.page_content for doc in docs])

# Define Memory State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build LangGraph StateGraph
graph_builder = StateGraph(State)

def chatbot(state: State):
    user_message = state["messages"][-1].content
    sql_query = generate_sql_query(user_message)
    if "SELECT" in sql_query.upper():  # Prioritize SQL Execution
        sql_response = ask_sql_agent(sql_query)
        if "Error:" not in sql_response and sql_response.strip():
            return {"messages": [ToolMessage(content=sql_response, name="SQLQuery", tool_call_id="1")]}  # No JSON wrapping

    # If SQL fails or isn't relevant, check vector DB
    policy_response = lookup_policy.invoke(user_message)
    if policy_response.strip():
        return {"messages": [ToolMessage(content=policy_response, name="PolicyLookup", tool_call_id="2")]}

    # If no SQL or vector DB results, use LLM
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# Tool Execution Node
class BasicToolNode:
    """Executes tools requested by AI."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        # Ensure the message is an AI response with a tool request
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return {"messages": [message]}  # No tools requested, return message as is

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[lookup_policy])
graph_builder.add_node("tools", tool_node)
def should_continue(state: State) -> str:
    messages = state['messages']
    last_message = messages[-1]

    # Ensure last_message has tool_calls before checking
    if isinstance(last_message, HumanMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END

graph_builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Memory Management
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Chatbot Interaction
config = {"configurable": {"thread_id": "1"}}
while True:
    user_input = input("Enter your message (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    final_state = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    print(final_state["messages"][-1].content)
