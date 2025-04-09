import os
import sys
import logging
from dotenv import load_dotenv
from textwrap import dedent  # Updated to directly import dedent

###############################################################################
# 1. LOGGING SETUP: everything logs to rag_debug.log
###############################################################################
LOG_FILENAME = os.path.join(os.getcwd(), "agent.log")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("rag_logger")
logger.info("Logger initialized. Writing debug info to '%s'", LOG_FILENAME)

###############################################################################
# 2. LOAD ENV (so we get OPENAI_API_KEY, etc.)
###############################################################################
load_dotenv()

###############################################################################
# 3. SET UP CHROMA + RETRIEVER + TOOLS
###############################################################################
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# https://docs.trychroma.com/updates/troubleshooting#sqlite
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    logger.warning("pysqlite3 not installed. Development envierement.")

logger.info("Initializing Chroma vectorstore from './chromadb'...")

vectorstore = Chroma(
    persist_directory="./chromadb",
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings(),  # requires OPENAI_API_KEY
)
retriever = vectorstore.as_retriever()
logger.info("Retriever ready.")

retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_sber2023",
    description="Search and return info about Sber 2023 report",
)
tools = [retriever_tool]
logger.info("Retriever tool created.")

###############################################################################
# 4. DEFINE THE AGENT STATE + NODES
###############################################################################
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The 'messages' field is appended via add_messages
    messages: Annotated[Sequence[BaseMessage], add_messages]


# --- Node 1: grade_documents ---
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines the relevance of retrieved documents to the user's question.

    Args:
        state (dict): The current state of the agent, including messages.

    Returns:
        Literal["generate", "rewrite"]: 'generate' if documents are relevant,
        'rewrite' if they are not.
    """
    logger.debug("grade_documents: Checking retrieved docs relevance.")

    class Grade(BaseModel):
        binary_score: str = Field(description="Either 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=False)
    llm_with_output = model.with_structured_output(Grade)

    prompt = PromptTemplate(
        template=dedent("""\
            You are a grader. Decide if the retrieved content is relevant to the user's question.
            Document:
            {context}

            Question:
            {question}

            Answer with 'yes' or 'no' in the 'binary_score' field.
        """),
        input_variables=["context", "question"],
    )

    messages = state["messages"]
    last_message = messages[-1]

    # The first user question is messages[0]
    question = messages[0].content
    docs = last_message.content

    chain = prompt | llm_with_output
    graded = chain.invoke({"question": question, "context": docs})
    score = graded.binary_score.lower().strip()

    if score == "yes":
        logger.debug("grade_documents => 'generate' (docs are relevant)")
        return "generate"
    else:
        logger.debug("grade_documents => 'rewrite' (docs are NOT relevant)")
        return "rewrite"


# --- Node 2: agent (decide whether to retrieve or not) ---
def agent_node(state):
    """
    Decides the next step in the pipeline or invokes a tool.

    Args:
        state (dict): The current state of the agent, including messages.

    Returns:
        dict: Updated state with new messages or tool outputs.
    """
    logger.debug("agent_node: Deciding next step or using a tool.")
    from langchain_core.messages import BaseMessage
    from langchain_openai import ChatOpenAI

    messages = state["messages"]

    # A ChatOpenAI model that can call the retriever_tool if it decides
    model = ChatOpenAI(temperature=0, streaming=False, model="gpt-4o-mini")
    model = model.bind_tools(tools)

    response = model.invoke(messages)
    return {"messages": [response]}


# --- Node 3: rewrite ---
from langchain_core.messages import HumanMessage

def rewrite(state):
    """
    Rewrites the user's question for better retrieval if documents are irrelevant.

    Args:
        state (dict): The current state of the agent, including messages.

    Returns:
        dict: Updated state with the rewritten question.
    """
    logger.debug("rewrite: Rewriting question for better retrieval.")
    messages = state["messages"]
    original_question = messages[0].content

    # We'll just do a simple re-ask with ChatOpenAI
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=False)
    rewrite_prompt = [
        HumanMessage(
            content=dedent(f"""\
                Rewrite the question for clarity:

                Original Question:
                {original_question}
            """)
        )
    ]
    response = model.invoke(rewrite_prompt)
    return {"messages": [response]}


# --- Node 4: generate ---
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

def generate(state):
    """
    Generates the final answer using retrieved documents and the user's question.

    Args:
        state (dict): The current state of the agent, including messages.

    Returns:
        dict: Updated state with the generated answer.
    """
    logger.debug("generate: Creating final answer from docs.")
    messages = state["messages"]
    user_question = messages[0].content
    last_message = messages[-1]
    retrieved_docs = last_message.content

    # We'll pull a RAG prompt from somewhere (like a hub), or you can define your own
    prompt_template = hub.pull("rlm/rag-prompt")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)
    rag_chain = prompt_template | model | StrOutputParser()

    final_answer = rag_chain.invoke({"context": retrieved_docs, "question": user_question})
    return {"messages": [final_answer]}

###############################################################################
# 5. BUILD THE GRAPH
###############################################################################
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode

logger.debug("Building state graph for RAG pipeline...")

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("agent", agent_node)
retrieve_node = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()
logger.debug("State graph compiled successfully.")

###############################################################################
# 6. PUBLIC FUNCTION: run_rag_agent(question) -> str
###############################################################################
def run_rag_agent(question: str) -> str:
    """
    Executes the RAG pipeline for a given user question.

    Args:
        question (str): The user's input question.

    Returns:
        str: The final answer generated by the pipeline.
    """
    logger.info("run_rag_agent called with question: %s", question)

    inputs = {"messages": [("user", question)]}
    final_output = None

    # stream(...) yields step outputs
    config = {"recursion_limit": 5}
    for step_output in graph.stream(inputs, config=config):
        final_output = step_output

    if not final_output:
        logger.warning("No output from the pipeline. Returning empty response.")
        return "No response produced."

    # Extract the final text from the last node
    answer_text = None
    for node_name, node_data in final_output.items():
        if isinstance(node_data, dict) and "messages" in node_data:
            msgs = node_data["messages"]
            if msgs:
                last_msg = msgs[-1]
                if isinstance(last_msg, str):
                    answer_text = last_msg
                else:
                    # If it's a message object
                    answer_text = getattr(last_msg, "content", None)

    answer_text = answer_text or "No answer generated."
    logger.info("Final RAG answer: %s", answer_text)
    return answer_text
