from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import dotenv
from pinecone import Pinecone
from typing import Annotated, List, TypedDict
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


dotenv.load_dotenv()

# pc = Pinecone(api_key = "")
pc = Pinecone()
index = pc.Index("pdfs")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks on the basis of context"),
    ("user", """Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} """)
])

embedding = PineconeEmbeddings(model = "multilingual-e5-large", query_params= {"input_type": "query"})
vector_store = PineconeVectorStore(index = index, embedding = embedding, namespace = "PMC4751334_SEN")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

@tool(response_format = "content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query"""
    retrieved_docs = vector_store.similarity_search(query, k = 3)
    # print("----",type(retrieved_docs),"----")
    serialized = "\n\n".join(f"Source: {doc.metadata}\n" f"Content: {doc.page_content}" for doc in retrieved_docs)
    return serialized, retrieved_docs

# Graph will consist of three steps
# 1. A node that fields the user input, either generating a query for the retriever or responding directly;
# 2. A node for the retriever tool that executes the retrieval step;
# 3. A node that generates the final response using the retrieved context.

model = ChatOpenAI(model = "gpt-4o-mini")

# Step 1
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or response"""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    # print("Respone : ",response)
    return {"messages": [response]}    

# Step 2
tools = ToolNode([retrieve])

# def generate(state: State):
#     docs_generate = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt_template.invoke({"question": state["question"], "context": docs_generate})
#     response = model.invoke(messages)
#     return {"answer": response.content}

# Step 3
def generate (state: MessagesState):
    """Generate the final response"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else :
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks"
        "Use the following peices of retrieved context to answer the question."
        "If you don't know the answer, say that you don't know."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human","system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    
    response = model.invoke(prompt)
    return {"messages": [response]}
        
# Old
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile(checkpointer=MemorySaver())

# result = graph.invoke({"question": "What is length of genome sequence ?"})
# print(f"Context : {result['context']}\n\n")

# query = "What is length of genome sequence ?"
# print(graph.invoke({"messages": [{"role":"user", "content": query}]}))
# for step in graph.stream({"messages": [{"role":"user", "content": query}]}, stream_mode="values"):
#     print(step["messages"][-1].pretty_print())

config = {"configurable": {"thread_id": "abc123"}}
while True:
    query = input("Enter your query : ")
    if query == "exit":
        break
    # result = graph.invoke({"messages": [{"role":"user", "content": query}]}, config)
    for step in graph.stream({"messages": [{"role":"user", "content": query}]}, stream_mode="values", config = config):
        print(step["messages"][-1].pretty_print())
    # print(result["messages"][-1].pretty_print())