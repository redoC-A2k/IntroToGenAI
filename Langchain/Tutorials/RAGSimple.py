from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import dotenv
from pinecone import Pinecone
from typing import Annotated, List, TypedDict
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langgraph.graph import START, StateGraph


dotenv.load_dotenv()

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

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k = 3)
    print("----",type(retrieved_docs),"----")
    return {"context": retrieved_docs}

model = ChatOpenAI(model = "gpt-4o-mini")

def generate(state: State):
    docs_generate = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt_template.invoke({"question": state["question"], "context": docs_generate})
    response = model.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What is length of genome sequence ?"})
# print(f"Context : {result['context']}\n\n")
for ctx in result['context']:
    print("--------------")
    print(ctx.page_content)
print(f"Answer : {result['answer']}")
