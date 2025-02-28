from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings

load_dotenv()
data = [
    "Apple is a popular fruit known for its sweetness and crisp texture.",
    "The tech company Apple is known for its innovative products like the iPhone.",
    "Many people enjoy eating apples as a healthy snack.",
    "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
    "An apple a day keeps the doctor away, as the saying goes.",
    "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."
]

# embeddings = pc.inference.embed(
#     model="multilingual-e5-large",
#     inputs=[d['text'] for d in data],
#     parameters={"input_type": "passage", "truncate": "END"}
# )
embeddings = PineconeEmbeddings(model = "multilingual-e5-large",query_params = {"input_type": "passage", "truncate": "END"})
print(embeddings)

embedding_docs= []
i=0
for i in range(len(data)):
    doc = Document(id=str(i), page_content=data[i])
    embedding_docs.append(doc)
    # embedding_vec.append(vec['embedding'])

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(embedding_docs)

result = vector_store.similarity_search(query="Tell me about the tech company known as Apple.", k=3)
# print(result)
for doc in result:
    print(doc)