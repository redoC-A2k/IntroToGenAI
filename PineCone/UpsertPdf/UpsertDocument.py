from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

index_name = "pdfs"
file_path = ("C:\\Users\\Vinayak\\Pinecone\\UpsertPdf\\PMC4751334.pdf")
loader = PyPDFLoader(file_path)

# for doc in loader.lazy_load():
#     # print(type(doc))
#     # print(doc.page_content)
#     # print("\n ------------------------------------------------- \n")
pages = loader.load()
# # # print(len(pages))
# # # print("\n")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 10)
documents = text_splitter.split_documents(pages)

# embeddings = pc.inference.embed(
#     model="multilingual-e5-large",
#     inputs=[d['text'] for d in data],
#     parameters={"input_type": "passage", "truncate": "END"}
# )
# print(len(documents))
vectors = []
id = 0
for doc in documents:
    # print(doc.page_content)
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[doc.page_content],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    vectors.append({
        "id": "vec"+str(id),
        "values": embedding[0]['values'],
        "metadata": {'text': doc.page_content}
    })
    id = id+1

    # print("\n ------------------------------------------------- \n")
index = pc.Index(index_name)

index.upsert(
    vectors=vectors,
    namespace="PMC4751334"
)