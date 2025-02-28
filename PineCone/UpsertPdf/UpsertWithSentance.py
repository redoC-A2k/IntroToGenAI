# from langchain_text_splitters import NLTKTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

index_name = "pdfs"
# text_splitter = NLTKTextSplitter(chunk_size = 800,chunk_overlap = 10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 10)
file_path = ("C:\\Users\\Vinayak\\Pinecone\\UpsertPdf\\PMC4751334.pdf")
loader = PyPDFLoader(file_path)

pages = loader.load()
WholeDocumentContent = ""
for page in pages:
    page_content = page.page_content.replace("\n", " ")
    # print("-------------------")
    # print(page_content)
    # print("-------------------")
    WholeDocumentContent += page_content
    WholeDocumentContent += " "
# print(WholeDocumentContent)

documents = text_splitter.split_text(WholeDocumentContent)
vectors = []
id = 0 

for doc in documents:
    # print(doc)
    # print("------------------------------------------------------")
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        # inputs=[doc.page_content],
        inputs=[doc],
        parameters={"input_type": "passage", "truncate": "END"},
    )
    vectors.append({
        "id": "vec"+str(id),
        "values": embedding[0]['values'],
        "metadata": {'text': doc}
    })
    id = id+1

index = pc.Index(index_name)

index.upsert(
    vectors=vectors,
    namespace="PMC4751334_SEN"
)