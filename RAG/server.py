from fastapi import FastAPI, File, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

pc = Pinecone(api_key = "Your api key")
load_dotenv()
index_name = "pdfs"
# text_splitter = NLTKTextSplitter(chunk_size = 800,chunk_overlap = 10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 10)
app = FastAPI()

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant . You need to answer the question given by user using the context given from vector database. Answer the question referring and understanding the context using your own knowledge in 3-4 lines . CONTEXT : \"{context}\""), ("user", "QUESTION : \"{query}\"")]
)

index = pc.Index(index_name)

@app.post("/uploadFile/")
async def upload_file(file: UploadFile = File(...)):
    file = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(file)
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    WholeDocumentContent = ""
    for page in pages:
        page_content = page.page_content.replace("\n", " ")
        WholeDocumentContent += page_content
        WholeDocumentContent += " "
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
    index.upsert(
        vectors=vectors,
        namespace="UploadedFile"
    )
    return "File uploaded successfully"

@app.get("/ask")
async def query(query: str):
    print("Quick Query", query)
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="UploadedFile",
        vector=embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    knowledge = ""
    for match in results['matches']:
        text = match['metadata']['text']
        knowledge = knowledge + text
        knowledge = knowledge + ' '
    prompt = prompt_template.invoke({"context": knowledge, "query": query})
    response = ChatOpenAI().invoke(prompt.to_messages())  
    return response.content