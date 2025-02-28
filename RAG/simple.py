from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import dotenv
from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

dotenv.load_dotenv()
index = pc.Index("pdfs")

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant . You need to answer the question given by user using the context given from vector database. Answer the question referring and understanding the context using your own knowledge in 3-4 lines . CONTEXT : \"{context}\""), ("user", "QUESTION : \"{query}\"")]
)

while True:
    print("-------------------")
    query = input("Enter your query: ")
    if(query == "exit" or query == "quit"):
        break
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="PMC4751334_SEN",
        vector=embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    knowledge = ""
    for match in results['matches']:
        text = match['metadata']['text']
        # print(text)
        knowledge = knowledge + text
        knowledge = knowledge + ' '

    prompt = prompt_template.invoke({"context": knowledge, "query": query})
    print("-------------------")
    # print(prompt.to_messages())
    response = ChatOpenAI().invoke(prompt.to_messages())  
    print(response.content)
    print("-------------------")

# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", "Translate the following from english into {language}"), ("user", "{text}")]
# )

# prompt = prompt_template.invoke({"language": "Italian", "text": "hi! how are you?"})


# response = model.invoke(prompt.to_messages())
# print(response.content)