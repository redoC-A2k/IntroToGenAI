from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

query = "Tell me about the tech company known as Apple."

embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

# print(len(embedding))
