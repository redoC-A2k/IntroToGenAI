from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

query = "Tell me about the tech company known as Apple."
index = pc.Index("quickstart")

embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

results = index.query(
    namespace="ns1",
    vector=embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)