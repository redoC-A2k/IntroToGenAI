from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

index = pc.Index("pdfs")
query = "what is total length of the genome shotgun sequence"

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

print(results['matches'])
# print(len(results['vectors']['vec0']['metadata']['text']))