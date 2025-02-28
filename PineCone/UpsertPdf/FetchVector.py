from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

index_name = "pdfs"
index = pc.Index(index_name)

vec = index.fetch(ids=["vec0"], namespace="PMC4751334")
print(len(vec['vectors']))
print(len(vec['vectors']['vec0']['values']))