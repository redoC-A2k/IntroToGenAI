from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")
from time import time

index_name = "quickstart"
index = pc.Index(index_name)

print(index.describe_index_stats())