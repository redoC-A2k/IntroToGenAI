from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

index = pc.Index("pdf136kb")
index.delete(delete_all=True, namespace='136')
