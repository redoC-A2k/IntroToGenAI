from pinecone import Pinecone,ServerlessSpec
pc = Pinecone(api_key = "Your api key")

index_name = "pdfs"

pc.create_index(
    name=index_name,
    dimension=1024, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)