from app.services.pinecone_client import get_index

index = get_index()
stats = index.describe_index_stats()
print(stats)