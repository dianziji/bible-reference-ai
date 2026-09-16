from app.services.pinecone_client import get_or_create_index

def main():
    index = get_or_create_index(dimension=1536)
    print("Index ready:", index.describe_index_stats())

if __name__ == "__main__":
    main()