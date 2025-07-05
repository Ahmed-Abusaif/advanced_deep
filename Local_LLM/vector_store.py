import chromadb
from sentence_transformers import SentenceTransformer

class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("course_material")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def store_embeddings(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.encoder.encode(texts)
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=[f"chunk_{i}" for i in range(len(texts))]
        )
    
    def search(self, query, k=3):
        query_embedding = self.encoder.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        return results['documents'][0]
