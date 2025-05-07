from chromadb import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def get_embedding(self, text):
        """Generate embeddings for given text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.detach().numpy()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
        
# def embedding_func():
#     embedding = Embeddings(model="model_name")

#     return embedding

def embedding_func():
    generator = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return generator.get_embedding

