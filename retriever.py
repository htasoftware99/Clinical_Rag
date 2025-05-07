from typing import List, Optional
from langchain.schema import Document
from vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the retriever with a vector store instance.
        
        Args:
            vector_store (VectorStore): An initialized vector store instance
        """
        self.vector_store = vector_store
        
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
            
    def retrieve_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """
        Retrieve relevant documents with their similarity scores.
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
            
        Returns:
            List[tuple]: List of (document, score) tuples
        """
        try:
            # Note: We need to modify VectorStore to add similarity_search_with_scores method
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during scored retrieval: {e}")
            return []
    
    def get_relevant_context(self, query: str, k: int = 3) -> Optional[str]:
        """
        Get concatenated context from retrieved documents.
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
            
        Returns:
            Optional[str]: Concatenated context or None if no results
        """
        documents = self.retrieve(query, k=k)
        if not documents:
            return None
            
        # Combine all retrieved document contents
        context = "\n\n".join(doc.page_content for doc in documents)
        return context
