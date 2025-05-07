import requests
from typing import Optional
from retriever import Retriever

def generate_answer(query: str, vector_store, docker_port: int = 8080) -> str:
    """
    Generate an answer using RAG approach with Llama model running in Docker.
    
    Args:
        query (str): User question
        vector_store: Vector store instance
        docker_port (int): Docker container port number for Llama API
        
    Returns:
        str: Generated answer or error message
    """
    # Initialize retriever and get context
    try:
        retriever = Retriever(vector_store)
        retrieved_docs = retriever.get_relevant_context(query)
        
        if not retrieved_docs:
            return "I couldn't find any relevant information to answer your question."
    except Exception as e:
        return f"Error during retrieval: {e}"
    
    # Create prompt
    prompt = f"""Context: {retrieved_docs}

Question: {query}

Using the provided context, please answer the question. If the context doesn't contain relevant information, please say so.

Answer:"""
    
    # Send request to Llama model in Docker
    try:
        response = requests.post(
            f"http://localhost:{docker_port}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=30
        )
        response.raise_for_status()
        
        response_data = response.json()
        answer = response_data.get("response")
        
        if not answer:
            return "Received empty response from the model."
            
        return answer.strip()
        
    except requests.ConnectionError:
        return "Could not connect to the Docker container. Please ensure the container is running and the port is correct."
    except requests.Timeout:
        return "Request to Llama model timed out. Please try again."
    except requests.RequestException as e:
        return f"Error communicating with Docker container: {e}"
    except ValueError as e:
        return f"Error parsing model response: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
