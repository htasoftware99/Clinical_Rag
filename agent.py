from rag_answer import generate_answer
from vector_store import VectorStore
from embedding_generator import embedding_func
from langchain_chroma import Chroma

class ChatAgent:
    def __init__(self, vector_store: VectorStore, docker_port: int = 8080):
        """
        Initialize chat agent.
        
        Args:
            vector_store (VectorStore): Vector store instance
            docker_port (int): Docker port for LLM
        """
        self.vector_store = vector_store
        self.docker_port = docker_port

    def run(self):
        """
        Run the chat loop.
        """
        print("🤖 Chat Agent Started!")
        print("Type 'exit' to end the conversation.")
        print("-" * 50)

        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n🤖 Goodbye! Chat ended.")
                break

            # Skip empty inputs
            if not user_input:
                print("Please enter a question!")
                continue

            try:
                # Generate answer using RAG
                db = Chroma(persist_directory=self.vector_store.persist_directory,
                             embedding_function=self.vector_store.embedding_model)
                answer = generate_answer(
                    query=user_input,
                    vector_store=db,
                    docker_port=self.docker_port
                )

                # Print the response
                print("\n🤖 Assistant:", answer)
                print("-" * 50)

            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again!")

def main():
    """
    Main function to run the chat agent.
    """
    try:
        # Initialize vector store (you'll need to configure this)
        vector_store = VectorStore(persist_directory="project/chroma_db")
        
        # Initialize and run chat agent
        agent = ChatAgent(vector_store=vector_store)
        agent.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Chat ended by user interruption.")
    except Exception as e:
        print(f"\n❌ Error initializing chat agent: {str(e)}")

if __name__ == "__main__":
    main()
