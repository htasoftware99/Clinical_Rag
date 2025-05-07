import argparse
from agent import ChatAgent
from vector_store import VectorStore

def validate_port(port: int) -> bool:
    """
    Validate if the port number is valid.
    
    Args:
        port (int): Port number to validate
        
    Returns:
        bool: True if port is valid, False otherwise
    """
    return 0 < port < 65536

def get_port_from_user() -> int:
    """
    Get and validate port number from user input.
    
    Returns:
        int: Valid port number
    """
    while True:
        try:
            port = input("Enter the Docker port number for LLM (default: 8080): ").strip()
            
            # Use default port if input is empty
            if not port:
                return 8080
            
            port = int(port)
            if validate_port(port):
                return port
            else:
                print("❌ Invalid port number! Port must be between 1 and 65535.")
        except ValueError:
            print("❌ Please enter a valid number!")

def main():
    """
    Main function to get port and run the chat agent.
    """
    try:
        # Get port number from user
        port = get_port_from_user()
        print(f"\n🚀 Initializing Chat Agent on port {port}...")
        
        # Initialize vector store
        try:
            vector_store = VectorStore(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"❌ Error initializing vector store: {e}")
            return
        
        # Initialize and run chat agent with the specified port
        agent = ChatAgent(vector_store=vector_store, docker_port=port)
        
        print("\n✅ Chat Agent initialized successfully!")
        print("Starting conversation...\n")
        
        # Run the agent
        agent.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")

if __name__ == "__main__":
    main()
