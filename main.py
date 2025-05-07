import os
from text_extractor import extract_pdf_text, load_config
from text_processor import split_text
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from request_concurrency import main as run_chat

def initialize_system():
    """
    Initialize the RAG system components.
    Returns:
        bool: True if initialization is successful, False otherwise
    """
    try:
        # Load configuration
        config = load_config()
        print("✅ Configuration loaded successfully")

        # Extract text from PDF
        pdf_path = config.get('pdf_path', 'project/paper1.pdf')
        text = extract_pdf_text(pdf_path)
        if not text:
            print("❌ Failed to extract text from PDF")
            return False
        print("✅ Text extracted from PDF")

        # Split text into chunks
        chunk_size = config.get('chunk_size', 1000)
        chunk_overlap = config.get('chunk_overlap', 200)
        chunks = split_text(text, chunk_size, chunk_overlap)
        if not chunks:
            print("❌ Failed to split text into chunks")
            return False
        print("✅ Text split into chunks")

        # Initialize vector store
        model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        persist_directory = config.get('persist_directory', 'chroma_db')
        vector_store = VectorStore(model_name=model_name, persist_directory=persist_directory)
        
        # Create and persist the database
        if not vector_store.create_database(chunks):
            print("❌ Failed to create vector database")
            return False
        print("✅ Vector database initialized (created or loaded).") # Daha doğru mesaj

        return True

    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        return False

def main():
    """
    Main function to run the entire system.
    """
    print("\n🚀 Starting RAG System...")
    print("-" * 50)

    # Check if config file exists
    if not os.path.exists("project/config.yaml"):
        print("❌ config.yaml not found! Please create a configuration file.")
        return

    # Initialize the system
    if not initialize_system():
        print("\n❌ System initialization failed. Exiting...")
        return

    print("\n✨ System initialized successfully!")
    print("Starting chat interface...\n")
    print("-" * 50)

    # Run the chat interface
    try:
        run_chat()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user")
    except Exception as e:
        print(f"\n❌ Error during chat: {str(e)}")

if __name__ == "__main__":
    main()


