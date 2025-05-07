from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size, chunk_overlap):
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []