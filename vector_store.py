import os
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, model_name, persist_directory="chroma_db"):
        # Get the current working directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.persist_directory = os.path.join(current_dir, persist_directory)
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.db = self._load_database() # Veritabanını yükleme veya oluşturma fonksiyonunu çağırıyoruz

    def _load_database(self):
        """Load database if it exists, otherwise return None."""
        if os.path.exists(self.persist_directory): # Klasörün varlığını kontrol et
            try:
                print(f"✅ Loading existing vector database from: {self.persist_directory}")
                db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
                return db # Mevcut veri tabanını yükle
            except Exception as e:
                print(f"⚠️ Warning: Error loading existing database, will create a new one. Error: {e}")
                return None # Yükleme hatası olursa None dön
        else:
            print(f"ℹ️ No existing vector database found at: {self.persist_directory}, will create a new one.")
            return None # Klasör yoksa None dön, create_database fonksiyonunda yeni oluşturulacak

    def create_database(self, chunks):
        """Create and persist vector database from text chunks."""
        if self.db is not None: # Eğer _load_database zaten yüklediyse, yeniden oluşturmaya gerek yok
            print("ℹ️ Vector database already loaded, skipping creation.")
            return True

        try:
            documents = [Document(page_content=chunk) for chunk in chunks]
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory
            )
            print("✅ Vector database created and persisted.")
            return True
        except Exception as e:
            print(f"❌ Error creating vector database: {e}")
            return False

    def similarity_search(self, query, k=3):
        """Perform similarity search on the vector database."""
        if not self.db:
            print("Database not initialized")
            return []
        try:
            return self.db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []