from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_core.documents import Document

def initialize_vector_store():
    """Initialize and return a Chroma vector store with HuggingFace embeddings."""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        
        # Initialize embeddings
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize vector store
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=hf,
            persist_directory="./chroma_langchain_db"
        )
        
        return vector_store
    
    except Exception as e:
        raise Exception(f"Error initializing vector store: {str(e)}")

def set_embeddings(text: str, user_id: str) -> bool:
    """
    Split text, create documents, and add them to the vector store.
    
    Args:
        text: Input text to be processed
        user_id: Unique identifier for the user
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not text or not user_id:
            raise ValueError("Text and user_id cannot be empty")
            
        # Initialize recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=350,
            chunk_overlap=50
        )
        
        # Split text into chunks
        texts = text_splitter.split_text(text)
        if not texts:
            raise ValueError("No text chunks created")
            
        # Initialize vector store
        vector_store = initialize_vector_store()
        
        # Create and add documents
        documents = [
            Document(
                page_content=text_chunk,
                metadata={"user_id": user_id},
                id=f"{user_id}_{i}"
            ) for i, text_chunk in enumerate(texts)
        ]
        
        # Add documents to vector store
        vector_store.add_documents(documents)
        return True
        
    except Exception as e:
        print(f"Error in set_embeddings: {str(e)}")
        return False

def search_similar_documents(query: str, user_id: str, k: int = 2) -> list:
    """
    Perform a similarity search in the vector store for a specific user_id.
    
    Args:
        query: The search query string
        user_id: Unique identifier for the user to filter results
        k: Number of results to return (default: 2)
        
    Returns:
        list: List of dictionaries containing document content and metadata
    """
    try:
        if not query or not user_id:
            raise ValueError("Query and user_id cannot be empty")
            
        # Initialize vector store
        vector_store = initialize_vector_store()
        
        # Perform similarity search with user_id filter
        results = vector_store.similarity_search(
            query,
            k=k,
            filter={"user_id": user_id}
        )
        
        # Return results as a list of dictionaries
        return [{"content": res.page_content, "metadata": res.metadata} for res in results]
        
    except Exception as e:
        print(f"Error in search_similar_documents: {str(e)}")
        return []