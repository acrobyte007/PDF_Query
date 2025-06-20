from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from vector_db import search_similar_documents
from dotenv import load_dotenv
import os
load_dotenv()
def initialize_llm():
    """Initialize and return the Mistral AI language model."""
    try:
        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            max_retries=1,
        )
        return llm
    except Exception as e:
        raise Exception(f"Error initializing LLM: {str(e)}")

def generate_rag_response(query: str, user_id: str, k: int = 2) -> dict:
    """
    Retrieve relevant documents and generate an answer to the query using Mistral AI.
    
    Args:
        query: The question to be answered
        user_id: Unique identifier for the user
        k: Number of documents to retrieve (default: 2)
        
    Returns:
        dict: Contains the generated answer and retrieved documents
    """
    try:
        if not query or not user_id:
            raise ValueError("Query and user_id cannot be empty")
        
        # Retrieve relevant documents
        documents = search_similar_documents(query, user_id, k)
        if not documents:
            return {"response": "No relevant documents found to answer the query.", "documents": []}
        
        # Initialize LLM and prompt
        llm = initialize_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions based on the provided context. "
                    "Use only the information in the context to answer the question accurately and concisely. "
                    "If the context does not contain enough information to answer, say so.\n"
                    "Context:\n{context}"
                ),
                ("human", "{input}"),
            ]
        )
        
        # Create chain
        chain = prompt | llm
        
        # Combine retrieved documents into context
        context = "\n".join([doc["content"] for doc in documents])
        
        # Generate response
        result = chain.invoke(
            {
                "input": query,
                "context": context
            }
        )
        
        # Extract the response content
        response_content = result.content if hasattr(result, 'content') else str(result)
        
        return {
            "response": response_content,
            "documents": documents
        }
        
    except Exception as e:
        return {"response": f"Error generating response: {str(e)}", "documents": []}