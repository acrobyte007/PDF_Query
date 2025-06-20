from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from vector_db import search_similar_documents

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

def generate_rag_response(query: str, user_id: str, input_language: str = "English", output_language: str = "German", k: int = 2) -> dict:
    """
    Retrieve relevant documents and generate a response using Mistral AI.
    
    Args:
        query: The search query string
        user_id: Unique identifier for the user
        input_language: Language of the input query (default: English)
        output_language: Desired language for the response (default: German)
        k: Number of documents to retrieve (default: 2)
        
    Returns:
        dict: Contains the generated response and retrieved documents
    """
    try:
        if not query or not user_id:
            raise ValueError("Query and user_id cannot be empty")
        
        # Retrieve relevant documents
        documents = search_similar_documents(query, user_id, k)
        if not documents:
            return {"response": "No relevant documents found.", "documents": []}
        
        # Initialize LLM and prompt
        llm = initialize_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that translates {input_language} to {output_language}. "
                    "Use the following context to inform your response:\n{context}"
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
                "input_language": input_language,
                "output_language": output_language,
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