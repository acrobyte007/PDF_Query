# RAG Application with FastAPI and LangChain

This project implements a Retrieval-Augmented Generation (RAG) application using FastAPI, LangChain, and Mistral AI. It allows users to upload PDF or Word documents, store their text in a vector database, and perform similarity searches with language model-generated responses (e.g., translations). The application is accessible via a browser through FastAPI's Swagger UI.

## Features
- **Upload Documents**: Upload PDF or Word files and store their extracted text in a Chroma vector store, associated with a user ID.
- **Search and Generate**: Perform similarity searches on stored documents filtered by user ID and generate responses (e.g., translations) using Mistral AI.
- **Browser Access**: Interact with the API through an interactive Swagger UI at `http://localhost:8000/docs`.

## Prerequisites
- Python 3.8+
- Mistral AI API key (set as environment variable `MISTRAL_API_KEY`)

## Installation
1. **Clone the Repository** (or create the files manually):
   Ensure the following files are in your project directory:
   - `main.py`: FastAPI application with endpoints.
   - `vector_store.py`: Vector store management (document storage and retrieval).
   - `response.py`: RAG logic with Mistral AI.

2. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install fastapi uvicorn PyPDF2 python-docx langchain_chroma langchain_huggingface langchain_core langchain_text_splitters sentence-transformers langchain_mistralai
   ```

3. **Set Environment Variable**:
   Set your Mistral AI API key:
   ```bash
   export MISTRAL_API_KEY='your-api-key'
   ```
   On Windows, use:
   ```cmd
   set MISTRAL_API_KEY=your-api-key
   ```

## Project Structure
- `main.py`: Defines FastAPI endpoints for uploading documents and performing RAG searches.
- `vector_store.py`: Handles document storage and similarity search using Chroma and HuggingFace embeddings.
- `response.py`: Implements RAG by combining document retrieval with Mistral AI response generation.

## Usage
1. **Run the FastAPI Server**:
   Navigate to the project directory and run:
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:8000`. If port 8000 is in use, edit `main.py` to change the port (e.g., `port=8001`).

2. **Access in Browser**:
   Open a browser and navigate to:
   ```
   http://localhost:8000/docs
   ```
   This opens the Swagger UI, where you can test the API endpoints interactively.

3. **API Endpoints**:
   - **POST /upload-document/**:
     - **Purpose**: Upload a PDF or Word document and store its text in the vector store.
     - **Parameters**:
       - `file`: PDF or Word file (`.pdf` or `.docx`).
       - `user_id`: Unique identifier (e.g., UUID like `123e4567-e89b-12d3-a456-426614174000`).
     - **Example**:
       - In Swagger UI, click "Try it out", upload a file, enter a `user_id`, and execute.
       - Response:
         ```json
         {
           "message": "Document processed and stored successfully",
           "user_id": "123e4567-e89b-12d3-a456-426614174000"
         }
         ```
   - **POST /search/**:
     - **Purpose**: Perform a similarity search and generate a response (e.g., translation) using retrieved documents.
     - **Parameters** (JSON body):
       - `query`: Search query string.
       - `user_id`: Same user ID used in document upload.
       - `input_language`: Input language (default: "English").
       - `output_language`: Desired output language (default: "German").
     - **Example**:
       - In Swagger UI, enter:
         ```json
         {
           "query": "LangChain provides abstractions to make working with LLMs easy",
           "user_id": "123e4567-e89b-12d3-a456-426614174000",
           "input_language": "English",
           "output_language": "German"
         }
         ```
       - Response:
         ```json
         {
           "message": "Search and generation completed successfully",
           "user_id": "123e4567-e89b-12d3-a456-426614174000",
           "query": "LangChain provides abstractions to make working with LLMs easy",
           "response": "Ich liebe programmieren.",
           "documents": [
             {
               "content": "This is a sample text to be embedded.",
               "metadata": {"user_id": "123e4567-e89b-12d3-a456-426614174000"}
             }
           ]
         }
         ```

4. **Testing with cURL** (Alternative):
   - Upload a document:
     ```bash
     curl -X POST -F "file=@document.pdf" -F "user_id=123e4567-e89b-12d3-a456-426614174000" http://localhost:8000/upload-document/
     ```
   - Perform a search:
     ```bash
     curl -X POST http://localhost:8000/search/ -H "Content-Type: application/json" -d '{"query":"LangChain provides abstractions to make working with LLMs easy","user_id":"123e4567-e89b-12d3-a456-426614174000","input_language":"English","output_language":"German"}'
     ```

## Notes
- **Mistral AI API Key**: Required for `response.py`. Ensure it’s set in the environment or configured in `response.py`.
- **Document Requirements**: Uploaded files must be valid PDFs or Word documents with extractable text (e.g., not scanned PDFs without OCR).
- **User ID**: The `user_id` in the `/search/` endpoint must match the one used in `/upload-document/` to retrieve relevant documents.
- **Customization**: Modify the prompt in `response.py` to change the RAG behavior (e.g., for question answering instead of translation).
- **Vector Store**: Documents are stored in `./chroma_langchain_db`. Ensure write permissions in the directory.
- **Troubleshooting**:
  - **No Results**: Ensure documents are uploaded with the same `user_id` before searching.
  - **Port Conflicts**: Change the port in `main.py` if 8000 is in use.
  - **API Errors**: Check the server console for detailed error messages.

## Dependencies
- `fastapi`: API framework
- `uvicorn`: ASGI server
- `PyPDF2`: PDF text extraction
- `python-docx`: Word document text extraction
- `langchain_chroma`: Chroma vector store
- `langchain_huggingface`: HuggingFace embeddings
- `langchain_core`: LangChain core components
- `langchain_text_splitters`: Text splitting utilities
- `sentence-transformers`: Embedding models
- `langchain_mistralai`: Mistral AI integration

