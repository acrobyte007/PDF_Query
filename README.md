# Streamlit-Based PDF Knowledge Base with SBERT and Mistral API

This application allows users to create and manage projects for processing PDF documents, generating semantic embeddings using SBERT, and querying a knowledge base using the Mistral API for question answering.
Appliaction [https://huggingface.co/spaces/ajoy0071998/PDF_Query]
## Features

- **Project Management**: Create, delete, and manage projects.
- **PDF Upload**: Upload and process PDF files by extracting text.
- **Text Chunking**: Split extracted text into manageable chunks based on sentences and word limits.
- **Embedding Generation**: Generate embeddings for text chunks using SBERT (`all-MiniLM-L6-v2`).
- **FAISS Indexing**: Use FAISS for efficient similarity search across text embeddings.
- **Question Answering**: Query a selected project's knowledge base, retrieving the most relevant text chunks and leveraging the Mistral API for contextual answers.


## Tech Stack

- **Frontend**: Streamlit for user interface.
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`).
- **Search**: FAISS for approximate nearest-neighbor search.
- **API Integration**: Mistral API for question answering.
- **PDF Processing**: PyPDF2 for text extraction.



