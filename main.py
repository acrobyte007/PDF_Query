from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
from docx import Document as DocxDocument
from vector_db import set_embeddings
from response import generate_rag_response

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    user_id: str

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...), user_id: str = Form(...)):
    """
    Endpoint to upload a PDF or Word document and store its text in the vector store.
    
    Args:
        file: Uploaded PDF or Word file
        user_id: Unique identifier for the user
        
    Returns:
        JSON response indicating success or failure
    """
    try:
        # Validate file type
        if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            raise HTTPException(status_code=400, detail="Only PDF or Word files are allowed")
        
        # Validate user_id
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID cannot be empty")
        
        # Read file content
        text = ""
        if file.content_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file.file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        else:  # Word document
            doc = DocxDocument(file.file)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from the document")
        
        # Store text in vector store
        success = set_embeddings(text, user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document in vector store")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Document processed and stored successfully", "user_id": user_id}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/search/")
async def search_documents(query_request: QueryRequest):
    """
    Endpoint to perform a similarity search and generate an answer using RAG.
    
    Args:
        query_request: JSON body containing query and user_id
        
    Returns:
        JSON response with generated answer and retrieved documents
    """
    try:
        query = query_request.query
        user_id = query_request.user_id
        
        # Validate inputs
        if not query or not user_id:
            raise HTTPException(status_code=400, detail="Query and user_id cannot be empty")
        
        # Perform RAG (retrieve and generate)
        result = generate_rag_response(
            query=query,
            user_id=user_id,
            k=2
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Search and generation completed successfully",
                "user_id": user_id,
                "query": query,
                "response": result["response"],
                "documents": result["documents"]
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)