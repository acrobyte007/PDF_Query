from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from chain import extract_pdf_with_user, get_final_aswer

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract/")
async def extract_pdf(
    user_id: str = Form(...),
    name: str = Form(...),
    uploaded_file: UploadFile = File(...)
):
    if not uploaded_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    upload_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    documents = extract_pdf_with_user(user_id, upload_path, name)
    return {"message": "PDF processed!", "documents": documents}


@app.post("/answer/")
async def get_answer(
    user_id: str = Form(...),
    name: str = Form(...),
    query: str = Form(...)
):
    pdf_path = os.path.join(UPLOAD_DIR, f"{name}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"No PDF found with name '{name}.pdf'. Please upload and extract first.")
    
    answer = get_final_aswer(query, user_id, pdf_path)
    return {"message": "Answer generated!", "answer": answer}
