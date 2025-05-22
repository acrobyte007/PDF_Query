
from embeddings import set_embedding
from embeddings import get_chunks
from user_data import user_doc,get_docs_by_user
from response import get_answer
from PyPDF2 import PdfReader
import os
import shutil


def extract_pdf_with_user(user_id: str,pdf_path: str, name: str) -> tuple:
    # Set new PDF path based on provided name
    dir_path = os.path.dirname(pdf_path)
    new_pdf_path = os.path.join(dir_path, f"{name}.pdf")
    
    if pdf_path != new_pdf_path:
        shutil.copy(pdf_path, new_pdf_path) 
    reader = PdfReader(new_pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    set_embedding(text,new_pdf_path,user_id)
    user_doc(user_id, new_pdf_path)
    return get_docs_by_user(user_id)

    ##return (user_id, text,pdf_path)

def get_final_aswer(query: str, user_id: str,pdf_path: str) -> list:
    chunks = get_chunks(query,user_id,pdf_path)
    full_text = ""
    for chunk in chunks:
        full_text += chunk
    
    final_answer = get_answer(query,full_text)

    return final_answer





