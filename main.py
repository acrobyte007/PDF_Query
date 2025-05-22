import streamlit as st
import os
import shutil
from chain import extract_pdf_with_user, get_final_aswer

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("PDF Extraction and Question Answering")

# --- Step 1: Upload PDF and extract ---

st.header("Upload PDF and extract text")

user_id = st.text_input("User ID")
name = st.text_input("Save PDF name (without extension)")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if st.button("Extract PDF"):

    if not user_id or not name or not uploaded_file:
        st.error("Please provide User ID, name, and upload a PDF file.")
    else:
        # Save uploaded PDF
        upload_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        # Call your extraction function
        documents = extract_pdf_with_user(user_id, upload_path, name)
        st.success("PDF processed!")
        st.write("Extracted documents:")
        st.write(documents)

# --- Step 2: Ask a question ---

st.header("Ask a question")

query = st.text_input("Enter your question")

if st.button("Get Answer"):

    if not user_id or not name or not query:
        st.error("Please provide User ID, name, and your question.")
    else:
        pdf_path = os.path.join(UPLOAD_DIR, f"{name}.pdf")
        if not os.path.exists(pdf_path):
            st.error(f"No PDF found with name '{name}.pdf'. Please upload and extract first.")
        else:
            answer = get_final_aswer(query, user_id, pdf_path)
            st.success("Answer generated!")
            st.write(answer)
