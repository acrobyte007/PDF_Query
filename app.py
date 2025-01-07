import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import json
from PyPDF2 import PdfReader
import re
import hashlib

# Set Mistral API Key
MISTRAL_API_KEY = "API KEY"

# Initialize SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Faiss index
def create_faiss_index(dim=384):
    return faiss.IndexFlatL2(dim)

# Cache for repeated queries
cache = {}

# Function to generate SBERT embeddings
def generate_sbert_embeddings(texts):
    return sbert_model.encode(texts)

# Function to query Mistral API using curl
def query_mistral(prompt, MISTRAL_API_KEY):
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}]
    }
    curl_command = [
        "curl",
        "--location", "https://api.mistral.ai/v1/chat/completions",
        "--header", "Content-Type: application/json",
        "--header", "Accept: application/json",
        "--header", f"Authorization: Bearer {MISTRAL_API_KEY}",
        "--data", json.dumps(payload)
    ]
    response = subprocess.run(curl_command, capture_output=True, text=True)
    if response.returncode == 0:
        try:
            response_json = json.loads(response.stdout)
            return response_json['choices'][0]['message']['content']
        except (KeyError, json.JSONDecodeError):
            return "Error parsing the LLM response."
    return f"Error: {response.stderr}"

# Function to split text into chunks based on sentences and word limits
def chunk_text_by_sentence(text, max_words=300):
    sentences = re.split(r'(?<=\.)\s+', text)
    chunks, current_chunk, current_word_count = [], [], 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_word_count = [sentence], word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Initialize project storage
if "projects" not in st.session_state:
    st.session_state.projects = {}

# Sidebar: Project Management
st.sidebar.header("Project Management")
project_name = st.sidebar.text_input("Enter Project Name:")
if st.sidebar.button("Create Project") and project_name:
    if project_name not in st.session_state.projects:
        st.session_state.projects[project_name] = {
            "path": f"projects/{project_name}",
            "texts": [],
            "embeddings": None,
            "index": create_faiss_index(),
        }
        os.makedirs(st.session_state.projects[project_name]["path"], exist_ok=True)
        st.sidebar.success(f"Project '{project_name}' created!")
    else:
        st.sidebar.warning("Project already exists.")

# List existing projects
projects = list(st.session_state.projects.keys())
selected_project = st.sidebar.selectbox("Select a Project", projects)

# Delete a project
if st.sidebar.button("Delete Selected Project") and selected_project:
    del st.session_state.projects[selected_project]
    st.sidebar.success(f"Project '{selected_project}' deleted!")

# Main Section
if selected_project:
    st.header(f"Manage Project: {selected_project}")

    # File Upload for PDFs
    project_path = st.session_state.projects[selected_project]["path"]
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        texts = []
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            full_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            chunks = chunk_text_by_sentence(full_text)
            texts.extend(chunks)
        st.session_state.projects[selected_project]["texts"].extend(texts)

        # Generate embeddings and store in Faiss
        embeddings = generate_sbert_embeddings(texts)
        index = st.session_state.projects[selected_project]["index"]
        index.add(np.array(embeddings).astype(np.float32))
        st.session_state.projects[selected_project]["embeddings"] = embeddings
        st.success(f"Uploaded and processed {len(uploaded_files)} PDFs.")

    # Question-Answer Interface
    question = st.text_input("Enter your question:")
    if question:
        # Check cache for repeated queries
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        if question_hash in cache:
            st.write("Cached Answer:")
            st.write(cache[question_hash])
        else:
            # Generate embedding for the question
            query_embedding = generate_sbert_embeddings([question])[0].reshape(1, -1)
            index = st.session_state.projects[selected_project]["index"]
            D, I = index.search(query_embedding, k=5)
            retrieved_texts = "\n".join(
                [st.session_state.projects[selected_project]["texts"][i] for i in I[0]]
            )
            prompt = f" Only Based on the following information:\n{retrieved_texts}\nAnswer the question: {question}"
            mistral_answer = query_mistral(prompt, MISTRAL_API_KEY)
            cache[question_hash] = mistral_answer  # Cache the answer
            st.write("Answer:")
            st.write(mistral_answer)
