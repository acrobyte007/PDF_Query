##return embedding vector for a given text
##uses senetence based emebdings
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)   


vector_store = Chroma(
collection_name="collection",
embedding_function=hf,
persist_directory="chroma_langchain_db",
)


def set_embedding(text:str,doc_id:str,user_id:str):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=300, chunk_overlap=40
)
    texts = text_splitter.split_text(text)
    print(type(texts[0]))## IT IS LIST OF STRINGS 
    
    for i in range(len(texts)):
        ##vector=hf.embed_query(texts[i])
        vector_id=(user_id+doc_id+str(i))
        globals()[f"document_{i}"]=Document(
            page_content= texts[i],
            metadata={"doc_id": doc_id, "user_id": user_id},
            id= vector_id,
        )
        vector_store.add_documents([globals()[f"document_{i}"]])
        print(f"Added document {i} with id {vector_id}")

def get_chunks(query:str,user_id:str,doc_id:str):
    results = vector_store.similarity_search(
    query,
    k=5,
    filter={"user_id": user_id}
)
    list_of_chunks=[]
    for res in results:
        list_of_chunks.append(res.page_content)
    return list_of_chunks