from fastapi import FastAPI, File, UploadFile, Form,Body
from fastapi.responses import JSONResponse
from API.sama_updated import read_docx, process_pdf, convert_to_markdown, extract_text_from_pptx
from API.searchmethods import qdrant_search, ReciprocalRankFusion, bm25s_search
from API.open import process_file_with_gpt_vision
from API.florence import process_file_with_florence
from API.googlevision import process_file_with_google_vision
from API.caludeocr import process_file_with_claude
from API.getllm import get_llm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
import bm25s
# import Stemmer
import os
from uuid import UUID
from typing import List, Dict

if not os.path.exists("KnowledgeBase"):
    os.makedirs("KnowledgeBase")

app = FastAPI()

conversation_histories: Dict[UUID, List[Dict[str, str]]] = {}

def get_or_create_conversation(session_uuid: UUID):
    if session_uuid not in conversation_histories:
        conversation_histories[session_uuid] = []
    return conversation_histories[session_uuid]

@app.post("/extract_text/")
async def extract_text_api(
    file: UploadFile = File(...),
    ocr_method: str = Form("tesseract"),
    session_id:str = Form("Generic"),
    search_method:str = Form("Embedding + Qdrant")
):
    uploads_folder = f"uploads/{session_id}"
    os.makedirs(uploads_folder, exist_ok=True)
    file_path = os.path.join(uploads_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if ocr_method == "tesseract":
        if file.filename.endswith(".pdf"):
            content = process_pdf(file_path)
        elif file.filename.endswith(".docx"):
            content = read_docx(file_path)
        elif file.filename.endswith(".pptx"):
            content = extract_text_from_pptx(file_path)
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported file format."})
        all_text = ""
        for item in content:
            all_text += convert_to_markdown(item)

    if ocr_method == "openai":
        all_text = process_file_with_gpt_vision(file_path, uploads_folder, verbose=True)

    elif ocr_method == "florence":
        all_text = process_file_with_florence(file_path, uploads_folder, verbose=True)
        all_text = all_text[0].get("<OCR>")

    elif ocr_method == "google":
        all_text = process_file_with_google_vision(file_path, uploads_folder, verbose=True)

    elif ocr_method == "claude":
        all_text = process_file_with_claude(file_path, uploads_folder, verbose=True)
        all_text = all_text[0].get("text")

    return JSONResponse(all_text)


import pickle

class KnowledgeBase:
    def __init__(self):
        self.bm25_retriever = None
        self.bm25_corpus = None
        self.bm25_stemmer = None
        self.qdrant = None
        self.session_id = None



@app.post("/create_kb/")    
def create_knowledge_base(session_id:str = Form("Generic"),
                          text: str = Form(""),
                          search_method:str = Form("Embedding + Qdrant")):
    if os.path.exists(f"KnowledgeBase/kb_{session_id}.pkl"):
        return JSONResponse(status_code=200, content={"message": "Knowledge base already exists."})
    
    if text:
        doc_texts = load_from_string(text)
        knowledge_base = KnowledgeBase()
        if search_method in ["Embedding + Qdrant", "RRF"]:
            knowledge_base.qdrant = process_documents_with_qdrant(doc_texts)

        if search_method in ["BM25S", "RRF"]:
            knowledge_base.bm25_retriever, knowledge_base.bm25_corpus, knowledge_base.bm25_stemmer = init_bm25s_retriever(doc_texts)

        file_path = os.path.join("KnowledgeBase", f"kb_{session_id}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(knowledge_base, f)
        return JSONResponse("sucessfully create the Knowledge Base")
    else:
        raise ValueError("Error! Text value is Empty")
        

def load_from_string(text: str):
    document = Document(page_content=text, metadata={"source": "string_input"})
    return [document]

def process_documents_with_qdrant(docs, model_name="./paraphrase-multilingual-MiniLM-L12-v2"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    qdrant = Qdrant.from_documents(split_docs, embedding_model, location=":memory:", collection_name="my_documents")
    return qdrant

def init_bm25s_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ".", ""])
    split_docs = text_splitter.split_documents(docs)
    corpus = [{'id': i, 'metadata': doc.metadata, 'text': doc.page_content} for i, doc in enumerate(split_docs)]
    stemmer = Stemmer.Stemmer("english")
    texts = [doc['text'] for doc in corpus]
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever, corpus, stemmer


from fastapi import Body



if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8001)
