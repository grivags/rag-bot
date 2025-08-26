import os

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

# Init retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="chroma_db",
    collection_name="docs",
    embedding_function=embeddings,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# Init LLM
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.2,
)

# Prompt for RAG
system_prompt = """Kamu adalah asisten yang membantu menjawab pertanyaan berdasarkan konten dokumen. Jika jawaban tidak ditemukan di konteks, jawab singkat: "Aku tidak menemukan jawaban dari pertanyaan tersebut di dokumen." Jawab ringkas, jelas dan sertakan poin sumber jika relevan.
"""

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("human", "Pertanyaan: {question}\n\nKonteks:\n{context}")
])

def format_docs(docs):
    parts = []
    for i, d in enumerate (docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        parts.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(parts)

# Chain RAG
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# --- FastAPI ---
app = FastAPI(title="RAG Bot")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # boleh dibatesin ke ["http://localhost:5500"] kalo mau
    allow_credentials=False,      # True juga boleh, tapi kalo True jangan pakai "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # ambil dokumen untuk sumber
    docs = retriever.get_relevant_documents(req.question)
    answer = rag_chain.invoke(req.question).content

    # kumpulin sumber minimal (path + potongan pendek)
    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source", "unknown"),
            "preview": d.page_content[:160] + ("..." if len(d.page_content) > 160 else "")
        })

    return AskResponse(answer=answer, sources=sources)

@app.get("/")
def root():
    return {"ok": True, "message": "RAG Bot up. POST /ask {question}"}