# Inget Script (Load -> Split -> Embed -> Save to Chroma).

# Import libraries.
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load .env.
load_dotenv()

# Setting up directories.
DOCS_DIR = Path("docs")
DB_DIR = Path("chroma_db")
DB_DIR.mkdir(exist_ok=True)

def load_documents():
    docs = [] # Declare empty list.
    
    for p in DOCS_DIR.rglob("*"):
        if p.suffix.lower() in [".pdf"]:
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in [".txt", ".md"]:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    
    if not docs:
        raise RuntimeError("No file found in directory /docs. Place your desired file in docs folder first.")
    return docs

def main(): # ETL Pipeline: Extract -> Transform -> Load
    # Load documents function.
    print("Loading documents...")
    docs = load_documents()
    
    # Chunking with separators listed, with chunk size of 800 and overlaps of 120 each context, to prevent chunks cut-offs.
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=800, chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)
    
    # Embeddings with MiniLM model (Each chunk is transformed to number vectors (dense embeddings)).
    # Embeddings = semantics text representation -> can be used to search similarity later on.
    print("Preparing embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Build vector DB
    print("Building / updating Chroma vector store")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="docs",        
    )
    
    print("Finished. Database stored at: ", DB_DIR.resolve())

if __name__ == "__main__":
    main()
