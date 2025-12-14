# build_rag_faiss.py
# OFFICIAL FAISS RAG BUILDER — FASTER & LIGHTER THAN CHROMA
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
import shutil

# CONFIG
PDF_FOLDER = Path("medical_docs")
DB_FOLDER = Path("faiss_medical_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

print("="*80)
print("   FAISS MEDICAL RAG BUILDER — ULTRA FAST & LOCAL")
print("="*80)

pdf_files = list(PDF_FOLDER.glob("*.pdf"))
if not pdf_files:
    print("No PDFs in medical_docs/ — add your medical literature and run again!")
    exit()

print(f"Found {len(pdf_files)} PDFs:")
for f in pdf_files: print(f"   • {f.name}")

# Load & chunk
print(f"\nChunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
all_docs = []

for pdf_path in pdf_files:
    print(f"   Loading {pdf_path.name}...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = pdf_path.name
    all_docs.extend(chunks)
    print(f"      → {len(chunks)} chunks")

print(f"\nTotal: {len(all_docs)} chunks")

# Build FAISS index
print(f"\nBuilding FAISS index with nomic-embed-text...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Remove old DB
if DB_FOLDER.exists():
    shutil.rmtree(DB_FOLDER)

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(all_docs, embeddings)

# Save FAISS index + metadata
vectorstore.save_local(str(DB_FOLDER))
print(f"\nFAISS INDEX BUILT & SAVED!")
print(f"   {len(all_docs)} chunks → {DB_FOLDER.resolve()}")
print(f"   Run your main script now — RAG will be 3x faster!")
print("="*80)