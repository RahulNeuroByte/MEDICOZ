import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Folder jahan PDF files rakhi hain
PDF_FOLDER_PATH = "data/"  # 👈 Update this to match your actual folder
DB_FAISS_PATH = "vectorstore/db_faiss"

# Function to load all PDFs from folder
def load_documents(pdf_folder_path):
    documents = []
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, filename)
            print(f"📄 Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents

# Function to split documents into smaller chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print("✂️ Splitting documents into chunks...")
    return splitter.split_documents(documents)

# Function to create vectorstore using HuggingFace embeddings
def create_and_save_vectorstore(text_chunks, save_path):
    print("📐 Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📦 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(text_chunks, embedding_model)

    print(f"💾 Saving FAISS vectorstore to: {save_path}")
    vectorstore.save_local(save_path)

# Main controller function
def main():
    print("🚀 Starting document ingestion...")

    documents = load_documents(PDF_FOLDER_PATH)
    if not documents:
        print("❌ No PDF files found in the folder!")
        return

    text_chunks = split_documents(documents)
    create_and_save_vectorstore(text_chunks, DB_FAISS_PATH)

    print("✅ Ingestion completed successfully!")

# Run main if file is executed directly
if __name__ == "__main__":
    main()
