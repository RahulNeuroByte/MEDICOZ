

import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

# Load environment variable
from dotenv import load_dotenv
load_dotenv()

# Step 1: Hugging Face Token and Model ID
HF_TOKEN = os.getenv("HF_TOKEN")

# ✅ RECOMMENDED: Use a model that supports inference API
HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"  # or try "google/flan-t5-large"

def load_llm(model_id: str):
    return HuggingFaceHub(
        repo_id=model_id,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )

# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template: str):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Build RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Input Medical Query
try:
    user_query = input("❓ Ask your medical question: ")
    response = qa_chain.invoke({'query': user_query})

    # Step 6: Display Answer and Sources
    print("\n📋 Answer:\n", response["result"])
    print("\n📚 Source Documents:\n")
    for i, doc in enumerate(response["source_documents"], start=1):
        print(f"Source {i}:\n{doc.page_content}\n")

except Exception as e:
    print("🚨 Error occurred while processing your request:")
    print(str(e))
