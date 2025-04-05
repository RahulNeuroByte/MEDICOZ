


import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Path to FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vectorstore
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# ✅ Corrected: Load HuggingFace LLM with task
def load_llm(repo_id, token):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        max_length=512,
        huggingfacehub_api_token=token,
        task="text-generation"  # ✅ FIXED: This line prevents the 'unknown task' error
    )

# Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Streamlit App
def main():
    st.set_page_config(page_title="MediBot - Your Medical Assistant 💊", page_icon="🧠")
    st.title("🩺 MediBot")
    st.write("Hey! I am MediBot... how can I help you?")

    # Session message setup
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    # User input
    user_prompt = st.chat_input("Ask anything about a disease, symptoms, or treatment...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            # Load LLM and Vector DB
            HF_TOKEN = os.environ.get("HF_TOKEN")
            vectorstore = get_vectorstore()
            llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3", HF_TOKEN)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Get response
            response = qa_chain.invoke({"query": user_prompt})
            answer = response["result"]
            sources = response["source_documents"]

            # Format source info
            source_texts = "\n\n".join([f"- {doc.metadata.get('source', 'Unknown Source')}" for doc in sources])
            final_response = f"**🧠 Answer:**\n{answer}\n\n**📚 Source Documents:**\n{source_texts}"

            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()
