# pages/2_QA_RAG.py

import streamlit as st
import os
import tempfile
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# ---------------------------------------------
# FONCTION POUR SUPPRIMER LE CHAIN-OF-THOUGHT
# ---------------------------------------------
def remove_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ---------------------------------------------
# INITIALISATION DU MODELE & EMBEDDINGS
# ---------------------------------------------
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
model = OllamaLLM(model="deepseek-r1:7b")

# ---------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------
qa_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
GIVE ALL THE IMPORTANT DETAILS AND keep the answer concise.
REPONDS TOUJOURS EN FRANCAIS SAUF SI LA QUESTION EST POSEE EN ANGLAIS 
Do not include or reveal your internal reasoning or any <think> section.

Question: {question}
Context: {context}
Answer:
"""

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(qa_template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def retrieve_docs(query, vector_store, k=3):
    return vector_store.similarity_search(query, k=k)

# ---------------------------------------------
# FONCTIONS UTILES
# ---------------------------------------------
def save_uploaded_file(uploaded_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# ---------------------------------------------
# PAGE STREAMLIT
# ---------------------------------------------
def main():
    st.title("2️⃣ Q&A sur PDF (RAG)")

    rag_pdf = st.file_uploader("Upload un PDF pour indexation (RAG)", type=["pdf"])
    
    if rag_pdf:
        with st.spinner("🔄 Indexation du PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(rag_pdf.getbuffer())
                temp_pdf_path = temp_file.name
            
            # Création d'un vector store local
            local_vector_store = InMemoryVectorStore(embeddings)
            
            documents = load_pdf(temp_pdf_path)
            chunked_docs = split_text(documents)
            local_vector_store.add_documents(chunked_docs)
        
        st.success(f"✅ PDF {rag_pdf.name} indexé avec succès !")

        question = st.text_input("Pose ta question au document :")
        if question:
            st.write("**Question** :", question)
            with st.spinner("💬 Recherche..."):
                related_docs = retrieve_docs(question, local_vector_store, k=3)
                raw_answer = answer_question(question, related_docs)
                answer = remove_thinking_tags(raw_answer)
            
            st.write("**Réponse** :")
            st.write(answer)
    else:
        st.info("Upload un PDF pour activer la fonctionnalité Q&A.")

if __name__ == "__main__":
    main()
