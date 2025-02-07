import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ---- Configuration de la page Streamlit ----
st.set_page_config(page_title="CV vs Job Match", layout="wide", page_icon="📄")

# ---- Modèles d'embedding et LLM ----
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:7b")

# ---- Prompt de comparaison ----
comparison_template = """
You are an expert in resume analysis. Compare the provided resume with the job description.
Evaluate the match based on skills, experience, and keywords.

Resume:
{resume}

Job Description:
{job_description}

Provide a compatibility score (out of 100) and highlight strengths and weaknesses.
Answer:
"""

# ---- Fonction pour sauvegarder les fichiers uploadés ----
def save_uploaded_file(uploaded_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Créer un dossier temporaire s'il n'existe pas
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Sauvegarde le fichier
    
    return temp_path  # Retourne le chemin du fichier

# ---- Fonction pour charger un PDF ----
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# ---- Fonction pour extraire le texte ----
def extract_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# ---- Fonction d'analyse de compatibilité ----
def analyze_match(resume_text, job_text):
    prompt = ChatPromptTemplate.from_template(comparison_template)
    chain = prompt | model
    return chain.invoke({"resume": resume_text, "job_description": job_text})

# ---- Interface Utilisateur Streamlit ----
st.title("🔍 Analyse de compatibilité entre CV et Offre d'emploi")

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("📄 Upload ton CV (PDF)", type=["pdf"])
with col2:
    uploaded_job = st.file_uploader("📑 Upload l'offre d'emploi (PDF ou TXT)", type=["pdf", "txt"])

if uploaded_resume and uploaded_job:
    st.subheader("📊 Résultats de l'analyse")

    # Sauvegarde des fichiers uploadés
    resume_path = save_uploaded_file(uploaded_resume)
    job_path = save_uploaded_file(uploaded_job) if uploaded_job.type == "application/pdf" else None

    # Chargement et extraction des textes
    resume_docs = load_pdf(resume_path)
    resume_text = "\n".join([doc.page_content for doc in extract_text(resume_docs)])

    if job_path:
        job_docs = load_pdf(job_path)
        job_text = "\n".join([doc.page_content for doc in extract_text(job_docs)])
    else:
        job_text = uploaded_job.getvalue().decode("utf-8")

    # Analyse et affichage des résultats
    with st.spinner("📊 Analyse en cours..."):
        result = analyze_match(resume_text, job_text)

    st.success("✅ Analyse terminée !")
    st.write(result)

