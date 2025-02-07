import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time

# Configuration du modèle
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
model = OllamaLLM(model="deepseek-r1:7b")

# Template de génération de questions
question_template = """
You are a professional recruiter. Based on the resume provided, generate 5 interview questions 
tailored to the candidate's skills and experience.

Resume:
{resume}


Questions:
"""

# Fonction pour charger un PDF
def load_pdf(file):
    loader = PDFPlumberLoader(file)
    documents = loader.load()
    return documents

# Fonction pour extraire le texte du CV
def extract_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Fonction de génération de questions
def generate_questions(resume_text):
    prompt = ChatPromptTemplate.from_template(question_template)
    chain = prompt | model
    return chain.invoke({"resume": resume_text})

# Interface Streamlit
st.set_page_config(page_title="🎤 Générateur de Questions d'Entretien", layout="wide")

# Titre principal
st.markdown(
    "<h1 style='text-align: center; color: #2C3E50;'>🎤 Générateur de Questions d'Entretien</h1>",
    unsafe_allow_html=True,
)

# Upload du CV
st.markdown("### 📄 Upload ton CV (PDF)")
uploaded_resume = st.file_uploader("Sélectionne un fichier PDF", type="pdf")

# Bouton de traitement
if uploaded_resume:
    with st.spinner("📜 Lecture et analyse du CV..."):
        resume_docs = load_pdf(uploaded_resume)
        resume_text = "\n".join([doc.page_content for doc in extract_text(resume_docs)])

    if st.button("🚀 Générer des Questions"):
        with st.spinner("🔍 Analyse du CV et génération des questions..."):
            time.sleep(2)
            result = generate_questions(resume_text)

        # Affichage des résultats
        st.subheader("🎯 Questions Générées")
        st.write(result)

        # Style des résultats
        st.markdown(
            """
            <style>
            .stMarkdown {font-size: 18px; color: #34495E; font-weight: bold;}
            </style>
            """,
            unsafe_allow_html=True,
        )
else:
    st.warning("📢 Merci d'upload un CV pour commencer l'analyse.")
