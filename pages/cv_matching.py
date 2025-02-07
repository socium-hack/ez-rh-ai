# # pages/1_CV_Offre_Matching.py

# import streamlit as st
# import os
# import tempfile
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama.llms import OllamaLLM

# import re

# # ---------------------------------------------
# # FONCTION POUR SUPPRIMER LE CHAIN-OF-THOUGHT
# # ---------------------------------------------
# def remove_thinking_tags(text: str) -> str:
#     return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# # ---------------------------------------------
# # INITIALISATION DU MODELE & EMBEDDINGS
# # ---------------------------------------------
# embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
# model = OllamaLLM(model="deepseek-r1:7b")

# # ---------------------------------------------
# # PROMPT TEMPLATE
# # ---------------------------------------------
# comparison_template = """
# You are an expert in resume analysis. Compare the provided resume with the job description.
# Evaluate the match based on skills, experience, and keywords.
# Do not include or reveal your internal reasoning or any <think> section.

# Resume:
# {resume}

# Job Description:
# {job_description}

# Provide a compatibility score (out of 100) and highlight strengths and weaknesses.
# Answer:
# """

# def analyze_match(resume_text, job_text):
#     prompt = ChatPromptTemplate.from_template(comparison_template)
#     chain = prompt | model
#     return chain.invoke({"resume": resume_text, "job_description": job_text})

# # ---------------------------------------------
# # FONCTIONS UTILES
# # ---------------------------------------------
# def save_uploaded_file(uploaded_file):
#     temp_dir = "temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     temp_path = os.path.join(temp_dir, uploaded_file.name)
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return temp_path

# def load_pdf(file_path):
#     loader = PDFPlumberLoader(file_path)
#     return loader.load()

# def split_text(documents, chunk_size=1000, chunk_overlap=200):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         add_start_index=True
#     )
#     return text_splitter.split_documents(documents)

# # ---------------------------------------------
# # PAGE STREAMLIT
# # ---------------------------------------------
# def main():
#     st.title("1️⃣ Matching CV et Offre d'emploi")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         uploaded_resume = st.file_uploader("📄 Upload ton CV (PDF uniquement)", type=["pdf"])
#     with col2:
#         uploaded_job = st.file_uploader("📑 Upload l'offre d'emploi (PDF ou TXT)", type=["pdf", "txt"])

#     if uploaded_resume and uploaded_job:
#         st.subheader("📊 Résultats de l'analyse")

#         # CV
#         resume_path = save_uploaded_file(uploaded_resume)
#         with st.spinner("📝 Lecture du CV..."):
#             resume_docs = load_pdf(resume_path)
#             resume_chunks = split_text(resume_docs)
#             resume_text = "\n".join([doc.page_content for doc in resume_chunks])

#         # Offre
#         job_text = ""
#         if uploaded_job.type == "application/pdf":
#             job_path = save_uploaded_file(uploaded_job)
#             with st.spinner("📝 Lecture de l'offre..."):
#                 job_docs = load_pdf(job_path)
#                 job_chunks = split_text(job_docs)
#                 job_text = "\n".join([doc.page_content for doc in job_chunks])
#         else:
#             job_text = uploaded_job.getvalue().decode("utf-8")

#         # Analyse
#         with st.spinner("📊 Analyse en cours..."):
#             raw_result = analyze_match(resume_text, job_text)
#             result = remove_thinking_tags(raw_result)

#         st.success("✅ Analyse terminée !")
#         st.write(result)
#     else:
#         st.info("Veuillez uploader un CV et une offre d'emploi pour commencer l'analyse.")

# if __name__ == "__main__":
#     main()
import streamlit as st
import re
import os
import tempfile
from docx import Document
from PyPDF2 import PdfReader

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json

def main():
    st.set_page_config(layout="wide")
    
    st.title("3️⃣ Matching Multiple CV - Offre d’Emploi (UI Améliorée)")

    # -- CSS Pour le Design --
    st.markdown("""
    <style>
    /* Conteneur global */
    .offer-container {
        background-color: #fafafa;
        border: 2px solid #e5e5e5;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 5px;
    }
    .section-title {
        color: #2b7cff;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .candidate-card {
        background: #f7f7f9; 
        border-left: 6px solid #2b7cff;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .candidate-header {
        font-weight: bold;
        font-size: 1rem;
        color: #333;
    }
    .candidate-score {
        font-size: 0.9rem;
        color: #555;
        margin-top: 3px;
    }
    .reasons-list {
        margin-left: 1.2rem;
        margin-top: 5px;
    }
    .ranking-label {
        background-color: #2b7cff;
        color: white;
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    Sur cette page, vous pouvez :
    1. Définir ou Uploader une **offre d’emploi**.
    2. Uploader **plusieurs CV**.
    3. Obtenir pour chaque CV :
       - Un **score** (0 à 100) de compatibilité.
       - Des **raisons** (forces/faiblesses) en quelques points.
    4. **Classer** les CV automatiquement du plus adapté au moins adapté.

    L’anonymat est préservé : pas d'affichage du contenu brut du CV.
    repond toujours en francais 
    ---
    """)

    # -- Saisie / Upload Offre --
    with st.expander("1) Saisir / Uploader l’Offre d’Emploi", expanded=True):
        offer_text = get_offer_text()

    if offer_text:
        st.success("Offre d’emploi définie. Vous pouvez maintenant uploader plusieurs CV.")
    else:
        st.warning("Veuillez définir l’offre d’emploi avant de poursuivre.")

    # -- Multi-file uploader CV --
    with st.expander("2) Uploader plusieurs CV (PDF/DOCX)", expanded=True):
        uploaded_cvs = st.file_uploader(
            "Choisir un ou plusieurs fichiers",
            type=["pdf","docx"],
            accept_multiple_files=True
        )

    # -- Bouton Matching --
    if offer_text and uploaded_cvs:
        st.markdown("### 3) Lancer le Matching")
        if st.button("Analyser et Classer les CV"):
            with st.spinner("Analyse et matching en cours..."):
                results = []
                for idx, cv_file in enumerate(uploaded_cvs, start=1):
                    cv_text = load_and_extract_text(cv_file)
                    anonymized_cv_text = anonymize_text(cv_text)

                    raw_result = analyze_match(anonymized_cv_text, offer_text)
                    score, reasons = parse_matching_result(raw_result)

                    if score is None:
                        score = 0
                        reasons = ["Impossible d'extraire le score."]
                    candidate_name = f"Candidat #{idx}"
                    results.append({
                        "candidate": candidate_name,
                        "score": score,
                        "reasons": reasons
                    })

                # Trie par score
                results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

            # -- Affichage final sans table --
            st.success("Matching terminé !")
            st.write("Classement des CV par ordre décroissant de compatibilité :")

            for i, r in enumerate(results_sorted, start=1):
                # Container par candidat
                st.markdown(f"<div class='candidate-card'>", unsafe_allow_html=True)
                st.markdown(f"<span class='ranking-label'>#{i}</span> <span class='candidate-header'>{r['candidate']}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='candidate-score'><b>Score :</b> {r['score']}/100</div>", unsafe_allow_html=True)

                # Liste des raisons
                if r["reasons"]:
                    st.markdown("<ul class='reasons-list'>", unsafe_allow_html=True)
                    for reason in r["reasons"]:
                        st.markdown(f"<li>{reason}</li>", unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------
#  FONCTIONS - OFFRE
# --------------------------------------------------------------------
def get_offer_text():
    """
    Saisie ou upload d'une offre d’emploi.
    Retourne une chaîne de caractères (texte).
    """
   
    uploaded_offer = st.file_uploader("Uploader un document (PDF/TXT) décrivant l’offre", type=["pdf","txt"])
    final_offer_text = ""

    if st.button("Enregistrer l’Offre"):
        if uploaded_offer:
            if uploaded_offer.type == "application/pdf":
                # Dans un vrai cas, on lirait le PDF
                final_offer_text = f"(Offre PDF) {uploaded_offer.name}" 
                st.success(f"Offre définie à partir du PDF : {uploaded_offer.name}")
            else:
                text_data = uploaded_offer.read().decode("utf-8")
                final_offer_text = text_data
                st.success("Offre définie à partir du fichier texte uploadé.")
    
        st.session_state["offer_text"] = final_offer_text

    if "offer_text" in st.session_state and st.session_state["offer_text"]:
        return st.session_state["offer_text"]
    else:
        return ""

# --------------------------------------------------------------------
#  FONCTIONS - CV
# --------------------------------------------------------------------
def load_and_extract_text(cv_file):
    suffix = cv_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp_file:
        tmp_file.write(cv_file.getbuffer())
        file_path = tmp_file.name

    if cv_file.type == "application/pdf":
        return extract_text_pdf(file_path)
    else:
        return extract_text_docx(file_path)

def extract_text_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        txt = page.extract_text() or ""
        text += txt + "\n"
    return text

def extract_text_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def anonymize_text(text):
    # Emails
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL MASQUÉ]", text)
    # Téléphone
    text = re.sub(r"\+?\d[\d \-\.]{7,}\d", "[TEL MASQUÉ]", text)
    return text

# --------------------------------------------------------------------
# ANALYSE MATCHING (LLM)
# --------------------------------------------------------------------
def analyze_match(cv_text, job_text):
    template = """
You are an expert in resume-job matching. 
Do not reveal chain-of-thought (<think>).

Candidate's Resume (anonymized):
{resume}

Job Description:
{job_description}

Requirements:
1) Provide a "score" from 0 to 100 for how well this CV matches the job.
2) Provide 2-3 short bullet points about the reasons (strengths/weaknesses).
3) Format your answer as follows:

SCORE: XX
REASONS:
- reason 1
- reason 2

Answer:
"""
    prompt_text = template.format(resume=cv_text, job_description=job_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

def remove_thinking_tags(txt):
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

def parse_matching_result(result_text):
    score_match = re.search(r"SCORE:\s*(\d+)", result_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = None

    reasons_section = []
    reasons_match = re.search(r"REASONS:(.*)", result_text, re.IGNORECASE | re.DOTALL)
    if reasons_match:
        reasons_block = reasons_match.group(1).strip()
        lines = reasons_block.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                reasons_section.append(line[2:].strip())
            elif line:
                reasons_section.append(line)
    return score, reasons_section

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
