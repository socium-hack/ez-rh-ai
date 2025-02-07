# pages/3_Generation_Questions.py

import streamlit as st
import os
import tempfile
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# --------------------------------------------------------------------
# FONCTION POUR SUPPRIMER LE CHAIN-OF-THOUGHT
# --------------------------------------------------------------------
def remove_thinking_tags(text: str) -> str:
    """
    Supprime tout le contenu encadré par <think> ... </think>
    dans la chaîne de caractères 'text'.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# --------------------------------------------------------------------
# INITIALISATION DU MODELE & EMBEDDINGS
# --------------------------------------------------------------------
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
model = OllamaLLM(model="deepseek-r1:7b")

# --------------------------------------------------------------------
# PROMPT TEMPLATES
# --------------------------------------------------------------------
# 1) Questions générales pour un poste
general_questions_template = """
You are an expert HR manager. You know how to create relevant interview questions for a given job role.
Do not reveal your chain-of-thought or anything like <think> ... </think>.

Job Role or Domain: {job_role}

Task:
Generate {num_questions} concise interview questions that an HR manager would typically ask
to evaluate both technical and soft skills for this role.
Answer:
"""

# 2) Questions basées sur le CV
cv_questions_template = """
You are an expert HR manager. You know how to create relevant interview questions specifically based on a candidate's CV.
Do not reveal your chain-of-thought or anything like <think> ... </think>.

Candidate's CV:
{resume}

Task:
Generate {num_questions} interview questions focusing on the candidate's experiences, skills, and potential gaps.
Questions should be concise and directly related to the candidate's CV details.
Answer:
"""

def generate_general_questions(job_role, num_questions=5):
    """
    Génère des questions d'entretien générales pour un poste donné.
    """
    prompt_text = general_questions_template.format(job_role=job_role, num_questions=num_questions)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

def generate_cv_questions(resume_text, num_questions=5):
    """
    Génère des questions d'entretien basées sur le contenu d'un CV.
    """
    prompt_text = cv_questions_template.format(resume=resume_text, num_questions=num_questions)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

# --------------------------------------------------------------------
# FONCTIONS UTILES POUR CV
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# PAGE STREAMLIT
# --------------------------------------------------------------------
def main():
    st.title("3️⃣ Génération de Questions d’Entretien")
    st.write("""
    Sur cette page, vous pouvez générer :
    - **Des questions d’entretien générales** en fonction d’un poste.
    - **Des questions basées sur un CV** pour aller plus en profondeur.
    """)

    # Choix du mode
    mode = st.radio(
        "Choisissez le type de questions à générer :",
        ("Questions Générales", "Questions Basées sur CV")
    )

    # Nombre de questions
    num_questions = st.number_input("Nombre de questions à générer :", min_value=1, max_value=20, value=5)

    st.markdown("---")

    if mode == "Questions Générales":
        st.subheader("Questions d’Entretien Générales")
        job_role = st.text_input("Entrez l'intitulé du poste ou le domaine (Ex: 'Développeur Python', 'Data Scientist', 'Chef de Projet'):")
        
        if job_role:
            if st.button("Générer les questions"):
                with st.spinner("Génération de questions..."):
                    result = generate_general_questions(job_role, num_questions=num_questions)
                st.success("✅ Questions générées !")
                st.write(result)

    else:  # "Questions Basées sur CV"
        st.subheader("Questions d’Entretien Basées sur un CV")
        uploaded_cv = st.file_uploader("Téléversez un CV (PDF uniquement)", type=["pdf"])
        
        if uploaded_cv:
            if st.button("Analyser le CV & Générer des questions"):
                with st.spinner("Lecture du CV..."):
                    cv_path = save_uploaded_file(uploaded_cv)
                    cv_docs = load_pdf(cv_path)
                    cv_chunks = split_text(cv_docs)
                    # On combine tous les chunks en une seule string
                    resume_text = "\n".join([doc.page_content for doc in cv_chunks])
                
                # Génération des questions
                with st.spinner("Génération des questions basées sur le CV..."):
                    result = generate_cv_questions(resume_text, num_questions=num_questions)
                
                st.success("✅ Questions générées avec succès !")
                st.write(result)
        else:
            st.info("Veuillez téléverser un CV au format PDF pour générer des questions spécifiques au candidat.")

if __name__ == "__main__":
    main()
