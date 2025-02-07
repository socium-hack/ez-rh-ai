import streamlit as st
import os
import tempfile
import re
from docx import Document  # pip install python-docx
from PyPDF2 import PdfReader  # pip install PyPDF2

# Import LLM et Prompt de langchain/ollama
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # Configuration du layout et du titre
    st.title("4️⃣ Resume Scanner - Analyse Avancée de CV")
    st.markdown("""
    <style>
    /* Petit style pour rendre le container un peu plus joli */
    .analysis-box {
        background-color: #f9f9f9; 
        padding: 1rem; 
        margin-top: 1rem; 
        border-radius: 5px; 
        border-left: 4px solid #66c2ff;
    }
    .analysis-box h2 {
        color: #333333;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    Cette page **analyse** votre CV via un modèle LLM (DeepSeek) pour vous donner :
    - **5 points marquants** (High-level bullet points).
    - Un **score global de qualité** (0 à 100).
    - Les **forces principales** de votre profil.
    - Des **suggestions** d'amélioration personnalisées.

    **Nous n'affichons pas le texte brut** pour préserver la confidentialité. 
    Seul un résumé et des conseils d'optimisation sont présentés.
    ---
    """, unsafe_allow_html=True)

    # Uploader le fichier CV
    uploaded_file = st.file_uploader("Téléversez votre CV (PDF ou DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix="."+uploaded_file.name.split(".")[-1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name

        # Lecture du contenu (anonymisé)
        with st.spinner("Lecture du CV en cours..."):
            raw_text = extract_text(file_path, uploaded_file.type)
            # Anonymiser (email, téléphone) 
            anonymized_text = anonymize_text(raw_text)

        st.success("✅ Extraction du CV réussie. (Texte anonymisé en interne)")

        # Bouton pour lancer l'analyse
        if st.button("Analyser le CV et obtenir un diagnostic"):
            with st.spinner("Analyse en cours avec DeepSeek..."):
                analysis_result = analyze_cv_with_llm(anonymized_text)

            st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
            st.markdown(analysis_result, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Veuillez téléverser votre CV au format PDF ou DOCX.")

# --------------------------------------------------------------------
# FONCTIONS UTILES : EXTRACTION, ANONYMISATION, ANALYSE LLM
# --------------------------------------------------------------------
def extract_text(file_path, file_type):
    """
    Extrait le texte d'un PDF ou d'un DOCX sans l'afficher publiquement.
    """
    if file_type == "application/pdf":
        return extract_text_from_pdf(file_path)
    else:
        return extract_text_from_docx(file_path)

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def extract_text_from_docx(docx_path):
    text = ""
    doc = Document(docx_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def anonymize_text(text):
    """
    Anonymise :
      - Email
      - Téléphone (simple regex)
    """
    # Masquer e-mails
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL MASQUÉ]", text)
    # Masquer téléphones
    text = re.sub(r"\+?\d[\d \-\.]{7,}\d", "[TEL MASQUÉ]", text)
    return text

def analyze_cv_with_llm(cv_text):
    """
    Utilise le modèle DeepSeek pour :
    - Extraire 5 points marquants
    - Donner un Quality Score (0-100)
    - Lister les forces
    - Proposer des suggestions d'amélioration

    Formate la réponse en markdown pour un affichage esthétique.
    """
    # On peut splitter si le CV est long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(cv_text)
    combined_text = "\n\n".join(chunks)

    prompt_template = """
You are an expert resume reviewer. Do not reveal your internal chain-of-thought (<think>).
Task:
1) Identify 5 bullet points that capture the main highlights or key info from this anonymized resume.
2) Provide an overall QUALITY SCORE out of 100 for resume structure, clarity, and content.
3) Summarize the candidate's strengths in 3 bullet points.
4) Suggest 3 clear improvements for this CV.

Answer in Markdown, with sections:
## Points Marquants
(5 bullet points)

## Quality Score
(Example: 85/100)

## Strengths
(3 bullet points)

## Suggestions
(3 bullet points)

Resume Text:
{resume_text}
"""

    final_prompt = prompt_template.format(resume_text=combined_text)
    prompt = ChatPromptTemplate.from_template(final_prompt)

    # Modèle
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_result = chain.invoke({})

    # Nettoyer d'éventuels <think>...
    result = remove_thinking_tags(raw_result)
    return result

def remove_thinking_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# --------------------------------------------------------------------
# LANCEMENT SI EXECUTE DIRECTEMENT
# --------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="Resume Scanner - Analyse Avancée",
        page_icon="📄",
        layout="wide"
    )
    main()
