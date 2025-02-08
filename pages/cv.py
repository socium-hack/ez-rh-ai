import streamlit as st
import re
import tempfile
from docx import Document
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Pour DeepSeek (LLM via Ollama)
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

# -----------------------------------------------------------------------------
# Fonctions d'extraction de texte
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un PDF à l'aide de PyPDF2."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def extract_text_from_pdf_ocr(upload) -> str:
    """
    Utilise pdf2image pour convertir un PDF en images puis effectue l'OCR avec Tesseract.
    Retourne le texte extrait en français.
    """
    # Assurez-vous que le curseur est remis à zéro
    upload.file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload.file.read())
        pdf_path = tmp_file.name
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        st.error(f"Erreur lors de la conversion PDF -> Image: {e}")
        return ""
    ocr_text = ""
    for img in images:
        ocr_text += pytesseract.image_to_string(img, lang="fra") + "\n"
    return ocr_text

def extract_text_from_docx(docx_path: str) -> str:
    """Extrait le texte d'un document DOCX."""
    text = ""
    doc = Document(docx_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(txt_path: str) -> str:
    """Extrait le texte d'un fichier TXT."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_image(upload) -> str:
    """Utilise Tesseract pour extraire le texte d'une image."""
    upload.file.seek(0)
    try:
        image = Image.open(upload.file)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de l'image : {e}")
        return ""
    return pytesseract.image_to_string(image, lang="fra")

def extract_text_from_file(upload) -> str:
    """
    Extrait le texte d'un fichier (PDF, DOCX, TXT, PNG, JPG, JPEG) en se basant sur l'extension.
    Si c'est un PDF, il essaie d'abord PyPDF2 puis l'OCR si le texte extrait est insuffisant.
    """
    suffix = upload.filename.split(".")[-1].lower()
    upload.file.seek(0)
    if suffix in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(upload)
    
    # Pour les PDF, essayer d'abord PyPDF2
    if suffix == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp_file:
            tmp_file.write(upload.file.read())
            file_path = tmp_file.name
        text = extract_text_from_pdf(file_path)
        if len(text.strip()) < 100:  # Si le texte est trop court, utiliser l'OCR
            st.info("Texte insuffisant via extraction classique, utilisation de l'OCR...")
            upload.file.seek(0)
            text = extract_text_from_pdf_ocr(upload)
        return text
    
    # Pour DOCX
    if suffix in ["docx", "doc"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp_file:
            tmp_file.write(upload.file.read())
            file_path = tmp_file.name
        return extract_text_from_docx(file_path)
    
    # Pour TXT
    if suffix == "txt":
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp_file:
            tmp_file.write(upload.file.read())
            file_path = tmp_file.name
        return extract_text_from_txt(file_path)
    
    st.error("Type de fichier non supporté.")
    return ""

# -----------------------------------------------------------------------------
# Fonction pour structurer le CV via DeepSeek (sans extraire les informations sensibles)
# -----------------------------------------------------------------------------
def structure_cv(cv_text: str) -> str:
    """
    Utilise DeepSeek pour extraire et structurer les informations clés du CV,
    en **NE PAS** extraire le nom, l'email, l'adresse ou autres données sensibles.
    Le modèle doit répondre en français et uniquement retourner les sections suivantes :
    - formation
    - experience
    - competences
    - langues
    Réponds uniquement en JSON.
    """
    prompt_template = """
Tu es un expert en analyse de CV et tu dois extraire et structurer uniquement les informations professionnelles non sensibles.
NE PAS EXTRAIRE les informations personnelles telles que le nom, l'email, l'adresse, le numéro de téléphone, etc.
Extrait uniquement :
- La formation (diplômes, établissements, années)
- L'expérience professionnelle (postes occupés, entreprises, durées)
- Les compétences (techniques et soft skills)
- Les langues maîtrisées

Voici le texte du CV (anonymisé) :
{cv_text}

Réponds uniquement en JSON, par exemple :
{{
  "formation": "...",
  "experience": "...",
  "competences": "...",
  "langues": "..."
}}
"""
    prompt_text = prompt_template.format(cv_text=cv_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    structured_output = chain.invoke({})
    return re.sub(r"<think>.*?</think>", "", structured_output, flags=re.DOTALL).strip()

# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Extraction & Structuration des CV (Anonymat Garanti)",
    page_icon="📄",
    layout="wide"
)
st.title("Extraction & Structuration des Informations Clés des CV")
st.markdown("""
Cette application permet d'extraire automatiquement le texte d'un CV, d'appliquer l'OCR si nécessaire, et de structurer les informations professionnelles clés en JSON.<br>
Les données personnelles (nom, email, adresse, téléphone) ne sont PAS extraites et restent anonymes.<br>
La réponse est générée en français par DeepSeek.
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Uploader un CV (PDF, DOCX, TXT, PNG, JPG...)", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Extraction du texte..."):
        extracted_text = extract_text_from_file(uploaded_file)
    st.markdown("### Texte extrait (anonymisé) :")
    st.text_area("Texte extrait", value=extracted_text, height=200)

    with st.spinner("Structuration des informations..."):
        structured_cv = structure_cv(extracted_text)
    st.markdown("### CV Structuré (JSON) :")
    st.code(structured_cv, language="json")
