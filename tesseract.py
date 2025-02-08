import streamlit as st
import os
import pytesseract
import re
import tempfile
import pdf2image
from PIL import Image

# ----- CONFIGURATIONS TESSERACT (pour Windows) -----
# Adaptez ces chemins selon votre installation de Tesseract :
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
# ---------------------------------------------------

# DeepSeek (via Ollama)
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------
# Configuration Streamlit
# ------------------------------------------------------------
st.set_page_config(
    page_title="OCR Choix (Image/PDF) + Summarize / Q&A (DeepSeek)",
    layout="wide"
)

st.title("Choix Image ou PDF scanné → OCR (Tesseract) → Résumé / Q&A (DeepSeek)")
st.markdown("""
Cette page permet :

1. De **choisir le type de fichier** : Image (PNG/JPG/JPEG) ou PDF scanné.
2. D'effectuer l'**OCR** via Tesseract (en français).
3. De faire soit un **Résumé** (Summarize) soit une **Q&A** (question/réponse) sur le texte extrait, via le modèle **DeepSeek** (OllamaLLM), toujours en français.
""")

# ------------------------------------------------------------
# Fonctions
# ------------------------------------------------------------
def remove_thinking_tags(txt: str) -> str:
    """Retire les balises <think> si présentes."""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

def extract_text_from_image(uploaded_file) -> str:
    """
    OCR sur une image (PNG, JPG, JPEG) à l'aide de Tesseract (lang=fra).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        img_path = tmp_file.name
    try:
        image = Image.open(img_path)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de l'image : {e}")
        return ""
    text = pytesseract.image_to_string(image, lang="fra")
    return text

def extract_text_from_pdf_scanned(uploaded_file) -> str:
    """
    Convertit un PDF scanné en images (pdf2image), puis applique Tesseract (lang=fra).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        pdf_path = tmp_file.name

    try:
        images = pdf2image.convert_from_path(pdf_path)
    except Exception as e:
        st.error(f"Erreur conversion PDF->Image : {e}")
        return ""

    all_text = ""
    for idx, img in enumerate(images):
        text_part = pytesseract.image_to_string(img, lang="fra")
        all_text += text_part + "\n"
    return all_text

def summarize_text(extracted_text: str) -> str:
    """
    Utilise DeepSeek pour résumer le texte en français.
    """
    summarize_template = """
You are an expert summarizer. The following text is in French or partially in another language.
You MUST respond in French. Summarize the text in one paragraph, focusing on the main ideas.

Text:
{extracted_text}

Answer in French:
"""
    prompt_text = summarize_template.format(extracted_text=extracted_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

def answer_question(extracted_text: str, question: str) -> str:
    """
    Q&A en français à partir du texte extrait.
    """
    qa_template = """
You are an assistant that must use the provided context (extracted text) to answer the user's question.
If the text does not contain the answer, say "Je ne sais pas."
You MUST answer in French.

Context:
{extracted_text}

Question (French):
{question}

Answer in French:
"""
    prompt_text = qa_template.format(extracted_text=extracted_text, question=question)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

# ------------------------------------------------------------
# Interface
# ------------------------------------------------------------
st.markdown("### 1) Choisissez le type de fichier (Image ou PDF scanné)")
file_type = st.radio(
    "Type de fichier à uploader :",
    ["Image (PNG/JPG/JPEG)", "PDF scanné"]
)

if file_type == "Image (PNG/JPG/JPEG)":
    uploaded_file = st.file_uploader("Uploader une image", type=["png", "jpg", "jpeg"])
else:
    uploaded_file = st.file_uploader("Uploader un PDF scanné", type=["pdf"])

extracted_text = ""

if uploaded_file:
    st.info("Extraction du texte via Tesseract...")

    # Selon le choix, on appelle la fonction correspondante
    if file_type == "Image (PNG/JPG/JPEG)":
        extracted_text = extract_text_from_image(uploaded_file)
    else:
        extracted_text = extract_text_from_pdf_scanned(uploaded_file)

    st.success("Texte extrait (OCR) :")
    st.text_area("Texte OCR", extracted_text, height=200)

# Choix Summarize / Q&A
st.markdown("### 2) Que voulez-vous faire avec le texte extrait ?")
mode = st.radio("Choisissez l'action :", ["Résumé", "Q&A"], index=0)

if extracted_text.strip():
    if mode == "Résumé":
        if st.button("Générer le résumé"):
            with st.spinner("Résumé en cours..."):
                summary = summarize_text(extracted_text)
            st.markdown("### Résumé :")
            st.write(summary)
    else:
        # Q&A
        question = st.text_input("Posez votre question en français ici :")
        if st.button("Obtenir la réponse"):
            if question.strip():
                with st.spinner("Recherche de la réponse..."):
                    answer = answer_question(extracted_text, question)
                st.markdown("### Réponse :")
                st.write(answer)
            else:
                st.warning("Veuillez saisir une question.")
