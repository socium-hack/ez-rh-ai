from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import uvicorn
import re
import tempfile
from docx import Document
from PyPDF2 import PdfReader

# Import du LLM DeepSeek et du générateur de prompt via LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# -----------------------------------------------------------------------------
# Schéma de réponse (Pydantic)
# -----------------------------------------------------------------------------
class QuestionsResponse(BaseModel):
    questions: str

# -----------------------------------------------------------------------------
# Création de l'application FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API de Génération de Questions d’Entretien Basées sur CV",
    description="Cette API reçoit un CV (format PDF) et génère des questions d'entretien pertinentes basées sur le contenu du CV, en français.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Fonctions d'extraction de texte
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def extract_text_from_docx(docx_path: str) -> str:
    text = ""
    doc = Document(docx_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_file(upload: UploadFile) -> str:
    """
    Extrait le texte d'un fichier (PDF, DOCX ou TXT) en se basant sur l'extension.
    Remet le curseur de lecture au début pour éviter les erreurs lors d'upload multiple.
    """
    suffix = upload.filename.split(".")[-1].lower()
    upload.file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp_file:
        tmp_file.write(upload.file.read())
        file_path = tmp_file.name

    if suffix == "pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif suffix == "txt":
        return extract_text_from_txt(file_path)
    else:
        # En cas de type inconnu, on tente de lire comme texte
        return extract_text_from_txt(file_path)

# Pour les CV, nous utilisons la même fonction
load_and_extract_text = extract_text_from_file

# -----------------------------------------------------------------------------
# Fonction d'anonymisation
# -----------------------------------------------------------------------------
def anonymize_text(text: str) -> str:
    """
    Masque les adresses e-mail et numéros de téléphone pour préserver l'anonymat.
    """
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL MASQUÉ]", text)
    text = re.sub(r"\+?\d[\d \-\.]{7,}\d", "[TEL MASQUÉ]", text)
    return text

# -----------------------------------------------------------------------------
# Fonction pour supprimer le chain-of-thought
# -----------------------------------------------------------------------------
def remove_thinking_tags(txt: str) -> str:
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

# -----------------------------------------------------------------------------
# Fonction pour générer des questions basées sur le CV
# -----------------------------------------------------------------------------
def generate_cv_questions(resume_text: str, num_questions: int = 5) -> str:
    """
    Utilise le modèle DeepSeek pour générer des questions d'entretien basées sur le CV du candidat.
    La réponse est en français et le prompt ne doit pas révéler de chain-of-thought.
    """
    # Prompt en français pour générer des questions à partir du CV
    prompt_template = """
Tu es un expert en recrutement. Tu sais créer des questions d'entretien pertinentes basées sur le CV d'un candidat.
Ne révèle pas ta chaîne de réflexion (<think>).
Réponds toujours en français.

Voici le CV du candidat (anonymisé) :
{resume}

Tâche :
Génère {num_questions} questions d'entretien qui se concentrent sur les expériences, compétences et lacunes potentielles du candidat.
Les questions doivent être concises et directement liées aux informations du CV.

Réponse :
"""
    prompt_text = prompt_template.format(resume=resume_text, num_questions=num_questions)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    # Initialisation du modèle DeepSeek via Ollama
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

# -----------------------------------------------------------------------------
# Endpoint principal : /api/generate_cv_questions
# -----------------------------------------------------------------------------
@app.post("/api/generate_cv_questions", response_model=QuestionsResponse)
async def generate_cv_questions_endpoint(
    cv_file: UploadFile = File(..., description="CV du candidat (format PDF)"),
    num_questions: Optional[int] = 5
) -> QuestionsResponse:
    """
    Reçoit un CV (fichier PDF), en extrait le texte, anonymise le contenu,
    puis génère des questions d'entretien basées sur ce CV.
    La réponse est en français.
    """
    # Extraction du texte du CV
    raw_text = load_and_extract_text(cv_file)
    anonymized_text = anonymize_text(raw_text)
    
    # Ici, on peut éventuellement segmenter le CV s'il est très long.
    # Pour ce code, on utilise directement le texte complet.
    resume_text = anonymized_text
    
    # Générer les questions basées sur le CV
    questions = generate_cv_questions(resume_text, num_questions=num_questions)
    return QuestionsResponse(questions=questions)

# -----------------------------------------------------------------------------
# Lancement de l'API (mode développement)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
