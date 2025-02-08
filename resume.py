from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import re
import tempfile
from docx import Document
from PyPDF2 import PdfReader

# Import du LLM DeepSeek et du générateur de prompts
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------------------------
# Schéma de réponse pour le résumé
# -----------------------------------------------------------------------------
class SummarizeResponse(BaseModel):
    summary: str

# -----------------------------------------------------------------------------
# Création de l'application FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API de Résumé de Document - DeepSeek",
    description="Cette API reçoit un document (PDF, DOCX ou TXT) et renvoie un résumé détaillé généré par DeepSeek en français.",
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
    Extrait le texte d'un fichier PDF, DOCX ou TXT en se basant sur l'extension du fichier.
    """
    suffix = upload.filename.split(".")[-1].lower()
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

# -----------------------------------------------------------------------------
# Fonction utilitaire pour nettoyer la réponse du LLM
# -----------------------------------------------------------------------------
def remove_thinking_tags(txt: str) -> str:
    """Supprime toute trace de chain-of-thought dans le texte généré."""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

# -----------------------------------------------------------------------------
# Fonction de résumé via DeepSeek
# -----------------------------------------------------------------------------
def summarize_document(document_text: str) -> str:
    """
    Utilise le modèle DeepSeek pour générer un résumé détaillé en français.
    Le prompt demande d'extraire tous les points importants du document.
    """
    template = """
Tu es un expert en résumé de documents. Analyse attentivement le texte ci-dessous et produis un résumé en mentionnant tous les points importants et essentiels. Sois complet et précis.
Réponds toujours en français. tu ne dois pas afficher les noms et autres garde l'anonymat 
si c'est un cv souligne les grands points 
PLEASE DON'T LIE , PLEASE DON;T CREATE THINGS 

Document:
{document}

Résumé:
"""
    prompt_text = template.format(document=document_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_summary = chain.invoke({})
    return remove_thinking_tags(raw_summary)

# -----------------------------------------------------------------------------
# Endpoint principal : /api/summarize
# -----------------------------------------------------------------------------
@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(
    file: UploadFile = File(..., description="Document à résumer (PDF, DOCX ou TXT)")
) -> SummarizeResponse:
    """
    Reçoit un fichier (PDF, DOCX ou TXT), en extrait le texte, puis génère un résumé détaillé en français.
    """
    document_text = extract_text_from_file(file)
    summary = summarize_document(document_text)
    return SummarizeResponse(summary=summary)

# -----------------------------------------------------------------------------
# Lancement de l'API (mode développement)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
