from fastapi import FastAPI, File, UploadFile
from typing import List, Tuple
import uvicorn
import re
import tempfile
from docx import Document
from PyPDF2 import PdfReader

# Pour le matching via un LLM local (DeepSeek)
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Pour la gestion du format de la réponse (JSON)
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Schémas de données (Pydantic)
# ---------------------------------------------------------------------------
class CandidateMatchResult(BaseModel):
    """Résultat du matching pour un CV donné."""
    filename: str
    score: int
    reasons: str  # Remplacé List[str] par str pour un bloc de texte unique

class MatchingResponse(BaseModel):
    """
    Réponse globale, contenant la liste de tous les CV classés
    ou un seul si un seul CV.
    """
    results: List[CandidateMatchResult]

# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Matching CV - Offre d’Emploi",
    description=(
        "Cette API reçoit un fichier décrivant l'offre d'emploi et un ou plusieurs CV, "
        "puis renvoie un classement des CV par score de compatibilité, "
        "avec une justification textuelle axée sur le contenu du CV, en français."
    ),
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Fonctions d'extraction de texte pour PDF, DOCX et TXT
# ---------------------------------------------------------------------------
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
    Extrait le texte d'un fichier (PDF, DOCX, TXT) en se basant sur son extension.
    """
    suffix = upload.filename.split(".")[-1].lower() if upload.filename else "txt"
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
        # Par défaut, on tente de lire comme texte
        return extract_text_from_txt(file_path)

# Pour les CV, même approche
load_and_extract_text = extract_text_from_file

# ---------------------------------------------------------------------------
# Fonctions d’anonymisation et de matching
# ---------------------------------------------------------------------------
def anonymize_text(text: str) -> str:
    """Masque emails et téléphones."""
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL MASQUÉ]", text)
    text = re.sub(r"\+?\d[\d \-\.]{7,}\d", "[TEL MASQUÉ]", text)
    return text

def analyze_match(cv_text: str, job_text: str) -> str:
    """
    Interroge le LLM (DeepSeek) pour obtenir un score (0-100) et une justification
    dans un seul bloc de texte. Le texte doit faire référence au contenu du CV.
    """
    template = """
Tu es un expert en matching entre un CV et une demande d'emploi.
Ne révèle pas tes chain-of-thought (<think>).
Réponds toujours en français.

But : Évaluer la compatibilité du CV vis-à-vis du poste, 
mais concentre-toi principalement sur le contenu du CV (compétences, expériences, etc.).

Voici le CV (anonymisé) :
{resume}

Voici la description du poste :
{job_description}

Exigences pour la réponse :
1) Donne un "score" entre 0 et 100 pour la compatibilité de ce CV avec le poste.
2) Dans un seul bloc de texte (RAISONS), justifie ce score en t'appuyant explicitement sur le contenu du CV.
3) Utilise ce format :

SCORE: XX
RAISONS:
Texte unique ici, un paragraphe

Réponse :
"""
    prompt_text = template.format(resume=cv_text, job_description=job_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

def remove_thinking_tags(txt: str) -> str:
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

def parse_matching_result(result_text: str) -> Tuple[int, str]:
    """
    Extrait le SCORE et la section RAISONS (un bloc de texte) du résultat.
    Retourne (score, reasons_text).
    """
    # SCORE
    score_match = re.search(r"SCORE:\s*(\d+)", result_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = 0

    # RAISONS
    reasons = ""
    reasons_match = re.search(r"RAISONS:\s*(.*)", result_text, re.IGNORECASE | re.DOTALL)
    if reasons_match:
        reasons = reasons_match.group(1).strip()

    return score, reasons

# ---------------------------------------------------------------------------
# Endpoint : /api/match
# ---------------------------------------------------------------------------
@app.post("/api/match", response_model=MatchingResponse)
async def match_resumes(
    job_file: UploadFile = File(..., description="Fichier décrivant l'offre d'emploi (PDF, TXT ou DOCX)"),
    cv_files: List[UploadFile] = File(..., description="CV au format PDF, DOCX ou TXT")
) -> MatchingResponse:
    """
    Reçoit un fichier pour l'offre (job_file) et un ou plusieurs CV (cv_files).
    Retourne un classement par score (0-100) + justification en un seul bloc.
    """
    # Extraction du texte pour la description de poste
    job_text = extract_text_from_file(job_file)

    results = []
    for cv in cv_files:
        cv_text = load_and_extract_text(cv)
        anonymized_cv_text = anonymize_text(cv_text)
        raw_llm = analyze_match(anonymized_cv_text, job_text)

        score, reasons = parse_matching_result(raw_llm)
        # Créer un CandidateMatchResult
        candidate_result = CandidateMatchResult(
            filename=cv.filename,
            score=score,
            reasons=reasons
        )
        results.append(candidate_result)

    # Tri par score décroissant
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

    return MatchingResponse(results=sorted_results)

# ---------------------------------------------------------------------------
# main - Lancement
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
