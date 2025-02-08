from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, List, Tuple
import uvicorn
import re
import tempfile
from docx import Document
from PyPDF2 import PdfReader

# Imports LangChain et Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# -----------------------------------------------------------------------------
# Schémas de données (Pydantic)
# -----------------------------------------------------------------------------
class QuestionsResponse(BaseModel):
    questions: str

class QAResponse(BaseModel):
    answer: str

class CandidateMatchResult(BaseModel):
    filename: str
    score: int
    reasons: str

class MatchingResponse(BaseModel):
    results: List[CandidateMatchResult]

class SummarizeResponse(BaseModel):
    summary: str

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
        return extract_text_from_txt(file_path)

# -----------------------------------------------------------------------------
# Fonction d'anonymisation et suppression du chain-of-thought
# -----------------------------------------------------------------------------
def anonymize_text(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL MASQUÉ]", text)
    text = re.sub(r"\+?\d[\d \-\.]{7,}\d", "[TEL MASQUÉ]", text)
    return text

def remove_thinking_tags(txt: str) -> str:
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

# -----------------------------------------------------------------------------
# Fonctions spécifiques aux endpoints
# -----------------------------------------------------------------------------

# Génération de questions d'entretien basées sur le CV
def generate_cv_questions(resume_text: str, num_questions: int = 5) -> str:
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
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

# Q&A sur un document (RAG)
def load_pdf(file: UploadFile) -> List:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.file.read())
        file_path = tmp_file.name
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def answer_question(question: str, documents) -> str:
    context = "\n\n".join([doc.page_content for doc in documents])
    qa_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
GIVE ALL THE IMPORTANT DETAILS AND keep the answer concise.
REPONDS TOUJOURS EN FRANCAIS SAUF SI LA QUESTION EST POSEE EN ANGLAIS 
Do not include or reveal your internal reasoning or any <think> section.

Question: {question}
Context: {context}
Answer:
"""
    prompt = ChatPromptTemplate.from_template(qa_template)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_answer = chain.invoke({"question": question, "context": context})
    return remove_thinking_tags(raw_answer)

# Matching CV et offre d'emploi
def analyze_match(cv_text: str, job_text: str) -> str:
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

def parse_matching_result(result_text: str) -> Tuple[int, str]:
    score_match = re.search(r"SCORE:\s*(\d+)", result_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = 0
    reasons = ""
    reasons_match = re.search(r"RAISONS:\s*(.*)", result_text, re.IGNORECASE | re.DOTALL)
    if reasons_match:
        reasons = reasons_match.group(1).strip()
    return score, reasons

# Résumé de document
def summarize_document(document_text: str) -> str:
    template = """
Tu es un expert en résumé de documents. Analyse attentivement le texte ci-dessous et produis un résumé en mentionnant tous les points importants et essentiels. Sois complet et précis.
Réponds toujours en français. tu ne dois pas afficher les noms et autres garde l'anonymat 
si c'est un cv souligne les grands points 
PLEASE DON'T LIE , PLEASE DON'T CREATE THINGS

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
# Création de l'application FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API de DeepSeek HR Solutions",
    description="Rassemblement des endpoints pour génération de questions, Q&A sur documents, matching CV et résumé de document.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

# 1. Génération de questions d'entretien basées sur un CV
@app.post("/api/generate_cv_questions", response_model=QuestionsResponse)
async def generate_cv_questions_endpoint(
    cv_file: UploadFile = File(..., description="CV du candidat (format PDF)"),
    num_questions: Optional[int] = 5
) -> QuestionsResponse:
    raw_text = extract_text_from_file(cv_file)
    anonymized_text = anonymize_text(raw_text)
    questions = generate_cv_questions(anonymized_text, num_questions=num_questions)
    return QuestionsResponse(questions=questions)

# 2. Q&A sur un document (RAG)
@app.post("/api/qa", response_model=QAResponse)
async def qa_on_document(
    question: str = Form(..., description="La question à poser au document"),
    file: UploadFile = File(..., description="Le fichier PDF à interroger")
) -> QAResponse:
    documents = load_pdf(file)
    chunked_docs = split_text(documents)
    embeddings_instance = OllamaEmbeddings(model="deepseek-r1:7b")
    vector_store = InMemoryVectorStore(embeddings_instance)
    vector_store.add_documents(chunked_docs)
    related_docs = vector_store.similarity_search(question, k=3)
    answer = answer_question(question, related_docs)
    return QAResponse(answer=answer)

# 3. Matching CV – Offre d'emploi
@app.post("/api/match", response_model=MatchingResponse)
async def match_resumes(
    job_file: UploadFile = File(..., description="Fichier décrivant l'offre d'emploi (PDF, TXT ou DOCX)"),
    cv_files: List[UploadFile] = File(..., description="CV au format PDF, DOCX ou TXT")
) -> MatchingResponse:
    job_text = extract_text_from_file(job_file)
    results = []
    for cv in cv_files:
        cv_text = extract_text_from_file(cv)
        anonymized_cv_text = anonymize_text(cv_text)
        raw_llm = analyze_match(anonymized_cv_text, job_text)
        score, reasons = parse_matching_result(raw_llm)
        candidate_result = CandidateMatchResult(
            filename=cv.filename,
            score=score,
            reasons=reasons
        )
        results.append(candidate_result)
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    return MatchingResponse(results=sorted_results)

# 4. Résumé de document
@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(
    file: UploadFile = File(..., description="Document à résumer (PDF, DOCX ou TXT)")
) -> SummarizeResponse:
    document_text = extract_text_from_file(file)
    summary = summarize_document(document_text)
    return SummarizeResponse(summary=summary)

# -----------------------------------------------------------------------------
# Lancement de l'API (mode développement)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
