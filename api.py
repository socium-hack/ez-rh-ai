from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple
import uvicorn
import re
import tempfile
from docx import Document
from PyPDF2 import PdfReader

# LangChain / DeepSeek imports
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

class FileClassification(BaseModel):
    doc_type: str
    justification: str

# -----------------------------------------------------------------------------
# Fonctions d'extraction de texte
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extraction du texte d’un PDF via PyPDF2."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def extract_text_from_docx(docx_path: str) -> str:
    """Extraction du texte d’un DOCX via python-docx."""
    text = ""
    doc = Document(docx_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(txt_path: str) -> str:
    """Extraction d’un .txt en UTF-8 (ignore errors)."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_file(upload: UploadFile) -> str:
    """
    Extrait le texte d'un fichier (PDF, DOCX, TXT), fallback sur txt si extension inconnue.
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
        # Fallback : lecture en .txt
        return extract_text_from_txt(file_path)

# -----------------------------------------------------------------------------
# Anonymisation + suppression du chain-of-thought
# -----------------------------------------------------------------------------
def anonymize_text(text: str) -> str:
    """Masque adresses email et numéros de téléphone."""
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL MASQUÉ]", text)
    text = re.sub(r"\+?\d[\d \-\.]{7,}\d", "[TEL MASQUÉ]", text)
    return text

def remove_thinking_tags(txt: str) -> str:
    """Retire toute balise <think>."""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

# -----------------------------------------------------------------------------
# 1) Génération de Questions (CV)
# -----------------------------------------------------------------------------
def generate_cv_questions(resume_text: str, num_questions: int = 5) -> str:
    """
    Prompt pour générer des questions en français, en se basant sur le CV.
    """
    prompt_template = """
You are an expert in recruitment. You know how to create relevant interview questions based on a candidate's CV.
Do not reveal your chain-of-thought (<think>).
You MUST always answer in French.

Here is the anonymized CV:
{resume}

Task:
Generate {num_questions} interview questions that focus on the candidate's experiences, skills, and potential gaps.
Answer:
"""
    final_prompt = prompt_template.format(resume=resume_text, num_questions=num_questions)
    prompt = ChatPromptTemplate.from_template(final_prompt)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_res = chain.invoke({})
    return remove_thinking_tags(raw_res)

# -----------------------------------------------------------------------------
# 2) Q&A sur un Document PDF (RAG)
# -----------------------------------------------------------------------------
def load_pdf(file: UploadFile):
    """Charge un PDF via PDFPlumberLoader et renvoie une liste de Documents."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.file.read())
        file_path = tmp_file.name
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(documents)

def answer_question(question: str, documents) -> str:
    """
    Construit un prompt Q&A, toujours en français.
    """
    context = "\n\n".join([doc.page_content for doc in documents])
    qa_template = """
You are an assistant for question-answering tasks. Use the following context to answer the question.
If you don't know, say "Je ne sais pas."
Answer in French only.

Question: {question}
Context: {context}
Answer:
"""
    final_prompt = qa_template.format(question=question, context=context)
    prompt = ChatPromptTemplate.from_template(final_prompt)
    model = OllamaLLM(model="deepseek-r1:32b")  # un autre modele
    chain = prompt | model
    raw_answer = chain.invoke({"question": question, "context": context})
    return remove_thinking_tags(raw_answer)

# -----------------------------------------------------------------------------
# 3) Matching CV - Offre d’emploi
# -----------------------------------------------------------------------------
def analyze_match(cv_text: str, job_text: str) -> str:
    """
    Prompt : score (0-100) + justification tirée du CV uniquement.
    """
    template = """
You are an expert at matching a candidate's resume (CV) to a job description.
However, your justification must come strictly from the CV (no content from the job).
You MUST always answer in French.

Here is the anonymized CV:
{resume}

Task:
1) Provide a "score" (0-100),
2) Explain that score in a single block of text, referencing only the CV's data.

Format:
SCORE: XX
RAISONS:
[One block of text in French]

Answer:
"""
    final_prompt = template.format(resume=cv_text, job_description=job_text)
    prompt = ChatPromptTemplate.from_template(final_prompt)
    model = OllamaLLM(model="deepseek-r1:7b")
    chain = prompt | model
    raw_res = chain.invoke({})
    return remove_thinking_tags(raw_res)

def parse_matching_result(result_text: str) -> Tuple[int, str]:
    score_match = re.search(r"SCORE:\s*(\d+)", result_text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 0
    reasons = ""
    reasons_match = re.search(r"RAISONS:\s*(.*)", result_text, re.IGNORECASE | re.DOTALL)
    if reasons_match:
        reasons = reasons_match.group(1).strip()
    return score, reasons

# -----------------------------------------------------------------------------
# 4) Résumé de document
# -----------------------------------------------------------------------------
def summarize_document(document_text: str) -> str:
    template = """
You are an expert at summarizing documents. Provide a concise summary in French only.

Document:
{document}

Résumé:
"""
    final_prompt = template.format(document=document_text)
    prompt = ChatPromptTemplate.from_template(final_prompt)
    model = OllamaLLM(model="deepseek-v2")
    chain = prompt | model
    raw_summary = chain.invoke({})
    return remove_thinking_tags(raw_summary)

# -----------------------------------------------------------------------------
# 5) Classification de Documents RH
# -----------------------------------------------------------------------------
def classify_document(doc_text: str) -> str:
    classification_prompt = r"""
You are an expert in HR document classification. 
You MUST respond in French only.
Classify the text into one of these categories:
- "CV"
- "Lettre de motivation"
- "Offre d’emploi"
- "Demande d'emploi"
- "Contrat"
- "Avis de recrutement"
- "Attestation de travail"
- "Attestation de stage"
- "Autre document RH"

Provide a short justification in French, referencing the text content.
Do not reveal chain-of-thought (<think>).

Format strictly:

DOC_TYPE: ...
JUSTIFICATION:
[Short paragraph in French]

Document text:
{doc_text}

Answer:
"""
    final_prompt = classification_prompt.format(doc_text=doc_text)
    prompt = ChatPromptTemplate.from_template(final_prompt)
    model = OllamaLLM(model="deepseek-v2")
    chain = prompt | model
    raw_res = chain.invoke({})
    return remove_thinking_tags(raw_res)


def parse_classification_result(result_text: str) -> Tuple[str, str]:
    """
    On suppose un format strict:
    DOC_TYPE: ...
    JUSTIFICATION:
    ...
    """
    # 1) On capture doc_type jusqu’à la fin de la ligne
    doc_type_match = re.search(r"DOC_TYPE:\s*(.*?)(\r?\n|$)", result_text, re.IGNORECASE)
    if doc_type_match:
        doc_type = doc_type_match.group(1).strip()
    else:
        doc_type = "Inconnu"

    # 2) On capture justification après "JUSTIFICATION:"
    justification_match = re.search(r"JUSTIFICATION:\s*(.*)", result_text, re.IGNORECASE | re.DOTALL)
    if justification_match:
        justification = justification_match.group(1).strip()
    else:
        justification = ""

    return doc_type, justification

# -----------------------------------------------------------------------------
# Création de l'application FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API de DeepSeek HR Solutions",
    description="Endpoints pour generation de questions, Q&A (RAG), matching CV, résumé, et classification de documents RH.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.post("/api/generate_cv_questions", response_model=QuestionsResponse)
async def generate_cv_questions_endpoint(
    cv_file: UploadFile = File(..., description="CV du candidat (PDF, DOCX, TXT)"),
    num_questions: Optional[int] = 5
) -> QuestionsResponse:
    try:
        raw_text = extract_text_from_file(cv_file)
        anonymized_text = anonymize_text(raw_text)
        questions = generate_cv_questions(anonymized_text, num_questions=num_questions)
        return QuestionsResponse(questions=questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/qa", response_model=QAResponse)
async def qa_on_document(
    question: str = Form(..., description="La question (en français)"),
    file: UploadFile = File(..., description="Le PDF à interroger")
) -> QAResponse:
    try:
        docs = load_pdf(file)
        splitted = split_text(docs)
        emb = OllamaEmbeddings(model="deepseek-r1:32b")
        vs = InMemoryVectorStore(emb)
        vs.add_documents(splitted)
        top_docs = vs.similarity_search(question, k=3)
        answer = answer_question(question, top_docs)
        return QAResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match", response_model=MatchingResponse)
async def match_resumes(
    job_file: UploadFile = File(..., description="Offre d'emploi (PDF, DOCX, TXT)"),
    cv_files: List[UploadFile] = File(..., description="CV (PDF, DOCX, TXT) multiples")
) -> MatchingResponse:
    try:
        job_text = extract_text_from_file(job_file)
        results_list = []
        for cv_file in cv_files:
            cv_text = extract_text_from_file(cv_file)
            anonymized_cv = anonymize_text(cv_text)
            raw = analyze_match(anonymized_cv, job_text)
            score, reasons = parse_matching_result(raw)
            candidate_res = CandidateMatchResult(
                filename=cv_file.filename,
                score=score,
                reasons=reasons
            )
            results_list.append(candidate_res)
        results_list.sort(key=lambda x: x.score, reverse=True)
        return MatchingResponse(results=results_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(
    file: UploadFile = File(..., description="Document (PDF, DOCX, TXT) à résumer")
) -> SummarizeResponse:
    try:
        text = extract_text_from_file(file)
        summary = summarize_document(text)
        return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify_file", response_model=FileClassification)
async def classify_file_endpoint(
    file: UploadFile = File(..., description="Document RH (PDF, DOCX, TXT)")
) -> FileClassification:
    """
    Classe un document dans la liste :
    CV, Lettre de motivation, Offre d’emploi, Demande d'emploi, Contrat,
    Avis de recrutement, Attestation de travail, Attestation de stage,
    Autre document RH
    """
    try:
        doc_text = extract_text_from_file(file)
        raw_llm = classify_document(doc_text)
        doc_type, justification = parse_classification_result(raw_llm)
        return FileClassification(doc_type=doc_type, justification=justification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)