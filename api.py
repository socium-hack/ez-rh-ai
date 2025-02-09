from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple
import uvicorn
import re
import tempfile
from docx import Document
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import tempfile
import os
import ollama

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

class OCRResponse(BaseModel):
    text: str

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
        # Fallback
        return extract_text_from_txt(file_path)

# -----------------------------------------------------------------------------
# Anonymisation + Remove <think>
# -----------------------------------------------------------------------------
def anonymize_text(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL MASQUÉ]", text)
    text = re.sub(r"\+?\d[\d \-\.]{7,}\d", "[TEL MASQUÉ]", text)
    return text

def remove_thinking_tags(txt: str) -> str:
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

# -----------------------------------------------------------------------------
# Fonctions pour chaque endpoint
# -----------------------------------------------------------------------------

# 1) Génération de Questions d’entretien (CV)
def generate_cv_questions(resume_text: str, num_questions: int = 5) -> str:
    prompt_template = """
You are an expert in recruitment. You know how to create relevant interview questions based on a candidate's CV.
Do not reveal your chain-of-thought (<think>).
You MUST always answer in French, and do NOT respond in English or Chinese.
ALWAYS ANSWER IN FRENCH PLEASE 

Here is the anonymized CV:
{resume}

Task:
Generate {num_questions} interview questions that focus on the candidate's experiences, skills, and potential gaps.
The questions must be concise and directly linked to the CV's information.

Answer:
"""
    prompt_text = prompt_template.format(resume=resume_text, num_questions=num_questions)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-v2")  # ou "deepseek-v2", etc.
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

# 2) Q&A sur un Document PDF (RAG)
def load_pdf(file: UploadFile):
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
    model = OllamaLLM(model="deepseek-v2")
    chain = prompt | model
    raw_answer = chain.invoke({"question": question, "context": context})
    return remove_thinking_tags(raw_answer)

# 3) Matching CV - Offre
def analyze_match(cv_text: str, job_text: str) -> str:
    template = """
You are an expert at matching a candidate's resume (CV) to a job description. 
However, your justification (the reasons) must come strictly from the candidate's CV. 
Do NOT create new content, do NOT rely on or pull details from the job request. 
You MUST always answer in French, and not in any other language.
you must always give the reasons and you should classify all the cvs you received by score tooif you dont give reasons dont make content 
never give the candidate name please , please give the reasons please !!
never mention the candidate name just call him the candidate 
Here is the anonymized CV:
{resume}

Task:
1) Provide a "score" from 0 to 100 for how well this CV matches the job, 
   but base that decision purely on the CV's content (experiences, skills, etc.).
2) Explain that score in a single block of text (RAISONS), focusing exclusively on the CV's data.
   If the CV lacks certain information, do not invent or infer details from the job request.
   please always give the reason 

Use this output format (in French):

SCORE: XX
RAISONS:
[Single block of text in French, referencing only the CV content]

Answer:
"""
    prompt_text = template.format(resume=cv_text, job_description=job_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-r1:14b")
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

# 4) Résumé de document
def summarize_document(document_text: str) -> str:
    template = r"""
You are an expert at summarizing documents. Analyze the text below carefully and produce a summary 
mentioning all the important and essential points. Be thorough and precise.
You MUST always respond in French. Do not reveal any personal names or details, and maintain anonymity.
If it's a CV, highlight the key points.
PLEASE DO NOT LIE, PLEASE DO NOT INVENT ANY CONTENT. PLEASE BE CONCISE.

Document:
{document}

Résumé:
"""
    prompt_text = template.format(document=document_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model="deepseek-v2")
    chain = prompt | model
    raw_summary = chain.invoke({})
    return remove_thinking_tags(raw_summary)

# 5) Classification de Document RH
def classify_document(doc_text: str) -> str:
    """
    Classer (CV, lettre de motivation, offre d’emploi, autre doc RH).
    Réponse en français.
    """
    classification_prompt = r"""
You are an expert in HR document classification. We provide you a text from a file.
You MUST always respond in French.
Classify this text into one of these categories:
- "CV" (si le document ressemble à un résumé de profil, expériences, compétences)
- "Lettre de motivation"
- "Offre d’emploi"
- "Autre document RH" (if none of the above match)

You must also provide a short justification (in French) referencing the text content,
but do not reveal chain-of-thought (<think>).

Format output strictly as:

DOC_TYPE: [one of: CV, Lettre de motivation, Offre d’emploi, Autre document RH]
JUSTIFICATION:
[Single short paragraph in French]

Document text:
{doc_text}

Answer:
"""
    prompt_text = classification_prompt.format(doc_text=doc_text)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    # NOTE: ici on utilise un autre modèle (par ex. "deepseek-r1:32b") si vous le voulez
    model = OllamaLLM(model="deepseek-v2")
    chain = prompt | model
    raw_result = chain.invoke({})
    return remove_thinking_tags(raw_result)

def parse_classification_result(result_text: str) -> tuple[str, str]:
    doc_type_match = re.search(r"DOC_TYPE:\s*(.*)", result_text, re.IGNORECASE)
    doc_type = doc_type_match.group(1).strip() if doc_type_match else "Inconnu"

    justification_match = re.search(r"JUSTIFICATION:\s*(.*)", result_text, re.IGNORECASE | re.DOTALL)
    justification = justification_match.group(1).strip() if justification_match else ""

    return doc_type, justification

# -----------------------------------------------------------------------------
# Création de l'application FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API de DeepSeek HR Solutions",
    description="Endpoints pour génération de questions, Q&A (RAG), matching CV, résumé, et classification de documents RH.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

# (1) Génération de questions d'entretien (CV)
@app.post("/api/generate_cv_questions", response_model=QuestionsResponse)
async def generate_cv_questions_endpoint(
    cv_file: UploadFile = File(..., description="CV du candidat (PDF, DOCX ou TXT)"),
    num_questions: Optional[int] = 5
) -> QuestionsResponse:
    try:
        raw_text = extract_text_from_file(cv_file)
        anonymized_text = anonymize_text(raw_text)
        questions = generate_cv_questions(anonymized_text, num_questions=num_questions)
        return QuestionsResponse(questions=questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (2) Q&A sur un document PDF (RAG)
@app.post("/api/qa", response_model=QAResponse)
async def qa_on_document(
    question: str = Form(..., description="La question à poser au document"),
    file: UploadFile = File(..., description="Le fichier PDF à interroger")
) -> QAResponse:
    try:
        documents = load_pdf(file)
        chunked_docs = split_text(documents)
        embeddings_instance = OllamaEmbeddings(model="deepseek-r1:14b")
        vector_store = InMemoryVectorStore(embeddings_instance)
        vector_store.add_documents(chunked_docs)
        related_docs = vector_store.similarity_search(question, k=3)
        answer = answer_question(question, related_docs)
        return QAResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (3) Matching CV - Offre d'emploi
@app.post("/api/match", response_model=MatchingResponse)
async def match_resumes(
    job_file: UploadFile = File(..., description="Fichier décrivant l'offre d'emploi (PDF, DOCX ou TXT)"),
    cv_files: List[UploadFile] = File(..., description="CV au format PDF, DOCX ou TXT")
) -> MatchingResponse:
    try:
        job_text = extract_text_from_file(job_file)
        results = []
        for cv in cv_files:
            cv_text = extract_text_from_file(cv)
            anonymized_cv_text = anonymize_text(cv_text)
            # On pourrait décider de NE PAS passer job_text, 
            # car le prompt indique de ne pas s'appuyer sur l'offre
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (4) Résumé de document
@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(
    file: UploadFile = File(..., description="Document à résumer (PDF, DOCX ou TXT)")
) -> SummarizeResponse:
    try:
        document_text = extract_text_from_file(file)
        summary = summarize_document(document_text)
        return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (5) Classification de documents RH
@app.post("/api/classify_file", response_model=FileClassification)
async def classify_file_endpoint(
    file: UploadFile = File(..., description="Fichier RH (PDF, DOCX, TXT)")
) -> FileClassification:
    """
    Reçoit un fichier RH et classe le document: 
    'CV', 'Lettre de motivation', 'Offre d’emploi' ou 'Autre document RH'
    """
    try:
        doc_text = extract_text_from_file(file)
        llm_result = classify_document(doc_text)
        doc_type, justification = parse_classification_result(llm_result)
        return FileClassification(doc_type=doc_type, justification=justification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(..., description="Fichier image ou PDF scanné")):
    """
    Reçoit un fichier (image ou PDF scanné), utilise Llama3.2-Vision via Ollama pour extraire le texte,
    et renvoie le texte extrait.
    """
    try:
        # 1) Sauvegarder le fichier uploadé dans un fichier temporaire
        suffix = file.filename.split(".")[-1].lower() if file.filename else "jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(await file.read())
            file_path = tmp.name

        # 2) Construire le prompt pour l'OCR
        user_content = (
            "Extract and return all text from this image using OCR. "
            "Return the text as plain text."
            
        )

        # 3) Appeler ollama.chat avec le modèle Llama3.2-Vision
        response = ollama.chat(
            model="llama3.2-vision",
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                    "images": [file_path]  # On passe le chemin du fichier temporaire
                }
            ],
            options={"temperature": 0}
        )

        # 4) Récupérer le texte extrait depuis la réponse
        raw_text = response["message"]["content"]

        # 5) Nettoyer et supprimer le fichier temporaire
        os.remove(file_path)

        return OCRResponse(text=raw_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)