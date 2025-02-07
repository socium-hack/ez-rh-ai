from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Tuple
import uvicorn
import re
import tempfile

# Pour l'extraction du document PDF
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# Pour le prompt et le LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# Pour définir le schéma de la réponse JSON
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Schéma de la réponse (Pydantic)
# -----------------------------------------------------------------------------
class QAResponse(BaseModel):
    answer: str

# -----------------------------------------------------------------------------
# Création de l'application FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API de Q&A sur Document (RAG)",
    description="Cette API reçoit une question et un fichier PDF, puis renvoie une réponse générée en se basant sur les parties pertinentes du document.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------
def remove_thinking_tags(txt: str) -> str:
    """Supprime toute trace de chain-of-thought dans le texte généré par le LLM."""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

def load_pdf(file: UploadFile) -> List:
    """Extrait le contenu d'un PDF à l'aide de PDFPlumberLoader."""
    # Sauvegarde le fichier dans un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.file.read())
        file_path = tmp_file.name
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Divise le contenu des documents en morceaux (chunks) pour le traitement."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def answer_question(question: str, documents) -> str:
    """
    Construit le prompt pour la Q&A et interroge le LLM.
    Le prompt demande à obtenir une réponse concise en français.
    """
    # Concatène le contenu des documents (les chunks récupérés)
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

# -----------------------------------------------------------------------------
# Endpoint principal : /api/qa
# -----------------------------------------------------------------------------
@app.post("/api/qa", response_model=QAResponse)
async def qa_on_document(
    question: str = Form(..., description="La question à poser au document"),
    file: UploadFile = File(..., description="Le fichier PDF à interroger")
) -> QAResponse:
    """
    Reçoit une question et un fichier PDF.
    1. Charge le document PDF.
    2. Le divise en chunks.
    3. Crée un vector store local pour indexer les chunks.
    4. Recherche les parties les plus pertinentes en fonction de la question.
    5. Interroge le LLM pour générer une réponse concise.
    6. Retourne la réponse.
    """
    # 1. Charger le document PDF
    documents = load_pdf(file)

    # 2. Diviser le contenu en morceaux
    chunked_docs = split_text(documents)

    # 3. Créer un vector store local avec les embeddings
    embeddings_instance = OllamaEmbeddings(model="deepseek-r1:7b")
    vector_store = InMemoryVectorStore(embeddings_instance)
    vector_store.add_documents(chunked_docs)

    # 4. Rechercher les chunks pertinents pour la question (top 3)
    related_docs = vector_store.similarity_search(question, k=3)

    # 5. Interroger le LLM pour générer la réponse
    answer = answer_question(question, related_docs)

    # 6. Retourner la réponse
    return QAResponse(answer=answer)

# -----------------------------------------------------------------------------
# Lancement de l'API avec Uvicorn
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
