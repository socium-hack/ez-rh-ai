import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import tempfile

# Définition du modèle et du stockage vectoriel
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:7b")

# Template de réponse
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# PDF par défaut (stocké en mémoire)
default_pdf_path = "Prise de Masse.pdf"

# Fonction pour charger un PDF depuis un fichier temporaire
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

# Fonction pour segmenter le texte
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Fonction pour indexer les documents
def index_docs(documents):
    vector_store.add_documents(documents)

# Fonction pour récupérer les documents pertinents
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Fonction pour générer une réponse
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Interface utilisateur Streamlit
st.set_page_config(page_title=" prise de masse ", layout="wide")

st.sidebar.title("📂 Options")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

# Chargement du PDF (uploadé ou par défaut)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_file.name
    st.sidebar.success(f"✅ {uploaded_file.name} chargé !")
else:
    temp_pdf_path = default_pdf_path
    st.sidebar.info(f"📄 Utilisation du PDF par défaut : {default_pdf_path}")

# Traitement du PDF
st.sidebar.write("🔄 Chargement du PDF...")
documents = load_pdf(temp_pdf_path)
chunked_documents = split_text(documents)
index_docs(chunked_documents)
st.sidebar.success("✅ PDF indexé avec succès !")

# Affichage de l'interface de chat
st.title("💬 Pose ta question sur la prise de masse")
question = st.chat_input("Pose ta question ici...")

if question:
    st.chat_message("user").write(question)
    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)
    st.chat_message("assistant").write(answer)
