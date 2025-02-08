# import streamlit as st
# import os
# import tempfile
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate

# # --------------------------------------------------------------------
# # CONFIGURATION GENERALE STREAMLIT
# # --------------------------------------------------------------------
# st.set_page_config(
#     page_title="DeepSeek HR Solutions",
#     page_icon="📄",
#     layout="wide"
# )

# # --------------------------------------------------------------------
# # INITIALISATION DU MODELE & EMBEDDINGS (SANS CACHE)
# # --------------------------------------------------------------------
# # Attention : Sur un vrai système, recharger le modèle à chaque refresh peut être coûteux
# embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
# model = OllamaLLM(model="deepseek-r1:7b")

# # Crée un vector store vide (utilisé éventuellement pour la partie RAG)
# global_vector_store = InMemoryVectorStore(embeddings)

# # --------------------------------------------------------------------
# # FONCTIONS UTILES : TELECHARGEMENT, LECTURE & SPLIT PDF
# # --------------------------------------------------------------------
# def save_uploaded_file(uploaded_file):
#     """
#     Sauvegarde un fichier uploadé en local (temporaire)
#     et retourne le chemin absolu.
#     """
#     temp_dir = "temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     temp_path = os.path.join(temp_dir, uploaded_file.name)
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return temp_path

# def load_pdf(file_path):
#     """
#     Charge un PDF avec PDFPlumberLoader et renvoie une liste de Documents.
#     """
#     loader = PDFPlumberLoader(file_path)
#     return loader.load()

# def split_text(documents, chunk_size=1000, chunk_overlap=200):
#     """
#     Segmente le texte en chunks avec RecursiveCharacterTextSplitter.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         add_start_index=True
#     )
#     return text_splitter.split_documents(documents)

# # --------------------------------------------------------------------
# # 1) MATCHING CV / OFFRE
# # --------------------------------------------------------------------
# comparison_template = """
# You are an expert in resume analysis. Compare the provided resume with the job description.
# Evaluate the match based on skills, experience, and keywords.
# Do not include or reveal your internal reasoning or any <think> section.


# Resume:
# {resume}

# Job Description:
# {job_description}

# Provide a compatibility score (out of 100) and highlight strengths and weaknesses.
# Answer:
# """

# def analyze_match(resume_text, job_text):
#     """
#     Génère un score de compatibilité + feedback sur les forces/faiblesses.
#     """
#     prompt = ChatPromptTemplate.from_template(comparison_template)
#     chain = prompt | model
#     return chain.invoke({"resume": resume_text, "job_description": job_text})

# # --------------------------------------------------------------------
# # 2) Q&A (RAG)
# # --------------------------------------------------------------------
# qa_template = """
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# GIVE ALL THE IMPORTANT DETAILS AND keep the answer concise.
# REPONDS TOUJOURS EN FRANCAIS SAUF SI LA QUESTION EST POSEE EN ANGLAIS 
# Do not include or reveal your internal reasoning or any <think> section.


# Question: {question}
# Context: {context}
# Answer:
# """

# def retrieve_docs(query, vector_store, k=3):
#     """
#     Récupère les k documents les plus pertinents depuis un vector_store donné.
#     """
#     return vector_store.similarity_search(query, k=k)

# def answer_question(question, documents):
#     """
#     Construit le prompt, interroge le LLM et renvoie une réponse concise.
#     """
#     context = "\n\n".join([doc.page_content for doc in documents])
#     prompt = ChatPromptTemplate.from_template(qa_template)
#     chain = prompt | model
#     return chain.invoke({"question": question, "context": context})

# # --------------------------------------------------------------------
# # INTERFACE STREAMLIT
# # --------------------------------------------------------------------
# st.title("🔍 DeepSeek HR Solutions")

# tabs = st.tabs(["1️⃣ Matching CV - Offre", "2️⃣ Q&A (RAG) sur PDF"])

# # --------------------------------------------------------------------
# #  ONGLET 1 : MATCHING CV & OFFRE
# # --------------------------------------------------------------------
# with tabs[0]:
#     st.header("Analyse de compatibilité entre CV et Offre d'emploi")

#     col1, col2 = st.columns(2)
#     with col1:
#         uploaded_resume = st.file_uploader("📄 Upload ton CV (PDF uniquement)", type=["pdf"])
#     with col2:
#         uploaded_job = st.file_uploader("📑 Upload l'offre d'emploi (PDF ou TXT)", type=["pdf", "txt"])

#     if uploaded_resume and uploaded_job:
#         st.subheader("📊 Résultats de l'analyse")

#         # Sauvegarde des fichiers
#         resume_path = save_uploaded_file(uploaded_resume)
        
#         # Traitement CV
#         with st.spinner("📝 Lecture du CV..."):
#             resume_docs = load_pdf(resume_path)
#             resume_chunks = split_text(resume_docs)
#             resume_text = "\n".join([doc.page_content for doc in resume_chunks])

#         # Traitement Job Description
#         job_text = ""
#         if uploaded_job.type == "application/pdf":
#             job_path = save_uploaded_file(uploaded_job)
#             with st.spinner("📝 Lecture de l'offre..."):
#                 job_docs = load_pdf(job_path)
#                 job_chunks = split_text(job_docs)
#                 job_text = "\n".join([doc.page_content for doc in job_chunks])
#         else:
#             # Si c'est un fichier texte pur
#             job_text = uploaded_job.getvalue().decode("utf-8")

#         # Analyse
#         with st.spinner("📊 Analyse en cours..."):
#             result = analyze_match(resume_text, job_text)

#         st.success("✅ Analyse terminée !")
#         st.write(result)
#     else:
#         st.info("Veuillez uploader un CV et une offre d'emploi pour commencer l'analyse.")

# # --------------------------------------------------------------------
# #  ONGLET 2 : Q&A (RAG) sur PDF
# # --------------------------------------------------------------------
# with tabs[1]:
#     st.header("Recherche et Question-Réponse sur PDF")

#     # Choix du PDF à indexer
#     rag_pdf = st.file_uploader("Upload un PDF pour indexation (RAG)", type=["pdf"])
    
#     if rag_pdf:
#         with st.spinner("🔄 Indexation du PDF..."):
#             # Stockage temporaire
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                 temp_file.write(rag_pdf.getbuffer())
#                 temp_pdf_path = temp_file.name
            
#             # On crée un NOUVEAU vector store propre à ce PDF
#             local_vector_store = InMemoryVectorStore(embeddings)
            
#             # Lecture et segmentation
#             documents = load_pdf(temp_pdf_path)
#             chunked_docs = split_text(documents)
#             local_vector_store.add_documents(chunked_docs)
        
#         st.success(f"✅ PDF {rag_pdf.name} indexé avec succès !")

#         # Champ pour poser la question
#         question = st.text_input("Pose ta question au document :")
#         if question:
#             st.write("**Question** :", question)
#             with st.spinner("💬 Recherche..."):
#                 related_docs = retrieve_docs(question, local_vector_store, k=3)
#                 answer = answer_question(question, related_docs)
#             st.write("**Réponse** :")
#             st.write(answer)
#     else:
#         st.info("Upload un PDF pour activer la fonctionnalité Q&A.")
# Home.py

import streamlit as st

st.set_page_config(
    page_title="DeepSeek HR Solutions",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("Bienvenue sur DeepSeek HR Solutions")
    st.subheader("Un hub local pour analyser des CV, faire du Matching et effectuer du Q&A sur vos documents PDF.")
    
    st.markdown("""
    ### Description du Projet
    
    **DeepSeek HR Solutions** est un outil de démonstration montrant comment on peut utiliser des modèles de langage (LLM) et des embeddings
    pour plusieurs cas d'usage en **recrutement** et en **traitement documentaire** :
    
    - **Matching CV – Offre d’emploi** :  
      Téléversez un CV et une offre, puis obtenez un score de compatibilité et des retours sur les forces/faiblesses.

    - **Recherche et Question-Réponse (RAG) sur PDF** :  
      Indexez un PDF en local, posez-lui des questions en langage naturel et recevez des réponses ciblées.

    ### Navigation
    Pour accéder aux fonctionnalités, utilisez le menu `Pages` sur la gauche :
    - [1️⃣ CV – Offre Matching](./pages/1_CV_Offre_Matching.py)
    - [2️⃣ Q&A (RAG) sur PDF](./pages/2_QA_RAG.py)

    ---
    ### Tech Stack
    - **Streamlit** pour l'interface web
    - **LangChain** & **Ollama** pour l’IA locale (modèle DeepSeek)
    - **PDFPlumber** pour l’extraction de texte PDF
    - **InMemoryVectorStore** pour l’indexation locale des embeddings
    """)
    
    st.info("Naviguez dans le menu à gauche pour découvrir les fonctionnalités.")

if __name__ == "__main__":
    main()
