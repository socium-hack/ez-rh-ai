
# import os
# import streamlit as st
# import pdfplumber
# import ollama
# from PIL import Image
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama.llms import OllamaLLM

# # Configuration du modèle DeepSeek-R1
# MODEL_NAME = "deepseek-r1:7b"
# embeddings = OllamaEmbeddings(model=MODEL_NAME)
# vector_store = InMemoryVectorStore(embeddings)
# llm = OllamaLLM(model=MODEL_NAME)

# def extract_text_from_pdf(pdf_path):
#     """Extrait le texte d'un PDF."""
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text

# def extract_text_from_image(image_path):
#     """Utilise un modèle de vision pour extraire du texte d'une image."""
#     res = ollama.chat(
#         model="llama3.2-vision",
#         messages=[
#             {"role": "user", "content": "Extract the text from this resume.", "images": [image_path]}
#         ]
#     )
#     return res['message']['content']

# def analyze_resume(text):
#     """Analyse le CV et retourne les informations clés."""
#     prompt = f"""
#     You are an AI specialized in resume analysis. Extract the following information:
#     - Name
#     - Contact Information
#     - Work Experience
#     - Education
#     - Skills
#     - Strengths and Weaknesses
#     Resume Content:
#     {text}
#     """
#     return llm.invoke(prompt)

# def main():
#     st.set_page_config(page_title="📄 CV Scanner", layout="wide")
#     st.markdown("""
#         <style>
#             .main-title {
#                 font-size: 36px;
#                 font-weight: bold;
#                 color: #4A90E2;
#                 text-align: center;
#             }
#             .sub-title {
#                 font-size: 20px;
#                 font-weight: bold;
#                 color: #555;
#                 text-align: center;
#             }
#             .stTextArea {
#                 background-color: #f5f5f5;
#                 border-radius: 10px;
#                 padding: 10px;
#             }
#         </style>
#     """, unsafe_allow_html=True)
    
#     st.markdown("<p class='main-title'>📄 CV Scanner - Analyse Automatique de CV</p>", unsafe_allow_html=True)
#     st.markdown("<p class='sub-title'>Téléchargez votre CV en PDF ou image et obtenez une analyse détaillée</p>", unsafe_allow_html=True)
    
#     uploaded_file = st.file_uploader("Téléchargez un CV (PDF ou Image)", type=["pdf", "jpg", "png"], help="Formats acceptés : PDF, JPG, PNG")
    
#     if uploaded_file:
#         file_path = f"temp_{uploaded_file.name}"
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         with st.spinner("📖 Extraction du texte..."):
#             if uploaded_file.type == "application/pdf":
#                 text = extract_text_from_pdf(file_path)
#             else:
#                 text = extract_text_from_image(file_path)
        
#         st.subheader("📝 Texte extrait du CV")
#         st.text_area("", text, height=250, key="extracted_text")
        
#         with st.spinner("🔍 Analyse du CV en cours..."):
#             result = analyze_resume(text)
        
#         st.subheader("📊 Résultats de l'analyse")
#         st.success(result)
        
#         os.remove(file_path)

# if __name__ == "__main__":
#     main()
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
