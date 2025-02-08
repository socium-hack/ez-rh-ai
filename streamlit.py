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
