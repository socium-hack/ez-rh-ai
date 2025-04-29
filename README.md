# HR Solutions API

This project provides a FastAPI-based backend for HR solutions, including functionalities such as:

- **Generating Interview Questions from CVs**  
  Automatically creates interview questions based on a candidate's CV.

- **Document Q&A (RAG)**  
  Answers questions based on uploaded PDF documents by retrieving the most relevant segments.

- **CV and Job Matching**  
  Compares candidate CVs against job descriptions, scoring the match and providing a justification based solely on the CV content.

- **Document Summarization**  
  Summarizes lengthy documents, highlighting all key points in concise French text.

- **HR Document Classification**  
  Classifies HR documents as either a CV, a cover letter ("Lettre de motivation"), a job offer ("Offre d'emploi"), or another HR document type.

- **OCR on Images or Scanned PDFs**  
  Uses the Llama3.2-Vision model via Ollama to extract text from images or scanned documents.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Endpoints](#endpoints)
    - [1. Generate Interview Questions from CV](#1-generate-interview-questions-from-cv)
    - [2. Q&A on Document (RAG)](#2-qa-on-document-rag)
    - [3. CV and Job Matching](#3-cv-and-job-matching)
    - [4. Document Summarization](#4-document-summarization)
    - [5. HR Document Classification](#5-hr-document-classification)
    - [6. OCR Endpoint](#6-ocr-endpoint)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements

- Python 3.8 or above
- FastAPI
- uvicorn
- pydantic
- [LangChain](https://github.com/langchain-ai/langchain) components (e.g., `langchain_core`, `langchain_ollama`, etc.)
- PyPDF2
- python-docx
- PDFPlumber
- ollama
- Other dependencies referenced in the code (consider creating a `requirements.txt`)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Use `venv\Scripts\activate` on Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8080 --reload
   ```
   The API will be available at [http://localhost:8080](http://localhost:8080).

## Usage

After starting the API server, you can interact with the endpoints using tools like [Postman](https://www.postman.com/), [curl](https://curl.se/), or directly via the interactive API docs provided by FastAPI at [http://localhost:8080/docs](http://localhost:8080/docs).

### Endpoints

#### 1. Generate Interview Questions from CV

- **URL**: `/api/generate_cv_questions`
- **Method**: `POST`
- **Description**:  
  Extracts text from a candidate's CV (PDF, DOCX, or TXT) and creates interview questions in French.
- **Parameters**:
  - `cv_file`: File upload of the candidate's CV.
  - `num_questions`: (Optional) Number of questions to generate (default is 5).
- **Response Example**:
  ```json
  {
      "questions": "Question 1: ... \nQuestion 2: ... \n..."
  }
  ```

#### 2. Q&A on Document (RAG)

- **URL**: `/api/qa`
- **Method**: `POST`
- **Description**:  
  Answers questions based on chunked segments of an uploaded PDF document.
- **Parameters**:
  - `question`: The question to ask (provided as a form field).
  - `file`: File upload of the PDF document.
- **Response Example**:
  ```json
  {
      "answer": "The answer to your question based on the document's content..."
  }
  ```

#### 3. CV and Job Matching

- **URL**: `/api/match`
- **Method**: `POST`
- **Description**:  
  Matches candidate CVs against a job description by returning a score and justifying the result based on the CV content.
- **Parameters**:
  - `job_file`: The job description file (PDF, DOCX, or TXT).
  - `cv_files`: One or more candidate CV files.
- **Response Example**:
  ```json
  {
      "results": [
          {
              "filename": "candidate1.pdf",
              "score": 85,
              "reasons": "Le CV démontre une expérience solide en..."
          },
          {
              "filename": "candidate2.pdf",
              "score": 78,
              "reasons": "Le CV présente des compétences pertinentes, mais..."
          }
      ]
  }
  ```

#### 4. Document Summarization

- **URL**: `/api/summarize`
- **Method**: `POST`
- **Description**:  
  Produces a concise summary for an uploaded document.
- **Parameters**:
  - `file`: File upload (PDF, DOCX, or TXT).
- **Response Example**:
  ```json
  {
      "summary": "Le document se concentre sur les points clés suivants..."
  }
  ```

#### 5. HR Document Classification

- **URL**: `/api/classify_file`
- **Method**: `POST`
- **Description**:  
  Classifies HR documents into categories like CV, Lettre de motivation, Offre d'emploi, or Autre document RH.
- **Parameters**:
  - `file`: File upload of the HR document.
- **Response Example**:
  ```json
  {
      "doc_type": "CV",
      "justification": "Le document présente un résumé de compétences et d'expériences..."
  }
  ```

#### 6. OCR Endpoint

- **URL**: `/ocr`
- **Method**: `POST`
- **Description**:  
  Uses the Llama3.2-Vision model via Ollama to extract text from image files or scanned PDFs.
- **Parameters**:
  - `file`: File upload (image or scanned PDF file).
- **Response Example**:
  ```json
  {
      "text": "Texte extrait de l'image..."
  }
  ```

## Configuration

- **Models in Use**:  
  The API leverages various models such as `deepseek-v2`, `deepseek-r1:14b`, and `llama3.2-vision` through Ollama. If needed, you can adjust the model names in their respective functions within the code.

- **Environment Variables**:  
  Set up any required environment variables (e.g., model configuration, API keys for external services) according to your deployment environment.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository.
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Commit your changes** with clear messages.
4. **Push** your branch:
   ```bash
   git push origin feature/my-feature
   ```
5. **Open a Pull Request** detailing your changes.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or further information, please open an issue on GitHub or contact the project maintainer.
