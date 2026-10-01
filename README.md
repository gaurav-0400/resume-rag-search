# Resume RAG Search

## Overview

Resume RAG Search is a Python-based web application that demonstrates a Retrieval-Augmented Generation (RAG) workflow for extracting, retrieving, and generating answers from resume documents. It combines text retrieval techniques with a Large Language Model (LLM) to provide context-aware responses to user queries.

The application uses Flask for the web interface, TF-IDF and cosine similarity for information retrieval, and the Groq API for natural-language answer generation.

## Features

* Extract text from PDF documents using PyMuPDF.
* Split extracted text into manageable chunks.
* Retrieve relevant information using TF-IDF vectorization and cosine similarity.
* Generate context-aware answers using the Groq API.
* Provide an interactive web interface using Flask.
* Deploy the application on Render.

## Technology Stack

* **Language:** Python
* **Web Framework:** Flask
* **PDF Processing:** PyMuPDF
* **Text Retrieval:** Scikit-learn, TF-IDF, Cosine Similarity
* **LLM Integration:** Groq API
* **Environment Configuration:** python-dotenv
* **Deployment:** Render
* **Production Server:** Gunicorn

## Project Architecture

```text
resume-rag-search/
│
├── app.py                    # Flask application and routes
├── ingest.py                 # PDF text extraction and chunking
├── query.py                  # Retrieval and answer generation
├── resume.pdf                # Input resume document
├── requirements.txt          # Project dependencies
├── .env                      # Environment configuration
│
├── vector_store/
│   └── chunks.pkl             # Stored text chunks
│
├── templates/
│   └── index.html             # Web interface
│
└── static/
    └── style.css              # Application styling
```

## Application Workflow

The application follows a retrieval-augmented generation workflow:

1. **Document Ingestion:** The resume PDF is processed using PyMuPDF to extract text.
2. **Text Chunking:** The extracted text is divided into smaller, manageable chunks.
3. **Data Storage:** The generated chunks are serialized and stored using Pickle.
4. **Query Processing:** The user submits a question through the Flask web interface.
5. **Information Retrieval:** TF-IDF vectorization and cosine similarity identify the most relevant text chunks.
6. **Context Preparation:** The retrieved chunks are combined to create context for the language model.
7. **Answer Generation:** The Groq API generates a natural-language response based on the retrieved context.
8. **Response Display:** Flask returns the generated answer to the web interface.

## Development Approach

The project initially explored local language models using Hugging Face Transformers and Qwen models, along with embedding-based retrieval using Sentence Transformers and FAISS.

To reduce memory consumption and simplify deployment, the current retrieval implementation uses TF-IDF and cosine similarity. Groq API integration handles natural-language answer generation, while Flask provides the application interface.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/gaurav-0400/resume-rag-search.git
cd resume-rag-search
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory and configure the Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

The application loads this variable through `python-dotenv` and uses it to authenticate requests to the Groq API.

### 5. Prepare the Resume Data

Place the resume PDF at the path expected by `ingest.py`, then run:

```bash
python ingest.py
```

This processes the PDF, extracts the text, creates chunks, and generates the data required by the retrieval component.

### 6. Run the Application

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000/
```

Enter a question about the resume to retrieve relevant information and generate an answer.

## Deployment

The application can be deployed on Render using Gunicorn as the production server.

### Deployment Configuration

* **Build Command:**

  ```bash
  pip install -r requirements.txt
  ```

* **Start Command:**

  ```bash
  gunicorn app:app
  ```

Configure the required environment variables and make the processed resume data available to the application in the deployment environment.

## Project Status

The project implements PDF text extraction, chunk-based data preparation, lightweight text retrieval, LLM-based answer generation, and a Flask web interface. It also supports deployment using Render.

## Future Improvements

* Support multiple document formats.
* Add document upload functionality.
* Improve retrieval accuracy using semantic embeddings.
* Add conversation history and follow-up questions.
* Introduce document-level metadata and source references.
* Enhance error handling and input validation.

Author

Gaurav

GitHub: gaurav-0400

Project Repository: [Resume RAG Search](https://github.com/gaurav-0400/resume-rag-search)
