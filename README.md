# Resume RAG Search

## Overview

Resume RAG Search is a Python project that demonstrates the basic workflow of Retrieval-Augmented Generation (RAG) using a resume PDF.

## Features

* Extract text from PDF files using PyMuPDF.
* Split extracted text into smaller chunks.
* Generate text embeddings using Sentence Transformers.
* Store and search vectors using FAISS.
* Retrieve relevant resume information based on user questions.

## Tech Stack

* Python
* PyMuPDF
* Sentence Transformers
* NumPy
* FAISS
* Pickle

## Project Status

The PDF ingestion, embedding generation, vector storage, and semantic retrieval steps are implemented. Connecting a Large Language Model (LLM) to generate natural-language answers is the next planned step.
