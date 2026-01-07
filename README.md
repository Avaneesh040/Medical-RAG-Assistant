# Healthcare RAG System (SQL + Policy Intelligence)

A hybrid healthcare Retrieval-Augmented Generation (RAG) system that answers user queries by combining structured patient data (SQL) with unstructured healthcare policy documents using LLMs, vector search, and graph-based orchestration.

## Overview

This project implements an intelligent query-routing pipeline for healthcare data. User queries are first interpreted as structured requests and translated into optimized SQL queries executed on a SQLite patient database. If structured data is insufficient, the system retrieves relevant information from healthcare policy documents using semantic search over a vector database. For open-ended queries, the system falls back to LLM-based reasoning.

The control flow is managed using LangGraph, enabling conditional execution between SQL querying, policy retrieval, and language model reasoning while maintaining conversational memory across turns.

## Repository Structure


healthcare-rag/
│
├── rag_agent.py        # LangGraph-based chatbot with SQL + RAG routing
├── build_vectordb.py   # Script to build vector DB from PDFs and SQL DB from CSV
└── README.md
Generated artifacts such as vector databases, SQLite databases, embeddings, and document files are intentionally not included and must be created locally.
```text
How It Works
Converts natural language queries into SQL when structured data is required

Executes SQL queries on a patient information database

Retrieves relevant healthcare policy content using Chroma-based semantic search

Uses LLM reasoning as a fallback for open-ended queries

Maintains multi-turn conversational context using graph-based memory

Setup
Install Dependencies
bash
Copy code
pip install langchain langgraph chromadb sentence-transformers groq python-dotenv
Environment Variables
Create a .env file in the project root:

env
Copy code
GROQ_API_KEY=your_api_key_here
Data Preparation
Run the following script to prepare all required data sources:

bash
Copy code
python build_vectordb.py
This script performs the following:

Loads healthcare policy PDFs

Splits and embeds text using HuggingFace MiniLM

Creates a persistent Chroma vector database

Builds a SQLite patient database from CSV data

Running the System
bash
Copy code
python rag_agent.py
Enter queries in natural language.
Type exit to terminate the session.

Use Cases
Healthcare policy question answering

Patient data lookup and analysis

Decision-support system prototyping

Hybrid SQL + RAG experimentation

Disclaimer
This project is intended for educational and research purposes only and must not be used for real clinical decision-making. 
