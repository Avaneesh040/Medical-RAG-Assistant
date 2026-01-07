import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import sqlite3
load_dotenv()

PDF_PATH = r"C:\Users\KIIT\Desktop\healthcare\CUSTOMER POLICY.pdf"
VECTORDB_DIR = "./vectordb/healthcare_vectordb"

# Check if vector DB already exists
if not os.path.exists(VECTORDB_DIR):
    os.makedirs(VECTORDB_DIR)
    loader = PyPDFLoader(PDF_PATH)
    docs_list = loader.load_and_split()
    # Split text into chunks of 1000 characters with 100-character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    # Load pre-trained embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # Change "cpu" to "cuda" if using GPU
        encode_kwargs={"normalize_embeddings": True}  # Ensures normalized embeddings
    )
    # Create and persist Chroma Vector Database
    vectordb = Chroma.from_documents(
        documents=doc_splits,
        collection_name="healthcare_rag_chroma",
        embedding=embeddings,
        persist_directory=VECTORDB_DIR
    )
    print("Healthcare VectorDB created and saved.")
else:
    print("Healthcare VectorDB already exists.")

#SQl database from csv file
db_path = r"C:\Users\KIIT\Desktop\healthcare\healthcare_data.db"
csv_path = r"C:\Users\KIIT\Desktop\healthcare\healtcare_db.csv"
table_name = "patient_info"
df = pd.read_csv(csv_path)
conn = sqlite3.connect(db_path)
df.to_sql(table_name, conn, if_exists='replace', index=False)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables in database after creation:", cursor.fetchall())
conn.commit()
conn.close()