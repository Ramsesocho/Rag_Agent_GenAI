import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Charger la clé API 
load_dotenv()

def build_vectorstore(data_path="data", db_path="vectorstore"):
    # 1. Charger les fichiers texte
    loader = DirectoryLoader(data_path, loader_cls=TextLoader)
    documents = loader.load()

    # 2. Diviser les documents en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # 3. Générer les embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Créer la base FAISS
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 5. Sauvegarder la base vectorielle
    vectorstore.save_local(db_path)
    print(f"base vectorielle créée et enregistrée dans {db_path}/")

if __name__ == "__main__":
    build_vectorstore()
