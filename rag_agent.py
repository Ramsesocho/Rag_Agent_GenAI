import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Charger les variables d’environnement
load_dotenv()

# LLM local (Ollama + mistral)
llm = Ollama(model="mistral")

def ask_agent(question: str):
    # Recharger la base FAISS actualisée
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)

    # Créer le retriever et QA chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Créer un outil utilisable par l’agent
    tools = [
        Tool(
            name="RechercheDoc",
            func=qa_chain.invoke,
            description="Répond à une question à partir des documents vectorisés."
        )
    ]

    # Initialiser l’agent à chaque appel
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent_executor.invoke({"input": question})
