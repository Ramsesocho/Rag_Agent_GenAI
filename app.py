import os
import streamlit as st
from rag_agent import ask_agent
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration de la page
st.set_page_config(
    page_title="🧠 Agent RAG Local",
    page_icon="🔍",
    layout="centered",
)

st.title("🤖 Agent RAG - Mistral local (Ollama)")
st.markdown("Pose une question sur tes documents CANAL PLUS TELECOM vectorisés avec **Mistral** exécuté localement via Ollama.")

# Initialiser l’historique
if "history" not in st.session_state:
    st.session_state.history = []

# 📁 Upload de fichier
st.subheader("📂 Importer un document")
uploaded_file = st.file_uploader("Uploader un fichier .pdf ou .txt", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("📚 Lecture du document..."):
        # Sauvegarde dans /data
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Loader adapté
        loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])

        st.success("✅ Document prêt à être vectorisé ou résumé.")

        # 📄 Afficher le contenu brut en option
        with st.expander("📝 Voir le contenu extrait"):
            st.text_area("Contenu du document :", full_text, height=200)

        # ➕ Bouton de résumé
        if st.button("🧠 Résumer ce document"):
            with st.spinner("Résumé en cours..."):
                from langchain_community.llms import Ollama
                llm = Ollama(model="mistral")
                prompt = f"Voici un document :\n{full_text}\n\nFais-moi un résumé clair et structuré en français."
                try:
                    summary = llm.invoke(prompt)
                    st.success("✅ Résumé généré :")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Erreur lors du résumé : {e}")

        # ➕ Bouton vectorisation
        if st.button("📡 Ajouter ce document à la base de connaissances"):
            with st.spinner("Découpage & vectorisation..."):
                from langchain.text_splitter import CharacterTextSplitter
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings

                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)

                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
                vectorstore.add_documents(chunks)
                vectorstore.save_local("vectorstore")

                st.success("✅ Document vectorisé et ajouté avec succès !")

# 💬 Question utilisateur
question = st.text_input("💬 Pose ta question :", placeholder="Ex: Quels sont les avantages de la fibre optique ?")

# Envoi de la question
if st.button("🧠 Interroger l'agent") and question.strip():
    with st.spinner("Réflexion en cours..."):
        try:
            response = ask_agent(question)
            st.session_state.history.append({"question": question, "response": response})
            st.success("✅ Réponse générée !")
        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")

# 🗂️ Historique des échanges
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Historique des interactions")
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}. Question :** {entry['question']}")
        st.markdown(f"🧠 **Réponse :** {entry['response']}")
        st.markdown("---")
