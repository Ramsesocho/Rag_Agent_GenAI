 Agent Conversationnel RAG – Projet GenAI

Ce projet est une implémentation complète d’un agent conversationnel de type RAG (Retrieval-Augmented Generation) développé pour le cours **Generative AI (GenAI)** à Aivancity.

---

##  Objectifs du projet

Ce projet répond aux 6 consignes du devoir :

 | Exigence GenAI | Réponse dans ce projet |
- Choisir une base de connaissances | Des fichiers PDF ou TXT stockés localement (`/data`) |
- Indexation avec un modèle d'embedding | Modèle `all-MiniLM-L6-v2` (HuggingFace) |
- Étape de retrieval | Recherche vectorielle avec **FAISS** |
- Génération avec un LLM | **Mistral 7B**, exécuté en local via **Ollama** |
- Interface conversationnelle | Interface **Streamlit** permettant upload, résumé et Q/R |
 - Évaluation (perf & impact) | Évalué manuellement + architecture **100% locale = impact réduit** |

---

 Accès au projet pour le professeur

### GitHub :  
🔗 [https://github.com/Ramsesocho/Rag_Agent_GenAI](https://github.com/Ramsesocho/Rag_Agent_GenAI)

### Interface en local :

1. Télécharger ou cloner le projet** :
   ```bash
   git clone https://github.com/Ramsesocho/Rag_Agent_GenAI.git
   cd Rag_Agent_GenAI
2. Activer l'environnement virtuel et installer les dépendances :

python -m venv env
source env/bin/activate  # Windows : .\env\Scripts\activate
pip install -r requirements.txt (fichier contenant toutes mes dependances)

3 . Installer et lancer Ollama avec Mistral :

brew install ollama              # si Homebrew est installé sur votre ordi ou terminal sinon installé homebrew avant
ollama pull mistral              # télécharge Mistral 7B
ollama serve                     # démarre Ollama localement

4. Lancer l’interface de l’agent :
streamlit run app.py

Fonctionnalités

  Upload de fichiers PDF ou TXT
  Indexation automatique dans FAISS
  Question/Réponse avec Mistral 7B
  Résumé automatique des documents
  Historique des échanges

Technologies utilisées

  Composant	Technologie
    Embedding	all-MiniLM-L6-v2 (HuggingFace)
    Vector DB	FAISS
    LLM	Mistral 7B via Ollama
    Orchestration	LangChain
    Interface	Streamlit
Évaluation et impact environnemental

  Agent exécuté 100% en local
   Aucune dépendance à OpenAI ou API cloud
   Pas de coût d’inférence externe
   Empreinte carbone réduite (pas d’appels sortants)


   
Arborescence du projet

rag-agent/
├── app.py                  # Interface Streamlit
├── rag_agent.py            # Backend RAG (retrieval + génération)
├── vectorstore/            # Index FAISS persistant
├── data/                   # Fichiers utilisateur uploadés
├── requirements.txt        # Dépendances Python
├── .gitignore              # Ignore .env, vectorstore, __pycache__
└── README.md               # Ce document


 Réalisé par

Ramsès Appesse
 Étudiant IA @ Aivancity Paris-Cachan
 Projet final du module Generative AI


