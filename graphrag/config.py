"""Configuration du Graph RAG.

Ce projet est SÉPARÉ du RAG vectoriel, mais il en lit les données par chemin
(`VECTOR_RAG_DIR`) : même corpus EUR-Lex, mêmes chunks, mêmes embeddings e5.
Il réutilise aussi son code de génération (même prompt, même LLM, même TOP_K) et son
jeu d'évaluation, pour que la comparaison vectoriel / graphe soit équitable.

Toutes les valeurs peuvent être surchargées dans `.env` ou par variable d'environnement.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Projet RAG vectoriel (source des données partagées) ---
VECTOR_RAG_DIR = Path(os.getenv(
    "VECTOR_RAG_DIR", ROOT.parent / "Advenced-Rag-Agents-main" / "Advenced-Rag-Agents-main")).resolve()

CORPUS_DIR = VECTOR_RAG_DIR / "data" / "eurlex"
ARTICLES_PATH = CORPUS_DIR / "articles.jsonl"
DOCUMENTS_PATH = CORPUS_DIR / "documents.csv"
RELATIONS_PATH = CORPUS_DIR / "relations.csv"
ARTICLE_REFS_PATH = CORPUS_DIR / "article_refs.csv"
CHUNKS_PATH = VECTOR_RAG_DIR / "data" / "index" / "chunks.jsonl"
VECTORSTORE_DIR = VECTOR_RAG_DIR / "app" / "vectorstore_juridique"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "eurlex_numerique")

# --- Données propres au graphe ---
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "eurlex_external"
EXTERNAL_RELATIONS_PATH = EXTERNAL_DIR / "relations_external.csv"   # AMENDS / REPEALS / BASED_ON vers des actes hors corpus
EXTERNAL_DOCUMENTS_PATH = EXTERNAL_DIR / "documents_external.csv"   # titres officiels de ces actes

EURLEX_URL = "https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:{celex}"

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def use_vector_rag() -> None:
    """Rend importable le code du RAG vectoriel (`from app.rag.pipeline import generate_answer`).

    Sa configuration (`app/config.py`) charge son propre `.env` : la clé GROQ_API_KEY
    est donc partagée sans être recopiée ici.
    """
    if not (VECTOR_RAG_DIR / "app" / "rag").is_dir():
        raise RuntimeError(f"Projet RAG vectoriel introuvable : {VECTOR_RAG_DIR} (régler VECTOR_RAG_DIR dans .env)")
    path = str(VECTOR_RAG_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
