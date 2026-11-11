"""Interface Streamlit commune : tous les modes du RAG vectoriel et du Graph RAG sur la même page.

L'application du RAG vectoriel n'est pas modifiée : ses fonctions sont importées et s'exécutent ici (pas d'API à lancer) ;
Neo4j doit être démarré. Les questions posées ici ont leur propre cache de candidats et leur propre base de conversations
de l'agent (`data/cache/`), pour ne rien écrire dans le projet vectoriel.

Chaque mode RAG utilise le même prompt, le même LLM et 5 articles distincts, envoyés en texte complet au LLM. Pour chaque
source : lien EUR-Lex, extrait et, en mode graphe, chemin qui l'a fait entrer dans le contexte. L'agent choisit lui-même
ses outils et son nombre d'extraits.

Le mode « Graph RAG » est `graph_linked` : meilleur Recall@5 des deux variantes sur les 60 questions d'évaluation.

Lancement (depuis ce dossier) :
    python -m streamlit run streamlit_app.py --server.port 8502 --server.headless true
"""
import os
import time
import uuid
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SEED_CACHE_PATH", str(CACHE_DIR / "streamlit_candidates.json"))
os.environ.setdefault("AGENT_DB_PATH", str(CACHE_DIR / "streamlit_agent.sqlite"))

import streamlit as st  # noqa: E402

from graphrag import config  # noqa: E402
from graphrag import retriever as graph  # noqa: E402
from graphrag.db import run  # noqa: E402

config.use_vector_rag()
from app.rag.pipeline import generate_answer, retrieve  # noqa: E402

K = 5
BASELINE, COMPARED_GRAPH = "advanced", "graph_linked"
VECTOR_MODES = ("vector", "hybrid", "advanced")
MODES = {
    "RAG vectoriel (similarité seule)": "vector",
    "RAG hybride (vectoriel + BM25)": "hybrid",
    "RAG avancé (MultiQuery + HyDE + BM25 + reranker) — baseline": "advanced",
    "Agent (outils juridiques)": "agent",
    "Graph RAG": "graph_linked",
    "Comparer : baseline et Graph RAG côte à côte": "compare",
}
MODE_NAMES = {v: k for k, v in MODES.items()}

st.set_page_config(page_title="RAG juridique", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=120, show_spinner=False)
def graph_stats() -> dict:
    return run("""MATCH (a:Article) WITH count(a) AS articles
                  MATCH (d:Document {in_corpus: true}) WITH articles, count(d) AS actes
                  MATCH (:Article)-[r:REFERS_TO]->(:Article) WITH articles, actes, count(r) AS renvois
                  OPTIONAL MATCH (c:Cluster) RETURN articles, actes, renvois, count(c) AS clusters""")[0]


@st.cache_resource(show_spinner=False)
def get_agent():
    from app.agent.agent import build_agent

    return build_agent()


def to_sources(chunks: list) -> list[dict]:
    sources = []
    for i, c in enumerate(chunks, start=1):
        m = c.metadata
        body = c.page_content.split("\n", 1)[-1].split(f"\n{graph.GRAPH_INFO_PREFIX}")[0]
        paths = [p for p in m.get("via", []) if " est voisin de " not in p and "cité dans la question" not in p]
        sources.append({"ref": i, "acte": m.get("doc_label"), "article": m.get("article_num"), "titre": m.get("article_title"),
                        "url": config.EURLEX_URL.format(celex=m.get("celex")), "extrait": body[:320],
                        "ajoute": m.get("graph_added", False), "chemins": paths[:3]})
    return sources


def answer(question: str, mode: str, agent_thread: str) -> dict:
    start = time.time()
    if mode == "agent":
        from app.agent.agent import run_agent
        from app.agent.tools import get_last_chunks

        text = run_agent(get_agent(), question, session_id=agent_thread)
        chunks = [c for c in get_last_chunks() if c.metadata.get("article_id")]
    else:
        if mode in VECTOR_MODES:
            chunks = retrieve(question, mode=mode, k=K)
        else:
            chunks = graph.graph_retrieve(question, K, mode=mode)
        text = generate_answer(question, graph.full_article_context(chunks))
    return {"mode": mode, "answer": text, "sources": to_sources(chunks), "seconds": time.time() - start}


def render_result(result: dict) -> None:
    st.write(result["answer"])
    st.caption(f"⏱ {result['seconds']:.1f} s · {MODE_NAMES[result['mode']]}")
    with st.expander(f"📚 {len(result['sources'])} sources"):
        for s in result["sources"]:
            with st.container(border=True):
                tag = " · 🔗 ajouté par le graphe" if s["ajoute"] else ""
                title = f" — {s['titre']}" if s["titre"] else ""
                st.markdown(f"**[{s['ref']}] {s['acte']} — Article {s['article']}**{title}{tag} · [EUR-Lex]({s['url']})")
                st.caption(f"{s['extrait']}…")
                for path in s["chemins"]:
                    st.markdown(f"↳ {path}")


def render_turn(message: dict) -> None:
    if message.get("error"):
        st.error(message["error"])
        return
    results = message["results"]
    if len(results) == 1:
        render_result(results[0])
        return
    for column, result in zip(st.columns(2), results):
        with column:
            st.markdown(f"#### {MODE_NAMES[result['mode']]}")
            render_result(result)


def submit_question() -> None:
    """Le calcul démarre au passage suivant : les commandes sont alors bloquées et le mode ne peut plus l'interrompre."""
    question = st.session_state.get("question_input")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.pending = {"question": question, "mode": MODES[st.session_state.mode_label]}


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("agent_thread", uuid.uuid4().hex)
st.session_state.setdefault("mode_label", next(iter(MODES)))
busy = st.session_state.pending is not None

with st.sidebar:
    st.markdown("## ⚖️ RAG juridique")
    st.caption("Questions-réponses sur 100 actes européens : RAG vectoriel et Graph RAG sur un graphe de connaissances Neo4j.")
    try:
        stats = graph_stats()
    except Exception as e:
        st.error(f"Neo4j injoignable : démarrer la base dans Neo4j Desktop. ({type(e).__name__})")
        st.stop()
    c1, c2 = st.columns(2)
    c1.metric("Actes", stats["actes"])
    c2.metric("Articles", stats["articles"])
    c1.metric("Renvois", stats["renvois"])
    c2.metric("Clusters", stats["clusters"])
    st.divider()
    st.radio("Mode", list(MODES), key="mode_label", disabled=busy)
    st.caption(f"{K} articles distincts dans le contexte pour chaque mode RAG, même prompt et même LLM.")
    st.caption("Sur CPU, compter de quelques secondes (vectoriel) à environ une minute (graphe) par question.")
    if st.button("🗑 Effacer la conversation", use_container_width=True, disabled=busy):
        st.session_state.messages = []
        st.session_state.agent_thread = uuid.uuid4().hex
        st.rerun()

st.markdown("## RAG juridique — droit européen des données et du numérique")
st.caption("Réponses générées à partir des textes officiels, avec citation des articles. Pas un conseil juridique.")
st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_turn(message)

st.chat_input("Ex. : Comment le DSA modifie-t-il la directive 2000/31/CE ?", key="question_input",
              on_submit=submit_question, disabled=busy)

if busy:
    pending = st.session_state.pending
    with st.chat_message("assistant"):
        with st.spinner(f"Calcul en cours : {MODE_NAMES[pending['mode']]}…"):
            modes = (BASELINE, COMPARED_GRAPH) if pending["mode"] == "compare" else (pending["mode"],)
            try:
                reply = {"role": "assistant", "results": [answer(pending["question"], m, st.session_state.agent_thread) for m in modes]}
            except Exception as e:
                reply = {"role": "assistant", "error": f"Erreur ({MODE_NAMES[pending['mode']]}) : {type(e).__name__} : {e}"}
    st.session_state.messages.append(reply)
    st.session_state.pending = None
    st.rerun()
