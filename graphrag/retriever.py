"""Étape 3 : recherche « graph » — mêmes graines que le mode advanced, plus une expansion à 1 saut dans Neo4j.

Chaîne (la seule différence avec le mode advanced est l'étape 2) :
    1. graines   : les 20 candidats du mode advanced, relus dans son cache (ordre RRF) ;
    2. graphe    : a) entités de la question : articles cités (« article 17 du RGPD ») et, si la question demande
                      une relation (abroge, modifie), articles qui l'opèrent ;
                   b) expansion à 1 saut depuis les 10 premiers articles graines, au plus 5 voisins par graine :
                      REFERS_TO sortants, REFERS_TO entrants, articles qui modifient ou abrogent l'acte de la graine, NEXT ;
                   chaque voisin entre dans le pool avec UN chunk, le plus proche de la question (cosinus e5) ;
    3. sélection : pool = candidats + voisins ; même reranker que advanced, puis `top_articles` (k articles distincts).

Modes
    graph            : reranker sur le pool complet
    graph_norerank   : sans reranker, fusion RRF (candidats, voisins) — pour isoler l'apport du graphe du reranker
    graph_linked     : même classement que graph, puis les 2 dernières places vont aux articles d'AUTRES actes vers
                       lesquels renvoie l'article classé 1er (le reranker ne les départage qu'entre eux) ; sans tel
                       renvoi, le résultat est celui de graph. Motif : le reranker note la ressemblance avec la
                       question, et un article lié par un renvoi ne ressemble pas forcément à la question.

Le contexte reste de k articles. Le graphe y ajoute seulement une ligne « Information du graphe » par extrait : renvois
suivis et titre officiel des actes modifiés ou abrogés. Ce surcoût est mesuré par `context_tokens_approx` à l'évaluation.

`full_article_context` remplace, après la sélection, le chunk de chaque article par son texte complet (borné). Sans cela,
le LLM peut recevoir le bon article sans le paragraphe qui répond : un article est découpé en 4 chunks en moyenne et
seul le mieux classé serait transmis. À appliquer à tous les modes comparés.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document

from graphrag import config
from graphrag.acts import question_acts, question_articles, relation_intents
from graphrag.db import run

config.use_vector_rag()
from app import config as rag_config  # noqa: E402
from app.dependencies import get_vectordb  # noqa: E402
from app.rag.corpus import get_corpus  # noqa: E402
from app.rag.embeddings import get_embeddings  # noqa: E402
from app.rag.hybrid_search import reciprocal_rank_fusion  # noqa: E402
from app.rag.pipeline import top_articles  # noqa: E402
from app.rag.reranker import rerank  # noqa: E402
from app.rag.retriever import advanced_retrieval  # noqa: E402

MODES = ("graph", "graph_norerank", "graph_linked")
N_SEEDS = 10
MAX_NEIGHBOURS = 5
MAX_RELATION_ARTICLES = 5
MAX_PATHS_IN_CONTEXT = 2
MAX_FACTS_IN_CONTEXT = 3
LINKED_SLOTS = 2
ARTICLE_CONTEXT_CHARS = 6000        # 95 % des articles tiennent entièrement dans cette limite
GRAPH_INFO_PREFIX = "Information du graphe (pas un extrait du texte) :"

VERBS = {"AMENDS": "modifie", "REPEALS": "abroge"}

_RETURN = "RETURN id AS seed, n.id AS article, n.chunk_ids AS chunk_ids"
EXPANSIONS = {
    "REFERS_OUT": f"UNWIND $ids AS id MATCH (:Article {{id: id}})-[:REFERS_TO]->(n:Article) {_RETURN}, null AS verb, null AS act",
    "REFERS_IN": f"UNWIND $ids AS id MATCH (:Article {{id: id}})<-[:REFERS_TO]-(n:Article) {_RETURN}, null AS verb, null AS act",
    "ACT_CHANGE": f"""UNWIND $ids AS id MATCH (:Article {{id: id}})<-[:HAS_ARTICLE]-(d:Document)<-[r:AMENDS|REPEALS]-(n:Article)
                      {_RETURN}, type(r) AS verb, d.short_name AS act""",
    "NEXT": f"UNWIND $ids AS id MATCH (:Article {{id: id}})-[:NEXT]-(n:Article) {_RETURN}, null AS verb, null AS act",
}

# Une relation entre deux actes nommés doit les relier tous les deux ; avec un seul acte, il est source ou cible.
RELATION_QUERY = """MATCH (n:Article)-[r]->(d:Document)
    WHERE type(r) IN $intents AND (n.celex IN $acts OR d.celex IN $acts)
      AND (size($acts) < 2 OR (n.celex IN $acts AND d.celex IN $acts))
    RETURN n.id AS article, n.chunk_ids AS chunk_ids, type(r) AS verb, d.short_name AS act"""

ACT_FACTS_QUERY = """MATCH (a:Article)-[r:AMENDS|REPEALS]->(d:Document) WHERE a.id IN $ids
    RETURN a.id AS article, type(r) AS verb, d.title AS title ORDER BY a.id, d.celex"""

LINKED_QUERY = """MATCH (:Article {id: $id})-[:REFERS_TO]->(n:Article) WHERE n.celex <> $celex
    RETURN n.id AS article, n.chunk_ids AS chunk_ids"""

# Le même pool est classé par graph et graph_linked : un seul passage du reranker par question.
_rerank_cache: dict[tuple, list[Document]] = {}


@dataclass
class Edge:
    """Un article atteint par le graphe, et comment."""
    article: str
    chunk_ids: list[str]
    rel: str
    text: str
    seed: str | None = None


def article_label(article_id: str) -> str:
    corpus = get_corpus()
    art = corpus.articles[article_id]
    return f"{corpus.doc_label(art.celex)}, article {art.num}"


def _describe(rel: str, seed: str, article: str, verb: str | None, act: str | None) -> str:
    if rel == "REFERS_OUT":
        return f"{article_label(seed)} renvoie à {article_label(article)}"
    if rel == "REFERS_IN":
        return f"{article_label(article)} renvoie à {article_label(seed)}"
    if rel == "ACT_CHANGE":
        return f"{article_label(article)} {VERBS[verb]} {act}"
    return f"{article_label(article)} est voisin de {article_label(seed)} dans l'acte"


def expansion_edges(seeds: list[str]) -> list[Edge]:
    edges = []
    for rel, query in EXPANSIONS.items():
        for r in run(query, ids=seeds):
            if r["article"] != r["seed"] and r["chunk_ids"]:
                edges.append(Edge(r["article"], r["chunk_ids"], rel,
                                  _describe(rel, r["seed"], r["article"], r["verb"], r["act"]), r["seed"]))
    return edges


def entity_edges(acts: list[str], intents: set[str], explicit: list[str]) -> list[Edge]:
    """Articles désignés par la question elle-même : cités, ou opérant la relation demandée."""
    edges = []
    if explicit:
        for r in run("UNWIND $ids AS id MATCH (n:Article {id: id}) RETURN n.id AS article, n.chunk_ids AS chunk_ids", ids=explicit):
            edges.append(Edge(r["article"], r["chunk_ids"], "QUESTION", f"{article_label(r['article'])} est cité dans la question"))
    if acts and intents:
        for r in run(RELATION_QUERY, acts=acts, intents=sorted(intents)):
            if r["chunk_ids"]:
                edges.append(Edge(r["article"], r["chunk_ids"], "RELATION",
                                  f"{article_label(r['article'])} {VERBS[r['verb']]} {r['act']}"))
    return edges


def best_chunks(question: str, chunk_ids_by_article: dict[str, list[str]]) -> dict[str, tuple[Document, float]]:
    """Pour chaque article : son chunk le plus proche de la question (Document, cosinus), lu dans Chroma."""
    ids = [cid for cids in chunk_ids_by_article.values() for cid in cids]
    if not ids:
        return {}
    res = get_vectordb().get(ids=ids, include=["embeddings", "documents", "metadatas"])
    query = np.asarray(get_embeddings().embed_query(question))
    similarities = np.asarray(res["embeddings"]) @ query       # vecteurs e5 normalisés : produit scalaire = cosinus
    best: dict[str, tuple[Document, float]] = {}
    for text, meta, sim in zip(res["documents"], res["metadatas"], similarities.tolist()):
        aid = meta["article_id"]
        if aid not in best or sim > best[aid][1]:
            best[aid] = (Document(page_content=text, metadata=meta), sim)
    return best


def select_neighbours(seeds: list[str], edges: list[Edge], chunks: dict, known: set[str], intents: set[str]) -> set[str]:
    """Au plus MAX_NEIGHBOURS nouveaux articles par graine : par type de lien, puis par proximité avec la question."""
    order = ["REFERS_OUT", "REFERS_IN", "ACT_CHANGE", "NEXT"]
    if intents & set(VERBS):                  # la question porte sur une modification / abrogation
        order = ["ACT_CHANGE", "REFERS_OUT", "REFERS_IN", "NEXT"]
    by_seed = defaultdict(list)
    for e in edges:
        if e.article in chunks and e.article not in known:
            by_seed[e.seed].append(e)
    selected: set[str] = set()
    for seed in seeds:
        picked: list[str] = []
        for e in sorted(by_seed[seed], key=lambda x: (order.index(x.rel), -chunks[x.article][1])):
            if e.article not in picked:
                picked.append(e.article)
            if len(picked) == MAX_NEIGHBOURS:
                break
        selected.update(picked)
    return selected


def annotate(chunks: list[Document], edges_by_article: dict[str, list[Edge]], added: set[str]) -> list[Document]:
    """Copies des chunks retenus, avec la ligne « Information du graphe » (les originaux, en cache, restent intacts)."""
    facts = defaultdict(list)
    for r in run(ACT_FACTS_QUERY, ids=[c.metadata["article_id"] for c in chunks]):
        facts[r["article"]].append(f"{article_label(r['article'])} {VERBS[r['verb']]} : {r['title']}")
    out = []
    for c in chunks:
        aid = c.metadata["article_id"]
        edges = edges_by_article.get(aid, [])
        paths = list(dict.fromkeys(e.text for e in edges if e.rel in ("REFERS_OUT", "REFERS_IN")))[:MAX_PATHS_IN_CONTEXT]
        lines = paths + facts[aid][:MAX_FACTS_IN_CONTEXT]
        text = f"{c.page_content}\n{GRAPH_INFO_PREFIX} {' ; '.join(lines)}" if lines else c.page_content
        out.append(Document(page_content=text, metadata={
            **c.metadata, "via": list(dict.fromkeys(e.text for e in edges)), "graph_added": aid in added}))
    return out


def full_article_context(chunks: list[Document], cap: int = ARTICLE_CONTEXT_CHARS) -> list[Document]:
    """Copies des chunks sélectionnés avec le texte complet de leur article (la sélection ne change pas).

    Article plus long que `cap` : début de l'article, puis le passage sélectionné s'il n'y figure pas déjà.
    """
    corpus = get_corpus()
    out = []
    for c in chunks:
        header, _, rest = c.page_content.partition("\n")
        body, sep, info = rest.partition(f"\n{GRAPH_INFO_PREFIX}")
        article = corpus.articles.get(c.metadata.get("article_id"))
        text = article.text if article else body
        if len(text) > cap:
            kept = text[:cap]
            text = f"{kept}\n[…]" if body[:80] in kept else f"{text[:max(cap - len(body), 0)]}\n[…]\n{body}"
        out.append(Document(page_content=f"{header}\n{text}{sep}{info}", metadata=c.metadata))
    return out


def rerank_pool(question: str, pool: list[Document]) -> list[Document]:
    key = (question, tuple(c.metadata["chunk_id"] for c in pool))
    if key not in _rerank_cache:
        _rerank_cache[key] = rerank(question, pool, top_k=None)
    return _rerank_cache[key]


def with_linked_slots(question: str, ranked: list[Document], k: int) -> tuple[list[Document], list[Edge]]:
    """Top k du classement, dont les LINKED_SLOTS dernières places cédées aux renvois inter-actes de l'article 1er."""
    base = top_articles(ranked, k)
    if len(base) <= LINKED_SLOTS:
        return base, []
    head = base[0].metadata
    rows = [r for r in run(LINKED_QUERY, id=head["article_id"], celex=head["celex"]) if r["chunk_ids"]]
    keep = base[:k - LINKED_SLOTS]
    kept_ids = {c.metadata["article_id"] for c in keep}
    chunks = best_chunks(question, {r["article"]: r["chunk_ids"] for r in rows if r["article"] not in kept_ids})
    if not chunks:
        return base, []
    linked = rerank(question, [chunks[a][0] for a in sorted(chunks)], top_k=LINKED_SLOTS)
    edges = [Edge(c.metadata["article_id"], [c.metadata["chunk_id"]], "REFERS_OUT",
                  _describe("REFERS_OUT", head["article_id"], c.metadata["article_id"], None, None), head["article_id"])
             for c in linked]
    return top_articles(keep + linked + ranked, k), edges


def graph_retrieve(question: str, k: int = rag_config.TOP_K, mode: str = "graph") -> list[Document]:
    """k chunks issus de k articles distincts, comme `pipeline.retrieve`."""
    if mode not in MODES:
        raise ValueError(f"mode inconnu : {mode!r} (attendu : {MODES})")
    candidates = advanced_retrieval(question, get_vectordb())
    known = {c.metadata["article_id"] for c in candidates}

    acts = question_acts(question)
    intents = relation_intents(question)
    explicit = question_articles(question, acts)
    seeds = list(dict.fromkeys(explicit + [c.metadata["article_id"] for c in candidates]))[:N_SEEDS]

    forced = entity_edges(list(acts), intents, explicit)
    expansion = expansion_edges(seeds)
    chunks = best_chunks(question, {e.article: e.chunk_ids for e in forced + expansion if e.article not in known})

    forced_ids = {e.article for e in forced if e.article in chunks}
    if len(forced_ids) > MAX_RELATION_ARTICLES:
        keep = set(explicit) | set(sorted(forced_ids - set(explicit), key=lambda a: -chunks[a][1])[:MAX_RELATION_ARTICLES])
        forced_ids &= keep
    added = forced_ids | select_neighbours(seeds, expansion, chunks, known, intents)

    # articles imposés par la question d'abord, puis voisins par proximité (ordre utilisé seulement sans reranker)
    new_docs = [chunks[a][0] for a in sorted(added, key=lambda a: (a not in forced_ids, -chunks[a][1], a))]
    if mode == "graph_norerank":
        ranked = reciprocal_rank_fusion([candidates, new_docs])
    else:
        ranked = rerank_pool(question, candidates + new_docs)

    linked_edges: list[Edge] = []
    if mode == "graph_linked":
        selected, linked_edges = with_linked_slots(question, ranked, k)
        added |= {e.article for e in linked_edges if e.article not in known}
    else:
        selected = top_articles(ranked, k)

    edges_by_article = defaultdict(list)
    for e in linked_edges + forced + expansion:      # le lien qui justifie une place réservée s'affiche en premier
        edges_by_article[e.article].append(e)
    return annotate(selected, edges_by_article, added)
