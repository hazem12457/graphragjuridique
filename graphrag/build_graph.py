"""Étape 1 : construction du graphe de connaissances dans Neo4j, à partir des MÊMES données que le RAG vectoriel.

Nœuds
    (:Document {celex, title, short_name, label, type, date, year, in_corpus, url})   100 du corpus + actes hors corpus
    (:Article  {id, celex, num, title, text, position, n_chunks, chunk_ids, embedding})
    (:EuroVoc  {label})
Arêtes
    (:Document)-[:HAS_SUBJECT]->(:EuroVoc)
    (:Document)-[:HAS_ARTICLE]->(:Article)
    (:Article)-[:NEXT]->(:Article)                         articles consécutifs du même acte
    (:Document)-[:CITES|AMENDS|REPEALS|BASED_ON]->(:Document)   Cellar (relations.csv + relations_external.csv)
    (:Article)-[:REFERS_TO]->(:Article)                    renvois du texte (target_in_corpus = True)
    (:Article)-[:AMENDS|REPEALS {evidence}]->(:Document)   l'article qui opère la modification / l'abrogation
    (:Article)-[:MENTIONS]->(:Document)                    actes cités dans le texte de l'article
Index
    contraintes d'unicité sur Document.celex, Article.id, EuroVoc.label
    index vectoriel `article_embedding` (768 dimensions, cosinus) : embedding d'un article =
    moyenne normalisée des embeddings e5 de ses chunks, repris de Chroma (rien n'est recalculé ;
    réencoder l'article entier le tronquerait à 512 tokens).

Usage
    python -m graphrag.build_graph --dry-run      # prépare et compte, sans Neo4j
    python -m graphrag.build_graph                # vide la base puis charge tout
"""
import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np
from loguru import logger

from graphrag import config
from graphrag.acts import article_relations, mentioned_acts, short_act_name_safe

config.use_vector_rag()
from app.rag.corpus import get_corpus  # noqa: E402

EMBEDDING_DIM = 768


# ----------------------------------------------------------------------------------------- préparation
def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_chunk_ids() -> dict[str, list[str]]:
    """article_id -> chunk_ids dans l'ordre (chunks.jsonl du RAG vectoriel)."""
    out = defaultdict(list)
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)["metadata"]
            out[m["article_id"]].append(m["chunk_id"])
    return out


def load_article_embeddings(chunk_ids_by_article: dict[str, list[str]]) -> dict[str, list[float]]:
    """Moyenne normalisée des embeddings e5 des chunks de chaque article, lus dans Chroma."""
    import chromadb

    col = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR)).get_collection(config.COLLECTION_NAME)
    vectors: dict[str, np.ndarray] = {}
    offset, page = 0, 2000
    while True:
        res = col.get(include=["embeddings"], limit=page, offset=offset)
        if not len(res["ids"]):
            break
        for cid, emb in zip(res["ids"], res["embeddings"]):
            vectors[cid] = np.asarray(emb, dtype=np.float32)
        offset += page
    logger.info(f"{len(vectors)} embeddings de chunks lus dans Chroma")

    out = {}
    for aid, cids in chunk_ids_by_article.items():
        vs = [vectors[c] for c in cids if c in vectors]
        if vs:
            v = np.mean(vs, axis=0)
            out[aid] = (v / (np.linalg.norm(v) or 1.0)).round(6).tolist()
    return out


def prepare(with_embeddings: bool = True) -> dict:
    corpus = get_corpus()
    data: dict[str, list] = {}

    # --- documents du corpus + hors corpus
    docs = [{
        "celex": d.celex, "title": d.title, "short_name": d.short_name, "label": d.label,
        "type": d.doc_type, "date": d.date, "year": d.year, "in_corpus": True, "url": d.url,
        "aliases": d.aliases,
    } for d in corpus.documents.values()]
    for r in read_csv(config.EXTERNAL_DOCUMENTS_PATH):
        if r["celex"] not in corpus.documents:
            short = short_act_name_safe(r["title"]) or r["celex"]
            docs.append({"celex": r["celex"], "title": r["title"], "short_name": short, "label": short,
                         "type": r["type"], "date": r["date"], "year": int(r["date"][:4]) if r["date"][:4].isdigit() else 0,
                         "in_corpus": False, "url": config.EURLEX_URL.format(celex=r["celex"]), "aliases": []})
    data["documents"] = docs
    known = {d["celex"] for d in docs}

    data["subjects"] = [{"celex": d.celex, "label": ev} for d in corpus.documents.values() for ev in d.eurovoc]

    # --- articles (ordre du fichier = ordre dans l'acte)
    chunk_ids = load_chunk_ids()
    articles, position = [], defaultdict(int)
    for art in corpus.articles.values():
        position[art.celex] += 1
        articles.append({"id": art.id, "celex": art.celex, "num": art.num, "title": art.title, "text": art.text,
                         "position": position[art.celex], "chunk_ids": chunk_ids.get(art.id, []),
                         "n_chunks": len(chunk_ids.get(art.id, []))})
    data["articles"] = articles

    data["next"] = [{"a": a["id"], "b": b["id"]} for a, b in zip(articles, articles[1:]) if a["celex"] == b["celex"]]

    # --- relations entre actes (Cellar)
    doc_rels = {}
    for r in read_csv(config.RELATIONS_PATH) + read_csv(config.EXTERNAL_RELATIONS_PATH):
        if r["source"] in known and r["target"] in known:
            doc_rels[(r["source"], r["target"], r["type"])] = {"s": r["source"], "t": r["target"], "type": r["type"]}
    data["doc_relations"] = list(doc_rels.values())
    rel_types_by_pair = defaultdict(set)
    for s, t, typ in doc_rels:
        rel_types_by_pair[(s, t)].add(typ)

    # --- renvois entre articles
    data["refers_to"] = [{"a": r["source_article"], "b": r["target_article"]}
                         for r in read_csv(config.ARTICLE_REFS_PATH)
                         if r["target_in_corpus"] == "True" and r["target_article"] in corpus.articles]

    # --- article -> acte : modifications / abrogations opérées, actes mentionnés
    art_rels, mentions = [], []
    for art in corpus.articles.values():
        targets = {t: types for (s, t), types in rel_types_by_pair.items() if s == art.celex}
        for (typ, target), evidence in article_relations(art.title, art.text, art.celex, targets).items():
            if target in known:
                art_rels.append({"a": art.id, "t": target, "type": typ, "evidence": evidence})
        for target in mentioned_acts(f"{art.title} {art.text}"):
            if target in known and target != art.celex:
                mentions.append({"a": art.id, "t": target})
    data["article_relations"] = art_rels
    data["mentions"] = mentions

    if with_embeddings:
        emb = load_article_embeddings(chunk_ids)
        data["embeddings"] = [{"id": aid, "embedding": v} for aid, v in emb.items()]
    return data


def summary(data: dict) -> None:
    docs = data["documents"]
    print(f"Documents   : {len(docs)} ({sum(d['in_corpus'] for d in docs)} du corpus, "
          f"{sum(not d['in_corpus'] for d in docs)} hors corpus)")
    print(f"EuroVoc     : {len({s['label'] for s in data['subjects']})} descripteurs, {len(data['subjects'])} HAS_SUBJECT")
    print(f"Articles    : {len(data['articles'])} ; NEXT : {len(data['next'])} ; REFERS_TO : {len(data['refers_to'])}")
    by_type = defaultdict(int)
    for r in data["doc_relations"]:
        by_type[r["type"]] += 1
    print(f"Doc -> Doc  : {dict(by_type)}")
    by_type = defaultdict(int)
    for r in data["article_relations"]:
        by_type[(r["type"], r["evidence"])] += 1
    print(f"Art -> Doc  : {dict(by_type)} ; MENTIONS : {len(data['mentions'])}")
    if "embeddings" in data:
        print(f"Embeddings  : {len(data['embeddings'])} articles")


# ----------------------------------------------------------------------------------------- chargement
SCHEMA = [
    "CREATE CONSTRAINT document_celex IF NOT EXISTS FOR (d:Document) REQUIRE d.celex IS UNIQUE",
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT eurovoc_label IF NOT EXISTS FOR (e:EuroVoc) REQUIRE e.label IS UNIQUE",
    "CREATE INDEX article_celex IF NOT EXISTS FOR (a:Article) ON (a.celex)",
    f"""CREATE VECTOR INDEX article_embedding IF NOT EXISTS FOR (a:Article) ON a.embedding
        OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBEDDING_DIM}, `vector.similarity_function`: 'cosine'}}}}""",
]

LOADS = [
    ("documents", """UNWIND $rows AS row MERGE (d:Document {celex: row.celex})
        SET d.title = row.title, d.short_name = row.short_name, d.label = row.label, d.type = row.type,
            d.date = row.date, d.year = row.year, d.in_corpus = row.in_corpus, d.url = row.url, d.aliases = row.aliases"""),
    ("subjects", """UNWIND $rows AS row MATCH (d:Document {celex: row.celex})
        MERGE (e:EuroVoc {label: row.label}) MERGE (d)-[:HAS_SUBJECT]->(e)"""),
    ("articles", """UNWIND $rows AS row MATCH (d:Document {celex: row.celex})
        MERGE (a:Article {id: row.id})
        SET a.celex = row.celex, a.num = row.num, a.title = row.title, a.text = row.text,
            a.position = row.position, a.chunk_ids = row.chunk_ids, a.n_chunks = row.n_chunks
        MERGE (d)-[:HAS_ARTICLE]->(a)"""),
    ("next", """UNWIND $rows AS row MATCH (a:Article {id: row.a}), (b:Article {id: row.b}) MERGE (a)-[:NEXT]->(b)"""),
    ("doc_relations", """UNWIND $rows AS row MATCH (s:Document {celex: row.s}), (t:Document {celex: row.t})
        CALL apoc.merge.relationship(s, row.type, {}, {source: 'cellar'}, t, {}) YIELD rel RETURN count(rel)"""),
    ("refers_to", """UNWIND $rows AS row MATCH (a:Article {id: row.a}), (b:Article {id: row.b}) MERGE (a)-[:REFERS_TO]->(b)"""),
    ("article_relations", """UNWIND $rows AS row MATCH (a:Article {id: row.a}), (t:Document {celex: row.t})
        CALL apoc.merge.relationship(a, row.type, {}, {evidence: row.evidence}, t, {}) YIELD rel RETURN count(rel)"""),
    ("mentions", """UNWIND $rows AS row MATCH (a:Article {id: row.a}), (t:Document {celex: row.t}) MERGE (a)-[:MENTIONS]->(t)"""),
    ("embeddings", """UNWIND $rows AS row MATCH (a:Article {id: row.id})
        CALL db.create.setNodeVectorProperty(a, 'embedding', row.embedding)"""),
]


def load(data: dict, reset: bool = True) -> None:
    from graphrag.db import run, run_batches

    if reset:
        logger.info("Base vidée")
        run("MATCH (n) DETACH DELETE n")
    for q in SCHEMA:
        run(q)
    for key, query in LOADS:
        if key in data:
            t = time.time()
            n = run_batches(query, data[key], batch_size=200 if key == "embeddings" else 500)
            logger.info(f"{key:<18} {n:>6} lignes  ({time.time() - t:.1f}s)")
    run("CALL db.awaitIndexes(300)")


def checks() -> None:
    from graphrag.db import run

    print("\n=== Nœuds par label")
    for r in run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"):
        print(f"  {r['label']:<10} {r['n']:>6}")
    print("=== Arêtes par type (et extrémités)")
    for r in run("""MATCH (a)-[r]->(b) RETURN type(r) AS type, labels(a)[0] AS de, labels(b)[0] AS vers, count(*) AS n
                    ORDER BY n DESC"""):
        print(f"  {r['type']:<12} {r['de']:>8} -> {r['vers']:<9} {r['n']:>6}")
    iso = run("MATCH (a:Article) WHERE NOT (a)-[:REFERS_TO]-() RETURN count(a) AS n")[0]["n"]
    iso_all = run("MATCH (a:Article) WHERE NOT (a)-[:REFERS_TO|NEXT]-() RETURN count(a) AS n")[0]["n"]
    print(f"=== Articles sans REFERS_TO : {iso} ; sans REFERS_TO ni NEXT : {iso_all}")
    emb = run("MATCH (a:Article) WHERE a.embedding IS NOT NULL RETURN count(a) AS n")[0]["n"]
    print(f"=== Articles avec embedding : {emb}")

    print("\n=== Critère de fin de l'étape 1")
    rows = run("""MATCH (a:Article {id: '32016R0679_art_9'})-[:REFERS_TO]->(b:Article {id: '32016R0679_art_89'})
                  RETURN a.id AS source, a.title AS titre_source, b.id AS cible, b.title AS titre_cible""")
    print(f"  {'OK' if rows else 'ÉCHEC'} : {rows}")
    print("=== Contrôles relationnels")
    for q, label in [
        ("""MATCH (a:Article)-[r:REPEALS]->(d:Document) WHERE a.celex IN ['32016R0679','32022L2555']
            RETURN a.id AS article, d.short_name AS abroge, r.evidence AS preuve""", "RGPD / NIS2 abrogent"),
        ("""MATCH (a:Article)-[r:AMENDS]->(d:Document {celex: '32000L0031'}) RETURN a.id AS article, r.evidence AS preuve""",
         "articles qui modifient 2000/31/CE"),
        ("""MATCH (a:Article {celex: '32022R1925'})-[r:AMENDS]->(d:Document)
            RETURN a.id AS article, d.short_name AS modifie, d.in_corpus AS dans_corpus""", "le DMA modifie"),
        ("""MATCH (s:Document)-[:AMENDS]->(d:Document {celex: '32019R0818'}) WHERE s.in_corpus
            RETURN s.short_name AS acte_du_corpus""", "actes du corpus qui modifient 2019/818"),
    ]:
        print(f"  {label} : {run(q)}")
    print("\nVecteur : 5 voisins de l'article 33 du RGPD")
    for r in run("""MATCH (a:Article {id: '32016R0679_art_33'})
                    CALL db.index.vector.queryNodes('article_embedding', 6, a.embedding) YIELD node, score
                    WHERE node <> a RETURN node.id AS id, node.title AS titre, round(score, 3) AS score"""):
        print(f"  {r['score']}  {r['id']}  {r['titre']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="préparer et compter, sans Neo4j")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--no-reset", action="store_true", help="ne pas vider la base avant chargement")
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()

    if args.checks_only:
        checks()
        return
    t = time.time()
    data = prepare(with_embeddings=not args.no_embeddings)
    summary(data)
    if args.dry_run:
        return
    load(data, reset=not args.no_reset)
    logger.info(f"Graphe chargé en {time.time() - t:.0f}s")
    checks()


if __name__ == "__main__":
    main()
