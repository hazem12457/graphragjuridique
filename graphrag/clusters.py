"""Clusters thématiques des articles (Louvain), écrits dans Neo4j : (:Article)-[:IN_CLUSTER]->(:Cluster {id, label, size}).

Deux graphes sont comparés, car 91 % des renvois restent à l'intérieur d'un acte :
    structure : REFERS_TO + NEXT                     -> on s'attend à retrouver les actes, pas des thèmes
    themes    : REFERS_TO + 10 plus proches voisins sémantiques (cosinus des embeddings e5)  -> graphe retenu pour le livrable

Étiquette d'un cluster : descripteurs EuroVoc sur-représentés (couverture dans le cluster × lift par rapport au corpus).
EuroVoc décrit l'acte, pas l'article : un cluster de clauses standard (« Comité », « Entrée en vigueur ») hériterait du
thème de ses actes. Si au moins 40 % de ses articles portent un titre de clause formelle, il est étiqueté comme tel.

À relancer après `build_graph` (qui vide la base).
Usage : python -m graphrag.clusters [--no-write]
Résultats : results/clusters.json
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from graphrag import config
from graphrag.db import run, run_batches

config.use_vector_rag()
from app.rag.corpus import normalize  # noqa: E402

FORMAL_TITLE = re.compile(r"entree en vigueur|abrogation|comite|exercice de la delegation|dispositions? transitoires?"
                          r"|procedure d.urgence|procedure d.examen|procedure consultative|transposition|reexamen|\bevaluation\b")
FORMAL_SHARE = 0.40
SEED = 42
KNN = 10
REFERS_WEIGHT = 1.0
MIN_COVERAGE = 0.15
RESULTS = config.ROOT / "results" / "clusters.json"


def load():
    arts = run("MATCH (a:Article) RETURN a.id AS id, a.celex AS celex, a.title AS title, a.embedding AS e ORDER BY a.id")
    ids = [a["id"] for a in arts]
    celex = {a["id"]: a["celex"] for a in arts}
    emb = np.asarray([a["e"] for a in arts], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    refs = [(r["a"], r["b"]) for r in run("MATCH (a:Article)-[:REFERS_TO]->(b:Article) RETURN a.id AS a, b.id AS b")]
    nexts = [(r["a"], r["b"]) for r in run("MATCH (a:Article)-[:NEXT]->(b:Article) RETURN a.id AS a, b.id AS b")]
    subjects = defaultdict(list)
    for r in run("MATCH (d:Document)-[:HAS_SUBJECT]->(e:EuroVoc) RETURN d.celex AS c, e.label AS l ORDER BY l"):
        subjects[r["c"]].append(r["l"])
    titles = {a["id"]: a["title"] or "" for a in arts}
    return ids, celex, titles, emb, refs, nexts, subjects


def knn_edges(ids: list[str], emb: np.ndarray, k: int = KNN) -> dict[tuple[str, str], float]:
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    edges = {}
    for i, row in enumerate(sim):
        for j in np.argpartition(-row, k)[:k]:
            key = tuple(sorted((ids[i], ids[int(j)])))
            edges[key] = max(edges.get(key, 0.0), float(row[j]))
    return edges


def louvain(ids: list[str], edges: dict[tuple[str, str], float]) -> tuple[list[set[str]], float]:
    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_weighted_edges_from((a, b, w) for (a, b), w in edges.items() if a != b)
    communities = nx.community.louvain_communities(g, weight="weight", seed=SEED)
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    return communities, nx.community.modularity(g, communities, weight="weight")


def describe(communities, celex, ids) -> dict:
    assign = {a: i for i, c in enumerate(communities) for a in c}
    sizes = [len(c) for c in communities]
    return {"n_clusters": len(communities), "n_singletons": sum(s == 1 for s in sizes),
            "taille_max": max(sizes), "taille_mediane": int(np.median(sizes)),
            "nmi_avec_acte": round(normalized_mutual_info_score([celex[a] for a in ids], [assign[a] for a in ids]), 3),
            "clusters_de_3_articles_ou_plus": sum(s >= 3 for s in sizes),
            "clusters_de_3_articles_ou_plus_couvrant_plusieurs_actes": sum(s >= 3 and len({celex[a] for a in c}) > 1
                                                                            for s, c in zip(sizes, communities)),
            "part_articles_dans_cluster_multi_actes": round(
                sum(len(c) for c in communities if len({celex[a] for a in c}) > 1) / len(ids), 3)}


def label_cluster(members: set[str], celex: dict, subjects: dict, global_freq: Counter, n_total: int) -> list[str]:
    local = Counter(s for a in members for s in subjects.get(celex[a], []))
    scored = []
    for label, n in local.items():
        coverage = n / len(members)
        if coverage >= MIN_COVERAGE:
            lift = coverage / (global_freq[label] / n_total)
            scored.append((coverage * math.log1p(lift), label))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [label for _, label in scored[:3]] or [label for label, _ in local.most_common(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true", help="ne pas écrire dans Neo4j")
    args = parser.parse_args()

    ids, celex, titles, emb, refs, nexts, subjects = load()
    print(f"{len(ids)} articles ; REFERS_TO {len(refs)} ; NEXT {len(nexts)}")

    structure = {tuple(sorted(e)): 1.0 for e in refs + nexts}
    themes = knn_edges(ids, emb)
    knn_only = len(themes)
    for a, b in refs:
        key = tuple(sorted((a, b)))
        themes[key] = themes.get(key, 0.0) + REFERS_WEIGHT
    print(f"graphe structure : {len(structure)} arêtes ; graphe thèmes : {len(themes)} arêtes (dont {knn_only} de similarité)")

    comm_struct, mod_struct = louvain(ids, structure)
    comm_theme, mod_theme = louvain(ids, themes)
    report = {"seed": SEED, "knn": KNN,
              "structure": {"modularite": round(mod_struct, 3), **describe(comm_struct, celex, ids)},
              "themes": {"modularite": round(mod_theme, 3), **describe(comm_theme, celex, ids)}}
    print(json.dumps(report, ensure_ascii=False, indent=2))

    global_freq = Counter(s for a in ids for s in subjects.get(celex[a], []))
    clusters = []
    for i, members in enumerate(comm_theme):
        acts = Counter(celex[a] for a in members)
        top_titles = [t for t, _ in Counter(titles[a] for a in members if titles[a]).most_common(3)]
        formal_share = sum(bool(FORMAL_TITLE.search(normalize(titles[a]))) for a in members) / len(members)
        if formal_share >= FORMAL_SHARE:
            label = "Clauses formelles : " + " · ".join(top_titles)
        else:
            label = " · ".join(label_cluster(members, celex, subjects, global_freq, len(ids)))
        clusters.append({"id": i, "size": len(members), "label": label, "n_acts": len(acts),
                         "formal_share": round(formal_share, 2), "top_titles": top_titles,
                         "top_acts": [c for c, _ in acts.most_common(3)]})
    report["clusters"] = clusters
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nClusters :")
    for c in clusters:
        print(f"  #{c['id']:<3} {c['size']:>4} articles  {c['n_acts']:>3} actes  {c['label']}")

    if not args.no_write:
        run("MATCH (c:Cluster) DETACH DELETE c")
        run("CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (c:Cluster) REQUIRE c.id IS UNIQUE")
        run_batches("UNWIND $rows AS row MERGE (c:Cluster {id: row.id}) SET c.label = row.label, c.size = row.size, c.n_acts = row.n_acts",
                    clusters)
        rows = [{"a": a, "c": i} for i, members in enumerate(comm_theme) for a in members]
        run_batches("UNWIND $rows AS row MATCH (a:Article {id: row.a}), (c:Cluster {id: row.c}) MERGE (a)-[:IN_CLUSTER]->(c)", rows)
        n = run("MATCH (:Article)-[r:IN_CLUSTER]->(:Cluster) RETURN count(r) AS n")[0]["n"]
        print(f"\nÉcrit dans Neo4j : {len(clusters)} Cluster, {n} IN_CLUSTER")
    print(f"Résultats : {RESULTS}")


if __name__ == "__main__":
    main()
