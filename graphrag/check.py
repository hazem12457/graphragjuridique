"""Vérifie que tout est prêt : données du RAG vectoriel, données hors corpus, connexion Neo4j.

Usage : python -m graphrag.check
"""
import csv
import json

from graphrag import config


def count_lines(path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> int:
    ok = True
    print(f"Projet RAG vectoriel : {config.VECTOR_RAG_DIR}")
    for label, path in [("articles", config.ARTICLES_PATH), ("actes", config.DOCUMENTS_PATH),
                        ("relations entre actes", config.RELATIONS_PATH),
                        ("renvois entre articles", config.ARTICLE_REFS_PATH),
                        ("chunks", config.CHUNKS_PATH),
                        ("relations hors corpus", config.EXTERNAL_RELATIONS_PATH),
                        ("actes hors corpus", config.EXTERNAL_DOCUMENTS_PATH)]:
        if path.exists():
            n = count_lines(path) - (1 if path.suffix == ".csv" else 0)
            print(f"  [ok] {label:<24} {n:>6}  ({path.name})")
        else:
            ok = False
            print(f"  [!!] {label:<24} absent : {path}")

    # le premier chunk doit porter un article_id (lien chunk -> nœud Article)
    if config.CHUNKS_PATH.exists():
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            meta = json.loads(f.readline())["metadata"]
        print(f"  [ok] métadonnées d'un chunk : {sorted(meta)}")

    with open(config.EXTERNAL_RELATIONS_PATH, encoding="utf-8") as f:
        types = {}
        for row in csv.DictReader(f):
            types[row["type"]] = types.get(row["type"], 0) + 1
    print(f"  relations hors corpus par type : {types}")

    print(f"\nNeo4j : {config.NEO4J_URI} (base « {config.NEO4J_DATABASE} », utilisateur « {config.NEO4J_USER} »)")
    try:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            version = driver.execute_query("CALL dbms.components() YIELD versions RETURN versions[0] AS v",
                                           database_=config.NEO4J_DATABASE).records[0]["v"]
            print(f"  [ok] connecté, Neo4j {version}")
    except Exception as e:
        ok = False
        print(f"  [!!] connexion impossible : {type(e).__name__}: {e}")
        print("       -> démarrer la base dans Neo4j Desktop et renseigner NEO4J_PASSWORD dans .env")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
