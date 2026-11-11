"""Connexion Neo4j (un seul driver, créé à la demande)."""
import logging
from functools import lru_cache

# Neo4j 2026 signale db.index.vector.queryNodes comme obsolète (remplacé par SEARCH) : la procédure
# fonctionne toujours ; on masque ces avis pour garder des sorties lisibles.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

from graphrag import config


@lru_cache(maxsize=1)
def get_driver():
    from neo4j import GraphDatabase

    if not config.NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD vide : le renseigner dans GraphRAG-juridique/.env")
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def run(query: str, **params) -> list[dict]:
    """Exécute une requête Cypher et renvoie les lignes sous forme de dictionnaires."""
    records = get_driver().execute_query(query, params, database_=config.NEO4J_DATABASE).records
    return [r.data() for r in records]


def run_batches(query: str, rows: list[dict], batch_size: int = 500) -> int:
    """`UNWIND $rows AS row …` par lots (500 lignes par défaut)."""
    for i in range(0, len(rows), batch_size):
        run(query, rows=rows[i:i + batch_size])
    return len(rows)
