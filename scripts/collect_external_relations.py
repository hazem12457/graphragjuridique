"""Complète le corpus avec les relations vers des actes HORS corpus.

Pourquoi : la collecte du RAG vectoriel (`scripts/collect_eurlex.py`, `fetch_relations`) ne garde
que les relations dont la cible est l'un des 100 actes collectés. Toutes les abrogations
disparaissent ainsi (le RGPD abroge la directive 95/46/CE, NIS2 abroge la directive (UE) 2016/1148 :
ces actes ne sont pas dans le corpus), ainsi qu'une partie des modifications (le DMA modifie la
directive (UE) 2020/1828, absente du corpus).

Ce script récupère, pour les 100 actes du corpus :
  - les relations AMENDS / REPEALS / BASED_ON sortantes ET entrantes, quel que soit l'autre acte ;
  - le titre officiel français, la date et le type de ces actes hors corpus.
Il réutilise les fonctions SPARQL de `collect_eurlex.py` (projet RAG vectoriel) et l'endpoint
public de l'Office des publications.

Entrée  : <VECTOR_RAG_DIR>/data/eurlex/documents.csv
Sorties : data/eurlex_external/relations_external.csv  (source,target,type)
          data/eurlex_external/documents_external.csv  (celex,type,date,title)

Usage : python scripts/collect_external_relations.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag import config  # noqa: E402

sys.path.insert(0, str(config.VECTOR_RAG_DIR / "scripts"))
from collect_eurlex import LEGIS_RE, REL_TYPES, batches, fetch_metadata, sparql, values  # noqa: E402

# CITES est exclu : trop nombreux vers l'extérieur, et déjà couvert par les renvois extraits du texte
KEPT = {p: t for p, t in REL_TYPES.items() if t != "CITES"}


def fetch_all_relations(celexes: list[str]) -> set[tuple[str, str, str]]:
    preds = " ".join(f"cdm:{p}" for p in KEPT)
    rels = set()
    for batch in batches(celexes):
        v = values(batch)
        outgoing = sparql(f"""SELECT DISTINCT ?s ?p ?t WHERE {{
            VALUES ?s {{ {v} }} VALUES ?p {{ {preds} }}
            ?w cdm:resource_legal_id_celex ?s . ?w ?p ?x . ?x cdm:resource_legal_id_celex ?t .
            FILTER(REGEX(STR(?t), "{LEGIS_RE}")) }}""")
        incoming = sparql(f"""SELECT DISTINCT ?s ?p ?t WHERE {{
            VALUES ?t {{ {v} }} VALUES ?p {{ {preds} }}
            ?x cdm:resource_legal_id_celex ?t . ?w ?p ?x . ?w cdm:resource_legal_id_celex ?s .
            FILTER(REGEX(STR(?s), "{LEGIS_RE}")) }}""")
        for r in outgoing + incoming:
            if r["s"] != r["t"]:
                rels.add((r["s"], r["t"], KEPT[r["p"].split("#")[1]]))
    return rels


def main():
    with open(config.DOCUMENTS_PATH, encoding="utf-8") as f:
        corpus = [row["celex"] for row in csv.DictReader(f)]
    in_corpus = set(corpus)

    print(f"Relations AMENDS/REPEALS/BASED_ON de {len(corpus)} actes (entrantes et sortantes)…")
    rels = fetch_all_relations(corpus)
    external = sorted({c for s, t, _ in rels for c in (s, t)} - in_corpus)
    print(f"  {len(rels)} relations, {len(external)} actes hors corpus")

    print("Titres des actes hors corpus…")
    meta = fetch_metadata(external)

    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EXTERNAL_RELATIONS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "type"])
        w.writerows(sorted(rels))
    with open(config.EXTERNAL_DOCUMENTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["celex", "type", "date", "title"])
        for c in external:
            m = meta.get(c, {})
            w.writerow([c, m.get("type", ""), m.get("date", ""), m.get("title", "")])

    by_type = {}
    for _, _, t in rels:
        by_type[t] = by_type.get(t, 0) + 1
    print(f"OK : {by_type} · {sum(1 for c in external if meta.get(c, {}).get('title'))}/{len(external)} titres")


if __name__ == "__main__":
    main()
