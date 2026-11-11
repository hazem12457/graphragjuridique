"""Tire au hasard (graine fixe) des renvois entre articles d'actes différents, pour rédiger des questions multi-hop neuves.

Le tirage est fait avant toute évaluation sur ces paires, et la liste est parcourue dans l'ordre : on ne choisit pas
les paires qui avantagent un système. Sont écartées d'office les paires déjà présentes dans un jeu d'évaluation et
les articles purement formels (définitions, modifications d'actes, abrogations, entrée en vigueur).

Usage : python scripts/sample_cross_act_refs.py [--n 40]
"""
import argparse
import csv
import json
import random
import re

from graphrag import config

config.use_vector_rag()
from app.rag.corpus import get_corpus, normalize  # noqa: E402
from app.rag.eval_dataset import EVAL_QUESTIONS  # noqa: E402

SEED = 2026
FORMAL = re.compile(r"definition|modification|abrogation|entree en vigueur|objet|champ d'application|comite|procedure de comite|reexamen|evaluation et")


def known_golds() -> set[str]:
    gold = {a for q in EVAL_QUESTIONS for a in q["gold"]}
    for name in ("questions_test_28.jsonl",):
        with open(config.DATA_DIR / name, encoding="utf-8") as f:
            gold |= {a for q in map(json.loads, f) for a in q["gold"]}
    return gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    corpus = get_corpus()
    gold = known_golds()
    with open(config.ARTICLE_REFS_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["target_in_corpus"] == "True" and r["target_article"] in corpus.articles]
    pairs = []
    for r in rows:
        a, b = corpus.articles.get(r["source_article"]), corpus.articles[r["target_article"]]
        if a and a.celex != b.celex and not (a.id in gold and b.id in gold) \
                and not FORMAL.search(normalize(a.title)) and not FORMAL.search(normalize(b.title)):
            pairs.append((a.id, b.id))
    pairs = sorted(set(pairs))
    print(f"{len(pairs)} paires inter-actes éligibles")
    random.Random(SEED).shuffle(pairs)
    for i, (x, y) in enumerate(pairs[:args.n], start=1):
        a, b = corpus.articles[x], corpus.articles[y]
        print(f"{i:>2}. {x} [{a.title}]  ->  {y} [{b.title}]")


if __name__ == "__main__":
    main()
