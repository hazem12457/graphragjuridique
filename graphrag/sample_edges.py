"""Étape 2 : qualité du graphe — échantillon d'arêtes extraites par expressions régulières, à vérifier à la main.

    REFERS_TO (renvois article -> article, `article_refs.csv`)       30 arêtes
    Article -[:AMENDS|REPEALS]-> Document (`acts.article_relations`)  20 arêtes

Graine aléatoire fixe : le même échantillon à chaque exécution. Pour chaque arête, le passage du texte
source qui l'a produite est affiché, avec le titre de la cible. Remplir la colonne `correct` (1 / 0) et
`erreur` (type d'erreur), puis :

    python -m graphrag.sample_edges                  # crée results/edge_sample.csv
    python -m graphrag.sample_edges --score          # précision à partir du fichier annoté
"""
import argparse
import csv
import random
import re

from graphrag import config
from graphrag.build_graph import prepare

config.use_vector_rag()
from app.rag.corpus import get_corpus  # noqa: E402

SEED = 42
OUT = config.ROOT / "results" / "edge_sample.csv"
ART_RE = re.compile(r"\barticles?\s+(\d+[a-z]?)", re.IGNORECASE)


def context_for_ref(text: str, target_num: str, width: int = 160) -> str:
    """Passage du texte source où apparaît le renvoi vers l'article `target_num`."""
    for m in ART_RE.finditer(text):
        if m.group(1) == target_num:
            return "…" + text[max(0, m.start() - width // 2): m.end() + width].replace("\n", " ") + "…"
    # « articles 12 à 15 », listes « articles 5, 6 et 7 » : le numéro n'est pas en tête
    m = re.search(rf"\b{re.escape(target_num)}\b", text)
    return "…" + text[max(0, m.start() - width): m.end() + width // 2] + "…" if m else "(passage non retrouvé)"


def build_sample() -> None:
    corpus = get_corpus()
    data = prepare(with_embeddings=False)
    rng = random.Random(SEED)
    rows = []

    for e in rng.sample(data["refers_to"], 30):
        src, tgt = corpus.articles[e["a"]], corpus.articles[e["b"]]
        rows.append({
            "kind": "REFERS_TO", "source": src.id, "target": tgt.id,
            "source_header": corpus.article_header(src), "target_header": corpus.article_header(tgt),
            "evidence": context_for_ref(src.text, tgt.num), "correct": "", "erreur": "",
        })

    docs = {d["celex"]: d for d in data["documents"]}
    for e in rng.sample(data["article_relations"], 20):
        src = corpus.articles[e["a"]]
        rows.append({
            "kind": f"{e['type']} ({e['evidence']})", "source": src.id, "target": e["t"],
            "source_header": corpus.article_header(src), "target_header": docs[e["t"]]["short_name"],
            "evidence": src.text[:300] + "…", "correct": "", "erreur": "",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig : ouverture correcte dans Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} arêtes à vérifier -> {OUT}")


def score() -> None:
    with open(OUT, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if r["correct"].strip() in ("0", "1")]
    for kind in sorted({r["kind"].split(" ")[0] for r in rows}):
        sub = [r for r in rows if r["kind"].startswith(kind)]
        ok = sum(int(r["correct"]) for r in sub)
        print(f"{kind:<10} précision = {ok}/{len(sub)} = {ok / len(sub):.0%}")
        errors = [r["erreur"] for r in sub if r["correct"].strip() == "0" and r["erreur"]]
        if errors:
            print("           erreurs :", "; ".join(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    score() if args.score else build_sample()
