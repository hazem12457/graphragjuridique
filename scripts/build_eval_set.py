"""Construit le jeu final de 60 questions (data/eval_60.jsonl) sans modifier les fichiers du projet vectoriel.

    B-*  24 questions de app/rag/eval_dataset.py (ayant servi à concevoir le retriever)
    T-*  28 questions de test formulées par l'auteur (data/questions_test_28.jsonl)
    U-R* questions relationnelles de data/eval_questions.jsonl qui ont des gold_articles (R03, R04)
    N-*  6 questions multi-hop neuves, tirées de renvois inter-actes au hasard (scripts/sample_cross_act_refs.py)

Usage : python scripts/build_eval_set.py
"""
import json

from graphrag import config

config.use_vector_rag()
from app.rag.corpus import get_corpus  # noqa: E402
from app.rag.eval_dataset import EVAL_QUESTIONS  # noqa: E402


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rows = [{**q, "id": f"B-{q['id']}"} for q in EVAL_QUESTIONS]
    rows += read_jsonl(config.DATA_DIR / "questions_test_28.jsonl")
    for q in read_jsonl(config.VECTOR_RAG_DIR / "data" / "eval_questions.jsonl"):
        if q["type"] == "relationnelle" and q.get("gold_articles"):
            rows.append({"id": f"U-{q['id']}", "type": q["type"], "question": q["question"], "gold": q["gold_articles"]})
    rows += read_jsonl(config.DATA_DIR / "questions_neuves_6.jsonl")

    corpus = get_corpus()
    missing = [(r["id"], g) for r in rows for g in r["gold"] if g not in corpus.articles]
    if missing:
        raise SystemExit(f"Articles attendus absents du corpus : {missing}")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("Identifiants en double")

    out = config.DATA_DIR / "eval_60.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    print(f"{len(rows)} questions écrites dans {out} : {counts}")


if __name__ == "__main__":
    main()
