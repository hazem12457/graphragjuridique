"""RAGAS sur des réponses déjà générées (scripts/answer_user_questions.py) : seul le juge gpt-oss-20b est appelé.

Sous-ensemble : B4, C3, C5, les 3 questions tirées au sort à article(s) attendu(s) dont le contexte est le plus court dans les deux modes,
car une requête du juge doit tenir dans la limite Groq de 8 000 tokens par minute. Ce critère de coût favorise les contextes courts :
ces scores ne se généralisent pas à l'ensemble des questions.

Usage (depuis la racine du projet) : $env:PYTHONPATH="."; python -u scripts/ragas_user_answers.py --answers results/answers_utilisateur_XXX.json
Résultats : results/ragas_utilisateur_<horodatage>.json
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("SEED_CACHE_PATH", str(ROOT / "data" / "cache" / "user_test_candidates.json"))

from graphrag import config  # noqa: E402
from graphrag.evaluate import retrieve_fn_for  # noqa: E402

config.use_vector_rag()
from app.rag.evaluate import evaluate_rag  # noqa: E402
from app.rag.pipeline import article_ids  # noqa: E402

IDS = ("B4", "C3", "C5")
PAUSE_SECONDS = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    args = parser.parse_args()

    data = json.loads((ROOT / args.answers).read_text(encoding="utf-8"))
    out = ROOT / "results" / f"ragas_utilisateur_{datetime.now():%Y%m%d_%H%M%S}.json"
    results = []
    for row in (r for r in data["rows"] if r["id"] in IDS):
        for mode in data["modes"]:
            chunks = retrieve_fn_for(mode, full_articles=True)(row["question"], 5)
            scores = evaluate_rag(row["question"], row[mode]["answer"], chunks, reference=row["reference"])
            same = article_ids(chunks) == row[mode]["selected"]
            results.append({"id": row["id"], "mode": mode, "meme_selection": same, **scores})
            print(results[-1])
            out.write_text(json.dumps({"reponses": args.answers, "juge": "openai/gpt-oss-20b", "resultats": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(PAUSE_SECONDS)
    print(f"Résultats : {out}")


if __name__ == "__main__":
    main()
