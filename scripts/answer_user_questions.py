"""Génère et enregistre les réponses de la baseline et du Graph RAG, avec la même chaîne que l'interface.

Mêmes conditions pour les deux modes : articles complets, même prompt, même LLM, 5 articles distincts.
Questions : les 2 cas signalés par l'utilisateur, puis les questions tirées au sort, un titre après l'autre (A, B, C, D, A, B…),
pour couvrir chaque titre même si le quota s'épuise. Les réponses sont à relire : le script ne note pas le fond.

Arrêt propre : au-delà de --max-tokens (estimation) ou après 3 réponses indisponibles de suite (quota journalier).

Usage (depuis la racine du projet) : $env:PYTHONPATH="."; python -u scripts/answer_user_questions.py
Résultats : results/answers_utilisateur_<horodatage>.json
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
from graphrag import retriever as graph  # noqa: E402
from graphrag.compare_answers import ANSWER_TOKENS, TOKENS_PER_MINUTE, cites  # noqa: E402
from graphrag.evaluate import retrieve_fn_for  # noqa: E402

config.use_vector_rag()
from app.rag.pipeline import NOT_AVAILABLE, article_ids, format_context, generate_answer  # noqa: E402

MODES = ("advanced", "graph_linked")
REPORTED = [
    {"id": "P1", "type": "P_cas_signale", "question": "Une plateforme peut-elle faire de la publicité ciblée à partir de données sensibles ?",
     "gold": ["32022R2065_art_26"], "reference": "Non : article 26, paragraphe 3, du DSA."},
    {"id": "P2", "type": "P_cas_signale", "question": "Comment l'analyse d'impact sur les droits fondamentaux de l'AI Act s'articule-t-elle avec l'AIPD ?",
     "gold": ["32024R1689_art_27"], "reference": "Article 27, paragraphe 4, de l'AI Act : elle complète l'AIPD."},
]


def round_robin(questions: list[dict]) -> list[dict]:
    sections = {}
    for q in questions:
        sections.setdefault(q["type"], []).append(q)
    ordered = []
    while any(sections.values()):
        for items in sections.values():
            if items:
                ordered.append(items.pop(0))
    return ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/questions_utilisateur_21.jsonl")
    parser.add_argument("--max-tokens", type=int, default=150000)
    parser.add_argument("--ids", nargs="+", default=None, help="questions ajoutées hors tirage (remplace le tirage et les 2 cas signalés)")
    args = parser.parse_args()

    with open(ROOT / args.file, encoding="utf-8") as f:
        all_questions = list(map(json.loads, f))
    if args.ids:
        questions = [q for q in all_questions if q["id"] in args.ids]
    else:
        questions = REPORTED + round_robin([q for q in all_questions if q["reponse_llm"]])

    out = ROOT / "results" / f"answers_utilisateur_{datetime.now():%Y%m%d_%H%M%S}.json"
    rows, spent, failures = [], 0, 0
    fns = {m: retrieve_fn_for(m, full_articles=True) for m in MODES}

    def save(status):
        out.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "modes": MODES, "statut": status,
                                   "tokens_estimes": spent, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    for i, item in enumerate(questions, start=1):
        row = {k: item[k] for k in ("id", "type", "question", "gold", "reference")}
        for mode in MODES:
            start = time.time()
            chunks = fns[mode](item["question"], 5)
            selected = article_ids(chunks)
            context_tokens = len(format_context(chunks)) // 4
            answer = generate_answer(item["question"], chunks)
            if answer == NOT_AVAILABLE:
                time.sleep(65)
                answer = generate_answer(item["question"], chunks)
            spent += context_tokens + ANSWER_TOKENS
            failures = failures + 1 if answer == NOT_AVAILABLE else 0
            row[mode] = {"selected": selected, "context_tokens": context_tokens, "answer": answer, "unavailable": answer == NOT_AVAILABLE,
                         "gold_selected": [g for g in item["gold"] if g in selected],
                         "gold_cited": [g for g in item["gold"] if answer != NOT_AVAILABLE and cites(answer, g)],
                         "seconds": round(time.time() - start, 1)}
            print(f"{i:>2}/{len(questions)} {item['id']:<4} {mode:<13} sélectionnés {len(row[mode]['gold_selected'])}/{len(item['gold'])} "
                  f"cités {len(row[mode]['gold_cited'])}/{len(item['gold'])}{' INDISPONIBLE' if row[mode]['unavailable'] else ''}")
            wait = (context_tokens + ANSWER_TOKENS) / TOKENS_PER_MINUTE * 60 - (time.time() - start)
            if wait > 0:
                time.sleep(wait)
            if failures >= 3:
                break
        rows.append(row)
        save("en cours")
        if failures >= 3:
            save("arrêté : 3 réponses indisponibles de suite")
            print("Arrêt : 3 réponses indisponibles de suite (quota probable).")
            return
        if spent >= args.max_tokens:
            save("arrêté : budget de tokens atteint")
            print(f"Arrêt : budget atteint ({spent} tokens estimés).")
            return
    save("terminé")
    print(f"Terminé : {spent} tokens estimés. Résultats : {out}")


if __name__ == "__main__":
    main()
