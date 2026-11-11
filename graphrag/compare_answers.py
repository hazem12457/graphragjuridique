"""Compare les RÉPONSES de deux modes du Graph RAG, sur les questions où leur sélection d'articles diffère.

Le Recall@5 dit si le bon article est sélectionné, pas si le LLM répond correctement. Là où les deux modes sélectionnent les
mêmes articles, le contexte est identique : seules les autres questions sont générées, avec les articles complets.

Critère auxiliaire, automatique et approximatif : part des articles attendus cités dans la réponse (numéro d'article et
nom ou numéro de l'acte présents dans le texte). Il ne juge pas le fond : les réponses sont enregistrées pour être relues.

Cadence réglée sur la limite Groq (8 000 tokens par minute et par modèle).

Usage : python -u -m graphrag.compare_answers --results results/eval_graph_XXX.json [--modes graph graph_linked]
Résultats : results/answers_<mode1>_vs_<mode2>.json
"""
import argparse
import json
import re
import time

from graphrag import config
from graphrag import retriever as graph

config.use_vector_rag()
from app.rag.corpus import get_corpus, normalize  # noqa: E402
from app.rag.pipeline import NOT_AVAILABLE, format_context, generate_answer  # noqa: E402

TOKENS_PER_MINUTE = 8000
ANSWER_TOKENS = 2000        # marge pour la réponse et le raisonnement du modèle


def cites(answer: str, gold: str) -> bool:
    corpus = get_corpus()
    article, doc = corpus.articles[gold], corpus.documents[corpus.articles[gold].celex]
    text = normalize(answer)
    names = [normalize(a) for a in doc.aliases] + re.findall(r"\d{2,4}/\d{1,4}", doc.short_name)
    has_act = any(n in text for n in names)
    has_article = re.search(rf"\b(?:article|art\.?)s?\s+(?:[\d\s,et]*\b)?{re.escape(normalize(article.num))}\b", text)
    return bool(has_act and has_article)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--modes", nargs=2, default=["graph", "graph_linked"])
    args = parser.parse_args()

    data = json.loads((config.ROOT / args.results).read_text(encoding="utf-8"))
    rows = {r["mode"]: {x["id"]: x for x in r["rows"]} for r in data["results"]}
    a, b = args.modes
    differing = [q for q in rows[a] if set(rows[a][q]["retrieved_articles"]) != set(rows[b][q]["retrieved_articles"])]
    print(f"{len(differing)}/{len(rows[a])} questions où {a} et {b} sélectionnent des articles différents")

    out = config.ROOT / "results" / f"answers_{a}_vs_{b}.json"
    results = []
    for i, qid in enumerate(differing, start=1):
        item = rows[a][qid]
        row = {"id": qid, "type": item["type"], "question": item["question"], "gold": item["gold"]}
        for mode in (a, b):
            start = time.time()
            chunks = graph.full_article_context(graph.graph_retrieve(item["question"], 5, mode=mode))
            context_tokens = len(format_context(chunks)) // 4
            answer = generate_answer(item["question"], chunks)
            if answer == NOT_AVAILABLE:
                time.sleep(65)
                answer = generate_answer(item["question"], chunks)
            row[mode] = {"selected": [c.metadata["article_id"] for c in chunks], "context_tokens": context_tokens, "answer": answer,
                         "unavailable": answer == NOT_AVAILABLE, "gold_cited": [g for g in item["gold"] if cites(answer, g)]}
            wait = (context_tokens + ANSWER_TOKENS) / TOKENS_PER_MINUTE * 60 - (time.time() - start)
            if wait > 0:
                time.sleep(wait)
        results.append(row)
        print(f"{i:>2}/{len(differing)} {qid:<8} " + "  ".join(
            f"{m}: {len(row[m]['gold_cited'])}/{len(item['gold'])}{' INDISPONIBLE' if row[m]['unavailable'] else ''}" for m in (a, b)))
        out.write_text(json.dumps({"modes": [a, b], "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    for m in (a, b):
        found = sum(len(r[m]["gold_cited"]) for r in results)
        total = sum(len(r["gold"]) for r in results)
        print(f"{m}: articles attendus cités {found}/{total}, réponses indisponibles {sum(r[m]['unavailable'] for r in results)}")
    print(f"Résultats : {out}")


if __name__ == "__main__":
    main()
