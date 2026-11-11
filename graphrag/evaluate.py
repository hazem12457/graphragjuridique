"""Évaluation du Graph RAG avec le code d'évaluation du RAG vectoriel (mêmes métriques, mêmes questions).

Modes
    advanced            baseline, recalculée ici dans les mêmes conditions (candidats en cache + reranker)
    advanced_norerank   mêmes candidats, ordre RRF, sans reranker : la référence de graph_norerank
    graph               candidats + voisins du graphe + reranker
    graph_norerank      candidats + voisins du graphe, fusion RRF
    graph_linked        comme graph, avec 2 places réservées aux renvois inter-actes de l'article classé 1er
    vector              similarité cosinus seule (RAG vectoriel « classique », sans reranker)

Latence : graph et graph_linked partagent le classement du reranker. Le mode exécuté en second le relit en mémoire,
sa latence affichée ne compte donc que son surcoût propre.

Jeux de questions
    base   24 questions de `app/rag/eval_dataset.py`
    user   `data/eval_questions.jsonl` du projet vectoriel (seules les questions avec gold_articles ; ids préfixés « U- »)

Exemples
    python -u -m graphrag.evaluate
    python -u -m graphrag.evaluate --dataset user --ids U-M01 U-M04 --modes advanced graph
Résultats : results/eval_graph_<horodatage>.json
"""
import argparse
import json
from datetime import datetime

from graphrag import config
from graphrag import retriever as graph

config.use_vector_rag()
from app import config as rag_config  # noqa: E402
from app.dependencies import get_vectordb  # noqa: E402
from app.rag.eval_dataset import EVAL_QUESTIONS  # noqa: E402
from app.rag.pipeline import generate_answer, retrieve, top_articles  # noqa: E402
from app.rag.retriever import advanced_retrieval  # noqa: E402
from app.rag.run_evaluation import evaluate_mode, print_table  # noqa: E402

RESULTS_DIR = config.ROOT / "results"
USER_QUESTIONS = config.VECTOR_RAG_DIR / "data" / "eval_questions.jsonl"
ALL_MODES = ("vector", "advanced", "advanced_norerank", *graph.MODES)


def load_questions(dataset: str, file: str | None = None) -> list[dict]:
    if file:           # JSONL : id, type, question, gold
        with open(file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if dataset == "base":
        return EVAL_QUESTIONS
    questions = []
    with open(USER_QUESTIONS, encoding="utf-8") as f:
        for item in map(json.loads, f):
            if item.get("gold_articles"):
                questions.append({"id": f"U-{item['id']}", "type": item["type"].replace("-", "_"),
                                  "question": item["question"], "gold": item["gold_articles"]})
    return questions


def retrieve_fn_for(mode: str, full_articles: bool = False):
    """`full_articles` : même sélection pour tous les modes, mais le LLM et RAGAS reçoivent les articles complets."""
    if mode in graph.MODES:
        fn = lambda question, k: graph.graph_retrieve(question, k, mode=mode)
    elif mode == "advanced_norerank":
        fn = lambda question, k: top_articles(advanced_retrieval(question, get_vectordb()), k)
    else:              # vector, advanced : `pipeline.retrieve` du projet vectoriel
        fn = lambda question, k: retrieve(question, mode=mode, k=k)
    return (lambda question, k: graph.full_article_context(fn(question, k))) if full_articles else fn


def graph_columns(_item: dict, chunks: list) -> dict:
    return {"graph_added": [c.metadata["article_id"] for c in chunks if c.metadata.get("graph_added")],
            "via": {c.metadata["article_id"]: c.metadata["via"] for c in chunks if c.metadata.get("via")}}


def columns_for(mode: str, generate: bool):
    """Colonnes en plus : traçabilité du graphe et, avec `generate`, la réponse du LLM (sans RAGAS)."""
    def extra(item: dict, chunks: list) -> dict:
        out = graph_columns(item, chunks) if mode in graph.MODES else {}
        if generate:
            out["answer"] = generate_answer(item["question"], chunks)
        return out
    return extra


def print_differences(results: list[dict]) -> None:
    """Questions où le rappel ou le rang réciproque diffère d'un mode à l'autre."""
    modes = [r["mode"] for r in results]
    by_id = {r["id"]: {} for r in results[0]["rows"]}
    for res in results:
        for row in res["rows"]:
            by_id[row["id"]][res["mode"]] = row
    print("\nQuestions où les modes diffèrent (rappel / rang réciproque) :")
    print(f"{'id':<8}" + "".join(f"{m:>20}" for m in modes))
    for qid, rows in by_id.items():
        if len({(rows[m]["recall"], rows[m]["rr"]) for m in modes}) > 1:
            print(f"{qid:<8}" + "".join(f"{rows[m]['recall']:>13.2f} /{rows[m]['rr']:>5.2f}" for m in modes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=ALL_MODES)
    parser.add_argument("--dataset", choices=["base", "user"], default="base")
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=rag_config.TOP_K)
    parser.add_argument("--file", default=None, help="fichier JSONL de questions (remplace --dataset)")
    parser.add_argument("--answers", action="store_true", help="générer les réponses + RAGAS (quota Groq)")
    parser.add_argument("--generate", action="store_true", help="générer les réponses, sans RAGAS (quota Groq)")
    parser.add_argument("--full-articles", action="store_true", help="envoyer au LLM le texte complet des articles sélectionnés")
    parser.add_argument("--types", nargs="+", default=None, help="filtrer : factuelle relationnelle multi_hop")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    questions = [q for q in load_questions(args.dataset, args.file)
                 if (not args.ids or q["id"] in args.ids) and (not args.types or q["type"] in args.types)]
    print(f"{len(questions)} questions ({args.file or args.dataset}) · k={args.k} · modes={args.modes}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_graph_{datetime.now():%Y%m%d_%H%M%S}.json"

    def save(results):
        out.write_text(json.dumps({
            "date": datetime.now().isoformat(timespec="seconds"), "dataset": args.file or args.dataset, "k": args.k,
            "contexte": "articles complets" if args.full_articles else "meilleur chunk par article",
            "embedding_model": rag_config.EMBEDDING_MODEL, "llm": rag_config.LLM_MODEL,
            "reranker": rag_config.LOCAL_RERANK_MODEL,
            "graph": {"n_seeds": graph.N_SEEDS, "max_neighbours": graph.MAX_NEIGHBOURS,
                      "max_relation_articles": graph.MAX_RELATION_ARTICLES},
            "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    results = []
    for mode in args.modes:
        fn = retrieve_fn_for(mode, args.full_articles)
        extra = columns_for(mode, args.generate)
        results.append(evaluate_mode(mode, args.k, questions, args.answers, args.sleep, retrieve_fn=fn, extra_fn=extra,
                                     on_progress=lambda partial: save(results + [partial])))
        save(results)
    print_table(results)
    print_differences(results)
    print(f"Résultats enregistrés : {out}")


if __name__ == "__main__":
    main()
