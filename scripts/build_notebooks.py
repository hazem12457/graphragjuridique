"""Construit et exécute les notebooks de `notebooks/` sans Jupyter (non installé sur le poste de travail).

Chaque cellule de code est exécutée dans l'ordre, dans un espace de noms partagé par notebook, depuis le dossier
`notebooks/` (comme le ferait Jupyter). Ses sorties réelles (affichages et dernière expression, tableaux pandas en HTML)
sont écrites dans le fichier .ipynb. Les notebooks restent exécutables tels quels dans Jupyter ou VS Code.

Aucun appel au LLM : les réponses montrées sont celles enregistrées dans `results/` par les scripts d'évaluation.
Pré-requis : Neo4j démarré (les notebooks 1 à 4 interrogent la base).

Usage (depuis la racine du projet) : $env:PYTHONPATH="."; python scripts/build_notebooks.py [01 02 ...]
"""
import ast
import contextlib
import io
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

SETUP = r'''import json
import os
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
# candidats de recherche déjà calculés (MultiQuery, HyDE) : aucune requête au LLM
os.environ.setdefault("SEED_CACHE_PATH", str(ROOT / "data" / "cache" / "user_test_candidates.json"))

import pandas as pd
pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 220)

from graphrag import config
config.use_vector_rag()
DATA = config.VECTOR_RAG_DIR / "data" / "eurlex"
RESULTS = ROOT / "results"
print("Projet :", ROOT)
print("Données du projet vectoriel :", DATA)'''


def intro(title: str, body: str) -> str:
    return (f"# {title}\n\n{body}\n\n"
            "_Notebook rejoué avec le code du projet (`graphrag/`, et `app/` du projet vectoriel). Ses sorties sont réelles : "
            f"il a été exécuté le {datetime.now():%d/%m/%Y à %H:%M} par `scripts/build_notebooks.py`. "
            "Aucun appel au LLM n'est fait ici ; Neo4j doit être démarré._")


NOTEBOOKS = {
    "01_donnees_et_graphe": [
        ("md", intro("1. Données et graphe de connaissances",
                     "Corpus EUR-Lex (100 actes du numérique), relations entre actes et renvois entre articles, graphe Neo4j "
                     "et contrôle manuel de la qualité des liens extraits.")),
        ("code", SETUP),
        ("md", "## 1.1 Le corpus\n\nCollecté par `collect_eurlex.py` (projet vectoriel) : textes français par l'API REST Cellar, "
               "métadonnées et relations par SPARQL. L'unité de travail est l'article."),
        ("code", r'''articles = pd.read_json(DATA / "articles.jsonl", lines=True)
documents = pd.read_csv(DATA / "documents.csv", dtype=str)
print(len(documents), "actes ;", len(articles), "articles")
articles["doc_type"].value_counts().rename("articles par type d'acte")'''),
        ("code", r'''longueur = articles["text"].str.len()
longueur.describe(percentiles=[0.5, 0.9]).round(0).rename("longueur d'un article (caractères)")'''),
        ("code", r'''relations = pd.read_csv(DATA / "relations.csv", dtype=str)
refs = pd.read_csv(DATA / "article_refs.csv", dtype=str)
print("Relations acte -> acte dans le corpus (Cellar) :", relations["type"].value_counts().to_dict())
print("Renvois article -> article extraits par regex :", len(refs))
print("  dont vers un article du corpus :", int((refs["target_in_corpus"] == "True").sum()))'''),
        ("md", "## 1.2 Le graphe Neo4j\n\nConstruit par `graphrag/build_graph.py`. Comptes lus directement dans la base."),
        ("code", r'''from graphrag.db import run

pd.DataFrame(run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS noeuds ORDER BY noeuds DESC"))'''),
        ("code", r'''pd.DataFrame(run("""MATCH (a)-[r]->(b)
    RETURN type(r) AS arete, labels(a)[0] AS de, labels(b)[0] AS vers, count(*) AS n ORDER BY n DESC"""))'''),
        ("code", r'''inter = run("MATCH (a:Article)-[:REFERS_TO]->(b:Article) WHERE a.celex <> b.celex RETURN count(*) AS n")[0]["n"]
print("Renvois REFERS_TO entre actes différents :", inter)'''),
        ("md", "### Deux exemples de chemins utilisés par le Graph RAG"),
        ("code", r'''# Quel article modifie le règlement 2019/1020 ? (arête Article -> Document)
pd.DataFrame(run("""MATCH (a:Article)-[r:AMENDS]->(d:Document {celex: '32019R1020'})
    RETURN a.id AS article, a.title AS titre, r.evidence AS preuve"""))'''),
        ("code", r'''# L'article 28 du DSA (mineurs) renvoie à l'article 4 du RGPD (définition du profilage)
pd.DataFrame(run("""MATCH (a:Article {id: '32022R2065_art_28'})-[:REFERS_TO]->(b:Article)
    RETURN a.title AS de, b.id AS vers, b.title AS titre"""))'''),
        ("md", "## 1.3 Contrôle qualité des liens extraits par règles\n\n50 arêtes tirées au hasard (graine 42) par "
               "`graphrag/sample_edges.py`, puis annotées à la main dans `results/edge_sample.csv` (colonne `correct`)."),
        ("code", r'''sample = pd.read_csv(RESULTS / "edge_sample.csv", sep=";")
qualite = sample.groupby("kind")["correct"].agg(corrects="sum", total="count")
qualite["precision"] = (qualite["corrects"] / qualite["total"]).round(3)
print("Total :", int(sample["correct"].sum()), "/", len(sample), "=", round(sample["correct"].mean(), 3))
qualite'''),
        ("code", r'''sample.loc[sample["correct"] == 0, ["kind", "source", "target", "erreur"]]'''),
    ],
    "02_clusters_thematiques": [
        ("md", intro("2. Clusters thématiques (Louvain)",
                     "91 % des renvois restent dans un même acte : sur les renvois seuls, Louvain retrouve les actes. "
                     "On compare ce graphe « structure » au graphe « thèmes » (renvois + 10 plus proches voisins sémantiques), "
                     "retenu pour le livrable. Les clusters sont recalculés ici avec le code de `graphrag/clusters.py`, sans rien écrire dans Neo4j.")),
        ("code", SETUP),
        ("code", r'''from graphrag import clusters as cl

ids, celex, titles, emb, refs, nexts, subjects = cl.load()
internes = sum(celex[a] == celex[b] for a, b in refs)
print(len(ids), "articles ;", len(refs), "renvois, dont", f"{internes / len(refs):.0%}", "à l'intérieur d'un acte")'''),
        ("code", r'''structure = {tuple(sorted(e)): 1.0 for e in refs + nexts}
themes = cl.knn_edges(ids, emb)                       # 10 plus proches voisins, poids = cosinus
for a, b in refs:
    key = tuple(sorted((a, b)))
    themes[key] = themes.get(key, 0.0) + cl.REFERS_WEIGHT

comm_s, mod_s = cl.louvain(ids, structure)
comm_t, mod_t = cl.louvain(ids, themes)
pd.DataFrame({
    "structure (REFERS_TO + NEXT)": {"modularite": round(mod_s, 3), **cl.describe(comm_s, celex, ids)},
    "themes (REFERS_TO + 10 voisins), retenu": {"modularite": round(mod_t, 3), **cl.describe(comm_t, celex, ids)},
})'''),
        ("md", "## Les 25 clusters retenus\n\nÉtiquette : descripteurs EuroVoc sur-représentés (couverture × log(1 + lift)), "
               "ou « Clauses formelles » si au moins 40 % des titres d'articles sont des clauses types."),
        ("code", r'''report = json.loads((RESULTS / "clusters.json").read_text(encoding="utf-8"))
print("Recalcul identique au fichier enregistré :",
      report["themes"]["n_clusters"] == len(comm_t) and report["themes"]["modularite"] == round(mod_t, 3))
pd.DataFrame(report["clusters"])[["id", "size", "n_acts", "label"]]'''),
        ("code", r'''from graphrag.db import run

pd.DataFrame(run("""MATCH (c:Cluster) OPTIONAL MATCH (a:Article)-[:IN_CLUSTER]->(c)
    RETURN count(DISTINCT c) AS clusters, count(a) AS articles_rattaches""")).assign(
    articles_sans_cluster=run("MATCH (a:Article) WHERE NOT (a)-[:IN_CLUSTER]->() RETURN count(a) AS n")[0]["n"])'''),
        ("code", r'''# Cluster « intelligence artificielle » : ses renvois internes (requête utilisée pour la capture Neo4j)
pd.DataFrame(run("""MATCH (c:Cluster {id: 10})<-[:IN_CLUSTER]-(a:Article)-[r:REFERS_TO]->(b:Article)-[:IN_CLUSTER]->(c)
    RETURN c.label AS cluster, count(r) AS renvois_internes, count(DISTINCT a.celex) AS actes"""))'''),
    ],
    "03_rag_vectoriel_et_graph_rag": [
        ("md", intro("3. RAG vectoriel et Graph RAG",
                     "Sélection des 5 articles par chaque mode, sur deux questions du test à l'aveugle. Les candidats MultiQuery et HyDE "
                     "sont relus dans le cache (aucune requête au LLM) ; le reranker `bge-reranker-v2-m3` tourne en local.")),
        ("code", SETUP),
        ("code", r'''from app import config as rag_config
from app.rag.prompts import ANSWER_TEMPLATE

print("Embeddings :", rag_config.EMBEDDING_MODEL)
print("Reranker   :", rag_config.LOCAL_RERANK_MODEL)
print("LLM        :", rag_config.LLM_MODEL, "| modèle rapide :", rag_config.FAST_LLM_MODEL)
print("Articles dans le contexte :", rag_config.TOP_K, "| candidats :", rag_config.CANDIDATES_K)'''),
        ("md", "## Sélection des articles, mode par mode\n\n`vector` : cosinus seul. `advanced` (baseline) : MultiQuery + HyDE + BM25, "
               "fusion RRF, reranker. `graph_linked` : baseline + graphe, 2 places réservées aux renvois inter-actes de l'article classé 1er."),
        ("code", r'''from graphrag.evaluate import retrieve_fn_for

def selection(question, modes=("vector", "advanced", "graph_linked")):
    lignes = []
    for mode in modes:
        for rang, c in enumerate(retrieve_fn_for(mode)(question, 5), start=1):
            m = c.metadata
            lignes.append({"mode": mode, "rang": rang, "article": f"{m['doc_label']}, art. {m['article_num']}",
                           "titre": m.get("article_title"), "ajoute_par_le_graphe": m.get("graph_added", False),
                           "chemin": (m.get("via") or [""])[0]})
    return pd.DataFrame(lignes)

selection("Quel acte du corpus modifie le règlement 2019/1020 ?")'''),
        ("code", r'''selection("Peut-on montrer à un mineur de la publicité fondée sur le profilage ?")'''),
        ("md", "## Contexte envoyé au LLM\n\nLe meilleur chunk de chaque article ne contient pas toujours le paragraphe qui répond. "
               "Tous les modes envoient donc l'article complet (6 000 caractères au plus) ; la sélection ne change pas."),
        ("code", r'''from graphrag import retriever as graph

question = "Peut-on montrer à un mineur de la publicité fondée sur le profilage ?"
chunks = graph.graph_retrieve(question, 5, mode="graph_linked")
complets = graph.full_article_context(chunks)
pd.DataFrame({"article": [c.metadata["article_id"] for c in chunks],
              "caracteres_meilleur_chunk": [len(c.page_content) for c in chunks],
              "caracteres_article_complet": [len(c.page_content) for c in complets]})'''),
        ("code", r'''print(complets[0].page_content[:900])'''),
        ("md", "## Le prompt partagé par tous les modes"),
        ("code", r'''print(ANSWER_TEMPLATE)'''),
        ("md", "## Réponses générées (enregistrées)\n\nProduites par `scripts/answer_user_questions.py` avec `gpt-oss-120b`, "
               "même chaîne que l'interface. Relues ensuite à la main contre le texte des articles."),
        ("code", r'''answers = json.loads((RESULTS / "answers_utilisateur_20261110_015420.json").read_text(encoding="utf-8"))
c2 = next(r for r in answers["rows"] if r["id"] == "C2")
print("Question :", c2["question"], "\nRéponse attendue :", c2["reference"])
for mode in ("advanced", "graph_linked"):
    print(f"\n--- {mode} (articles : {c2[mode]['selected']})\n{c2[mode]['answer']}")'''),
    ],
    "04_classification_gnn": [
        ("md", intro("4. Classification thématique : texte seul face aux GNN",
                     "Multi-étiquettes sur les 10 descripteurs EuroVoc présents dans au moins 10 actes. Validation GroupKFold à 5 plis par acte. "
                     "Résultats complets lus dans `results/classification.json` (`graphrag/classify.py`), puis pli 0 rejoué en direct pour 5 modèles.")),
        ("code", SETUP),
        ("code", r'''clf = json.loads((RESULTS / "classification.json").read_text(encoding="utf-8"))
print("Étiquettes :", clf["labels"])
print("Configuration :", clf["config"])
tableau = pd.DataFrame({m: {k: f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in s.items()} for m, s in clf["summary"].items()}).T
tableau.sort_values("macro_ap", ascending=False)'''),
        ("md", "## Les trois GNN (écrits en PyTorch, torch_geometric n'étant pas installable)"),
        ("code", r'''from graphrag import classify as C

print(C.GCN(768, 10))
print(C.GAT(768, 10))
print(C.GraphTransformer(768, 10))'''),
        ("md", "## Pli 0 rejoué\n\nMêmes données, même découpage, même graine que le script : on vérifie que les chiffres enregistrés se reproduisent."),
        ("code", r'''from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ids, celex, x, y, labels, edges = C.load_data()
graphs = {name: C.build_graph(pairs, len(ids)) for name, pairs in edges.items()}
train_all, test = next(GroupKFold(n_splits=5).split(x, y, groups=celex))
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=C.SEED).split(train_all, groups=celex[train_all]))
train, val = train_all[tr], train_all[va]
print(len(train), "articles d'entraînement,", len(val), "de validation,", len(test), "de test ;",
      "actes de test vus à l'entraînement :", len(set(celex[test]) & set(celex[train_all])))'''),
        ("code", r'''lignes = []
for name in ["logreg", "mlp", "gcn_all", "gat_all", "gt_cross"]:
    if name == "logreg":
        probs = C.fit_logreg(x, y, train, val)
    else:
        kind, _, edge_set = name.partition("_")
        probs, _ = C.fit_neural(kind, x, y, graphs[edge_set or "all"], train, val, C.SEED, C.MAX_EPOCHS)
    res = C.scores(y[test], probs[test], celex[test], C.best_thresholds(y[val], probs[val]))
    lignes.append({"modele": name, "macro_AP_rejoue": round(res["macro_ap"], 3),
                   "macro_AP_enregistre": round(clf["per_fold"][name][0]["macro_ap"], 3),
                   "micro_F1_rejoue": round(res["micro_f1"], 3)})
pd.DataFrame(lignes)'''),
        ("md", "**Lecture :** le meilleur modèle n'utilise aucune arête (régression logistique + moyenne de l'acte). À entraînement "
               "identique, Graph Transformer et GAT dépassent le MLP ; le GCN non. Hypothèse non testée : les étiquettes décrivent "
               "l'acte et 91 % des renvois restent dans l'acte."),
    ],
    "05_evaluation": [
        ("md", intro("5. Évaluation : recherche, réponses, RAGAS",
                     "Tous les chiffres de la présentation, recalculés à partir des fichiers de résultats.")),
        ("code", SETUP),
        ("code", r'''def resume(nom, metrique):
    d = json.loads((RESULTS / nom).read_text(encoding="utf-8"))
    lignes = [{"mode": r["mode"], "sous_ensemble": f"{k} ({s['n']})", metrique: s[metrique]}
              for r in d["results"] for k, s in r["summary"].items()]
    return pd.DataFrame(lignes).pivot(index="sous_ensemble", columns="mode", values=metrique).round(3)

print("60 questions : Recall@5")
resume("eval_graph_20261108_215757.json", "recall@k")'''),
        ("code", r'''resume("eval_graph_20261108_215757.json", "mrr")'''),
        ("code", r'''print("17 questions à l'aveugle : Recall@5")
resume("eval_graph_20261110_005139.json", "recall@k")'''),
        ("md", "## Test des signes\n\nQuestion par question, `graph_linked` contre la baseline `advanced` (rappel meilleur, moins bon, égal)."),
        ("code", r'''from scipy.stats import binomtest

def signes(nom, a="advanced", b="graph_linked"):
    d = json.loads((RESULTS / nom).read_text(encoding="utf-8"))
    rec = {r["mode"]: {x["id"]: x["recall"] for x in r["rows"]} for r in d["results"]}
    mieux = sum(rec[b][q] > rec[a][q] for q in rec[a])
    moins = sum(rec[b][q] < rec[a][q] for q in rec[a])
    return {"mieux": mieux, "moins_bien": moins, "egal": len(rec[a]) - mieux - moins,
            "p_bilateral": round(binomtest(mieux, mieux + moins, 0.5).pvalue, 4)}

pd.DataFrame({"60 questions": signes("eval_graph_20261108_215757.json"),
              "17 questions à l'aveugle": signes("eval_graph_20261110_005139.json")}).T'''),
        ("md", "## Réponses générées\n\nArticles attendus sélectionnés et cités, calculés automatiquement ; verdict issu de la relecture "
               "manuelle contre le texte de loi (détail et erreurs dans le README, §9.5)."),
        ("code", r'''VERDICT = {  # relecture manuelle : baseline, Graph RAG
    "P1": ("juste", "incomplète"), "P2": ("juste", "juste"), "A2": ("juste", "juste"), "A3": ("juste", "juste, 1 erreur de référence"),
    "B4": ("juste", "juste"), "B5": ("juste, 1 erreur de numéro", "juste"), "B6": ("juste", "juste"),
    "C3": ("juste", "juste, 1 erreur de détail"), "C5": ("juste", "juste"), "C2": ("ne permet pas de répondre", "juste"),
    "D1": ("refus", "refus"), "D2": ("refus", "refus"), "D3": ("refus", "refus"), "D4": ("refus", "refus")}
lignes = []
for nom in ("answers_utilisateur_20261110_010908.json", "answers_utilisateur_20261110_015420.json"):
    for r in json.loads((RESULTS / nom).read_text(encoding="utf-8"))["rows"]:
        n = len(r["gold"])
        lignes.append({"id": r["id"], "question": r["question"][:70],
                       "baseline_selectionnes": f"{len(r['advanced']['gold_selected'])}/{n}",
                       "graph_selectionnes": f"{len(r['graph_linked']['gold_selected'])}/{n}",
                       "verdict_baseline": VERDICT[r["id"]][0], "verdict_graph": VERDICT[r["id"]][1]})
pd.DataFrame(lignes)'''),
        ("md", "## Citations entre guillemets\n\nChaque fragment cité entre « » (coupé aux « … ») est-il présent mot pour mot dans les articles transmis ?"),
        ("code", r'''import re
import unicodedata

articles = {a["id"]: a for a in pd.read_json(DATA / "articles.jsonl", lines=True).to_dict("records")}

def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("’", "'").replace(" ", " ")
    return re.sub(r"\s+", " ", s).lower()

data = json.loads((RESULTS / "answers_utilisateur_20261110_010908.json").read_text(encoding="utf-8"))
compte = {}
for mode in data["modes"]:
    total = trouves = 0
    for row in data["rows"]:
        ctx = norm(" ".join(articles[i]["article_title"] + " " + articles[i]["text"] for i in row[mode]["selected"]))
        for q in re.findall(r"«([^«»]{15,})»", row[mode]["answer"]):
            for frag in re.split(r"…|\.\.\.", q):
                f = norm(frag).strip(" ,.;:")
                if len(f) >= 15:
                    total += 1
                    trouves += f in ctx
    compte[mode] = f"{trouves}/{total}"
compte'''),
        ("md", "## RAGAS (juge gpt-oss-20b)\n\nSur 3 questions à contexte court seulement (B4, C3, C5), à cause de la limite Groq de 8 000 tokens par minute."),
        ("code", r'''ragas = json.loads((RESULTS / "ragas_utilisateur_20261110_013124.json").read_text(encoding="utf-8"))
tab = pd.DataFrame(ragas["resultats"]).drop(columns="meme_selection")
display_tab = tab.pivot(index="id", columns="mode", values=["faithfulness", "answer_relevancy", "context_recall"])
display_tab.loc["moyenne"] = display_tab.mean().round(3)
display_tab'''),
    ],
}


def to_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def display_data(value) -> dict | None:
    if value is None:
        return None
    data = {"text/plain": to_lines(repr(value))}
    html = getattr(value, "to_html", None)
    if callable(html):
        data["text/html"] = to_lines(html())
    elif hasattr(value, "to_frame"):
        data["text/html"] = to_lines(value.to_frame().to_html())
    return data


def run_cell(source: str, namespace: dict) -> tuple[list, bool]:
    outputs, buffer, ok, value = [], io.StringIO(), True, None
    try:
        tree = ast.parse(source)
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        body = ast.Module(body=tree.body[:-1] if last else tree.body, type_ignores=[])
        with contextlib.redirect_stdout(buffer):
            exec(compile(body, "<cell>", "exec"), namespace)
            if last:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
    except Exception:
        ok = False
        outputs.append({"output_type": "error", "ename": "Error", "evalue": "", "traceback": traceback.format_exc().splitlines()})
    printed = buffer.getvalue()
    if printed:
        outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": to_lines(printed)})
    return outputs, ok, value


def build(name: str, cells: list) -> None:
    os.chdir(NB_DIR)
    namespace = {"__name__": "__main__"}
    nb_cells, count = [], 0
    for kind, source in cells:
        cell = {"id": uuid.uuid4().hex[:8], "metadata": {}, "source": to_lines(source)}
        if kind == "md":
            nb_cells.append({**cell, "cell_type": "markdown"})
            continue
        count += 1
        t0 = time.time()
        outputs, ok, value = run_cell(source, namespace)
        data = display_data(value) if ok else None
        if data:
            outputs.append({"output_type": "execute_result", "execution_count": count, "data": data, "metadata": {}})
        nb_cells.append({**cell, "cell_type": "code", "execution_count": count, "outputs": outputs})
        print(f"  {name} cellule {count} : {'ok' if ok else 'ERREUR'} ({time.time() - t0:.0f} s)", flush=True)
        if not ok:
            print("\n".join(outputs[-1]["traceback"]), flush=True)
            raise SystemExit(f"Arrêt : erreur dans {name}, cellule {count}")
    nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3 (rag-juridique)", "language": "python"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}}}
    (NB_DIR / f"{name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Écrit : notebooks/{name}.ipynb", flush=True)


def main():
    NB_DIR.mkdir(exist_ok=True)
    wanted = sys.argv[1:]
    for name, cells in NOTEBOOKS.items():
        if not wanted or any(name.startswith(w) for w in wanted):
            build(name, cells)


if __name__ == "__main__":
    main()
