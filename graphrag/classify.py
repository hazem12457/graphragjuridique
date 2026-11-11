"""Classification thématique des articles : le graphe aide-t-il par rapport au texte seul ?

Tâche : multi-étiquettes sur les descripteurs EuroVoc présents dans au moins 10 actes (règle fixée avant les essais).
Les étiquettes existent au niveau de l'acte : chaque article hérite de celles de son acte.

Validation : GroupKFold à 5 plis PAR ACTE (aucun article d'un acte de test à l'entraînement). 15 % des actes
d'entraînement servent à l'arrêt anticipé, au choix du seuil de décision par étiquette (les étiquettes sont rares : un seuil
fixe à 0,5 ne prédit presque rien) et au choix de C pour la régression logistique ; le test n'intervient dans aucun choix.
Cadre transductif : toutes les arêtes et tous les embeddings sont visibles, les étiquettes des actes de test jamais.
La mesure principale est la macro-AP, indépendante de tout seuil.

Modèles (mêmes embeddings e5 de 768 dimensions en entrée)
    logreg            texte seul, régression logistique par étiquette
    logreg_act_mean   article + moyenne des embeddings de son acte (témoin : ce que donne le contexte de l'acte, sans graphe)
    mlp               texte seul, perceptron
    gcn / gat / gt    GCN, GAT, Graph Transformer (attention restreinte aux voisins + encodage de position par marche
                      aléatoire), écrits en PyTorch car torch_geometric n'est pas installé
Graphes : `all` = REFERS_TO + NEXT ; `cross` = uniquement les renvois entre actes différents (ablation : 91 % des renvois
sont internes à un acte, ils ne transmettent donc que du contexte d'acte).

Usage : python -u -m graphrag.classify [--folds 5] [--models ...]
Résultats : results/classification.json
"""
import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from torch import nn
from torch.nn import functional as F

from graphrag import config
from graphrag.db import run

LABEL_MIN_ACTS = 10
SEED = 42
HIDDEN, HEADS, DROPOUT = 256, 4, 0.3
LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE = 0.002, 1e-4, 300, 30
RW_STEPS = 8
LOGREG_C = (1.0, 10.0, 100.0)
RESULTS = config.ROOT / "results" / "classification.json"
ALL_MODELS = ("logreg", "logreg_act_mean", "mlp", "gcn_all", "gat_all", "gt_all", "gcn_cross", "gat_cross", "gt_cross")


# ----------------------------------------------------------------------------------------- données
def load_data():
    arts = run("MATCH (a:Article) RETURN a.id AS id, a.celex AS celex, a.embedding AS e ORDER BY a.id")
    ids = [a["id"] for a in arts]
    index = {a: i for i, a in enumerate(ids)}
    celex = np.array([a["celex"] for a in arts])
    x = np.asarray([a["e"] for a in arts], dtype=np.float32)

    acts_by_label = defaultdict(set)
    for r in run("MATCH (d:Document {in_corpus: true})-[:HAS_SUBJECT]->(e:EuroVoc) RETURN d.celex AS c, e.label AS l"):
        acts_by_label[r["l"]].add(r["c"])
    labels = sorted(l for l, acts in acts_by_label.items() if len(acts) >= LABEL_MIN_ACTS)
    y = np.array([[c in acts_by_label[l] for l in labels] for c in celex], dtype=np.float32)

    refs = [(index[r["a"]], index[r["b"]]) for r in run("MATCH (a:Article)-[:REFERS_TO]->(b:Article) RETURN a.id AS a, b.id AS b")]
    nexts = [(index[r["a"]], index[r["b"]]) for r in run("MATCH (a:Article)-[:NEXT]->(b:Article) RETURN a.id AS a, b.id AS b")]
    edges = {"all": refs + nexts, "cross": [(a, b) for a, b in refs if celex[a] != celex[b]]}
    return ids, celex, x, y, labels, edges


@dataclass
class Graph:
    src: torch.Tensor
    dst: torch.Tensor
    adj: torch.Tensor      # adjacence normalisée (GCN)
    rwse: torch.Tensor     # encodage de position : probabilités de retour d'une marche aléatoire


def build_graph(pairs: list[tuple[int, int]], n: int) -> Graph:
    """Arêtes non orientées, boucles sur chaque nœud (un article sans lien garde ses propres caractéristiques)."""
    undirected = {(a, b) for a, b in pairs if a != b} | {(b, a) for a, b in pairs if a != b}
    a = np.array(sorted(undirected), dtype=np.int64).reshape(-1, 2)
    adj = sp.csr_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n, n))
    deg = np.asarray(adj.sum(1)).ravel()
    walk = sp.diags(np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)) @ adj
    power, rwse = sp.identity(n, format="csr"), []
    for _ in range(RW_STEPS):
        power = power @ walk
        rwse.append(power.diagonal())

    with_loops = adj + sp.identity(n, format="csr")
    d_inv = sp.diags(1.0 / np.sqrt(np.asarray(with_loops.sum(1)).ravel()))
    norm = (d_inv @ with_loops @ d_inv).tocoo()
    loops = np.vstack([a, np.column_stack([np.arange(n), np.arange(n)])])
    return Graph(torch.as_tensor(loops[:, 0]), torch.as_tensor(loops[:, 1]),
                 torch.sparse_coo_tensor(np.vstack([norm.row, norm.col]), norm.data.astype(np.float32), (n, n)).coalesce(),
                 torch.as_tensor(np.column_stack(rwse), dtype=torch.float32))


# ----------------------------------------------------------------------------------------- modèles
def edge_softmax(score: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax des scores d'attention sur les arêtes entrantes de chaque nœud (score : [arêtes, têtes])."""
    index = dst.unsqueeze(-1).expand_as(score)
    maximum = torch.zeros(n, score.size(1)).scatter_reduce(0, index, score.detach(), "amax", include_self=False)
    exp = (score - maximum[dst]).exp()
    return exp / (torch.zeros(n, score.size(1)).index_add(0, dst, exp)[dst] + 1e-16)


class MLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, out))

    def forward(self, x, g):
        return self.net(x)


class GCN(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.input, self.l1, self.l2 = nn.Linear(dim, HIDDEN), nn.Linear(HIDDEN, HIDDEN), nn.Linear(HIDDEN, out)

    def forward(self, x, g):
        h = F.dropout(F.relu(self.input(x)), DROPOUT, self.training)
        h = F.relu(self.l1(torch.sparse.mm(g.adj, h)))
        return self.l2(torch.sparse.mm(g.adj, F.dropout(h, DROPOUT, self.training)))


class GATLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = dim // HEADS
        self.proj = nn.Linear(dim, self.d * HEADS, bias=False)
        self.attn = nn.Parameter(torch.empty(HEADS, 2 * self.d))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, g):
        n = x.size(0)
        h = self.proj(x).view(n, HEADS, self.d)
        pair = torch.cat([h[g.src], h[g.dst]], dim=-1)
        alpha = edge_softmax(F.leaky_relu((pair * self.attn).sum(-1), 0.2), g.dst, n)
        alpha = F.dropout(alpha, DROPOUT, self.training)
        out = torch.zeros(n, HEADS, self.d).index_add(0, g.dst, alpha.unsqueeze(-1) * h[g.src])
        return out.view(n, HEADS * self.d)


class GAT(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.input, self.l1, self.l2, self.head = nn.Linear(dim, HIDDEN), GATLayer(HIDDEN), GATLayer(HIDDEN), nn.Linear(HIDDEN, out)

    def forward(self, x, g):
        h = F.dropout(F.elu(self.input(x)), DROPOUT, self.training)
        h = F.dropout(F.elu(self.l1(h, g)), DROPOUT, self.training)
        return self.head(F.elu(self.l2(h, g)))


class GTLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = dim // HEADS
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out = nn.Linear(self.d * HEADS, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(2 * dim, dim))

    def forward(self, x, g):
        n = x.size(0)
        q, k, v = (f(x).view(n, HEADS, self.d) for f in (self.q, self.k, self.v))
        alpha = edge_softmax((q[g.dst] * k[g.src]).sum(-1) / math.sqrt(self.d), g.dst, n)
        alpha = F.dropout(alpha, DROPOUT, self.training)
        out = torch.zeros(n, HEADS, self.d).index_add(0, g.dst, alpha.unsqueeze(-1) * v[g.src]).view(n, HEADS * self.d)
        x = self.norm1(x + F.dropout(self.out(out), DROPOUT, self.training))
        return self.norm2(x + F.dropout(self.ffn(x), DROPOUT, self.training))


class GraphTransformer(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.input, self.l1, self.l2, self.head = nn.Linear(dim + RW_STEPS, HIDDEN), GTLayer(HIDDEN), GTLayer(HIDDEN), nn.Linear(HIDDEN, out)

    def forward(self, x, g):
        return self.head(self.l2(self.l1(self.input(torch.cat([x, g.rwse], dim=-1)), g), g))


NEURAL = {"mlp": MLP, "gcn": GCN, "gat": GAT, "gt": GraphTransformer}


# ----------------------------------------------------------------------------------------- entraînement et évaluation
def fit_neural(kind: str, x, y, g, train, val, seed: int, max_epochs: int) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    X, Y = torch.as_tensor(x), torch.as_tensor(y)
    model = NEURAL[kind](x.shape[1], y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_loss, wait, epochs = None, float("inf"), 0, 0
    for epochs in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X, g)[train], Y[train])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(model(X, g)[val], Y[val]).item()
        if val_loss < best_loss - 1e-4:
            best_loss, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X, g)).numpy(), epochs


def fit_logreg(x, y, train, val) -> np.ndarray:
    """C choisi par macro-AP sur les actes de validation (embeddings normalisés : C = 1 régulariserait trop)."""
    present = np.where(y[val].sum(0) > 0)[0]
    best_ap, best_probs = -1.0, None
    for c in LOGREG_C:
        probs = OneVsRestClassifier(LogisticRegression(C=c, max_iter=3000)).fit(x[train], y[train]).predict_proba(x)
        ap = average_precision_score(y[val][:, present], probs[val][:, present], average="macro")
        if ap > best_ap:
            best_ap, best_probs = ap, probs
    return best_probs


def best_thresholds(y_val, probs_val) -> np.ndarray:
    """Seuil de chaque étiquette : celui qui maximise son F1 sur les actes de validation (0,5 si aucun positif)."""
    grid = np.arange(0.05, 0.96, 0.05)
    thresholds = np.full(y_val.shape[1], 0.5)
    for j in range(y_val.shape[1]):
        if y_val[:, j].sum() > 0:
            f1 = [f1_score(y_val[:, j], probs_val[:, j] >= t, zero_division=0) for t in grid]
            thresholds[j] = grid[int(np.argmax(f1))]
    return thresholds


def scores(y_true, probs, celex, thresholds) -> dict:
    present = np.where(y_true.sum(0) > 0)[0]
    pred = probs >= thresholds
    acts = sorted(set(celex))
    act_true = np.array([y_true[celex == c].max(0) for c in acts])
    act_pred = np.array([probs[celex == c].mean(0) >= thresholds for c in acts])
    return {"micro_f1": f1_score(y_true, pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y_true, pred, average="macro", labels=present, zero_division=0),
            "macro_ap": average_precision_score(y_true[:, present], probs[:, present], average="macro"),
            "act_micro_f1": f1_score(act_true, act_pred, average="micro", zero_division=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5, help="nombre de plis évalués (sur 5)")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    ids, celex, x, y, labels, edges = load_data()
    print(f"{len(ids)} articles, {len(set(celex))} actes, {len(labels)} étiquettes : {labels}")
    print(f"part d'articles positifs par étiquette : {np.round(y.mean(0), 2).tolist()}")
    graphs = {name: build_graph(pairs, len(ids)) for name, pairs in edges.items()}
    print({name: int(g.src.numel() - len(ids)) // 2 for name, g in graphs.items()}, "arêtes non orientées")

    act_mean = {c: x[celex == c].mean(0) for c in set(celex)}
    x_act = np.hstack([x, np.array([act_mean[c] for c in celex])])

    per_fold = defaultdict(list)
    for fold, (train_all, test) in enumerate(GroupKFold(n_splits=5).split(x, y, groups=celex)):
        if fold >= args.folds:
            break
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        tr, va = next(inner.split(train_all, groups=celex[train_all]))
        train, val = train_all[tr], train_all[va]
        assert not set(celex[test]) & set(celex[train_all])
        for name in args.models:
            t0, epochs = time.time(), 0
            if name == "logreg":
                probs = fit_logreg(x, y, train, val)
            elif name == "logreg_act_mean":
                probs = fit_logreg(x_act, y, train, val)
            else:
                kind, _, edge_set = name.partition("_")
                probs, epochs = fit_neural(kind, x, y, graphs[edge_set or "all"], train, val, SEED, args.epochs)
            res = scores(y[test], probs[test], celex[test], best_thresholds(y[val], probs[val]))
            per_fold[name].append({k: float(v) for k, v in res.items()})
            print(f"pli {fold} {name:<16} micro-F1={res['micro_f1']:.3f} macro-F1={res['macro_f1']:.3f} "
                  f"macro-AP={res['macro_ap']:.3f} F1-acte={res['act_micro_f1']:.3f} époques={epochs} ({time.time() - t0:.0f}s)")

    summary = {m: {k: {"mean": round(float(np.mean([f[k] for f in fs])), 3), "std": round(float(np.std([f[k] for f in fs])), 3)}
                   for k in fs[0]} for m, fs in per_fold.items()}
    print(f"\n{'modèle':<18}{'micro-F1':>16}{'macro-F1':>16}{'macro-AP':>16}{'F1 acte':>16}")
    for m, s in summary.items():
        print(f"{m:<18}" + "".join(f"{s[k]['mean']:>9.3f} ±{s[k]['std']:.3f}" for k in ("micro_f1", "macro_f1", "macro_ap", "act_micro_f1")))
    if not args.no_save:
        RESULTS.write_text(json.dumps({"labels": labels, "folds_evaluated": args.folds, "group": "acte", "seed": SEED,
                                       "config": {"hidden": HIDDEN, "heads": HEADS, "dropout": DROPOUT, "lr": LR, "max_epochs": args.epochs, "patience": PATIENCE},
                                       "summary": summary, "per_fold": per_fold}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Résultats : {RESULTS}")


if __name__ == "__main__":
    main()
