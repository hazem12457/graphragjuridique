# Graph RAG juridique — droit européen des données et du numérique

Projet S3 ESPRIT, module *Advanced Learning for Text and Graph Data* : **classification et question-answering sur les données
juridiques**. Validation le 10 novembre 2026.

Ce README explique **tout le projet**, dans l'ordre où il a été construit : les données, le RAG vectoriel (la référence), le graphe,
les clusters, le Graph RAG, la classification par GNN, l'évaluation et les résultats. Chaque chiffre vient d'une exécution dont le
fichier de résultats est cité.

---

## Sommaire

0. [En bref](#0-en-bref)
1. [Vue d'ensemble : deux projets, une comparaison](#1-vue-densemble--deux-projets-une-comparaison)
2. [Les données](#2-les-données)
3. [Le RAG vectoriel (la référence)](#3-le-rag-vectoriel-la-référence)
4. [Les modèles et LLM utilisés](#4-les-modèles-et-llm-utilisés)
5. [Le graphe de connaissances (Neo4j)](#5-le-graphe-de-connaissances-neo4j)
6. [Les clusters thématiques (Louvain)](#6-les-clusters-thématiques-louvain)
7. [Le Graph RAG](#7-le-graph-rag)
8. [La classification : texte seul contre GNN](#8-la-classification--texte-seul-contre-gnn)
9. [L'évaluation : métriques, protocole, résultats](#9-lévaluation--métriques-protocole-résultats)
10. [Comment les deux objectifs sont atteints](#10-comment-les-deux-objectifs-sont-atteints)
11. [Limites](#11-limites)
12. [Démonstration](#12-démonstration)
13. [Installation et commandes](#13-installation-et-commandes)
14. [Organisation du dépôt](#14-organisation-du-dépôt)
15. [Glossaire](#15-glossaire)
16. [Questions probables du jury](#16-questions-probables-du-jury)
17. [État d'avancement](#17-état-davancement)

---

## 0. En bref

**Question de recherche :** la structure du droit européen (renvois entre articles, modifications, abrogations) améliore-t-elle
(1) la réponse à des questions et (2) la classification thématique, par rapport à une approche sans graphe, **à conditions égales** ?

**Les deux livrables demandés**

| Livrable | Résultat | Preuve |
|---|---|---|
| Graphe de relations entre articles de loi, avec clusters thématiques | **Atteint** | 3 941 articles, 7 047 renvois (`REFERS_TO`), 94 % de liens corrects sur 50 vérifiés ; 25 clusters Louvain, tous les articles rattachés |
| Modèle capable de répondre à des questions spécifiques sur la législation | **Atteint** | Réponses citant l'acte et l'article ; sur les réponses relues à la main, 9 justes sur 10 pour chaque système, 4 pièges refusés sans invention |

**Ce que disent les mesures**

| Volet | Résultat |
|---|---|
| Recherche d'articles, 60 questions | Recall@5 : baseline 0,858 → Graph RAG **0,922** ; multi-hop 0,625 → **0,767** ; non significatif (p ≈ 0,065) |
| Recherche d'articles, 17 questions à l'aveugle | Recall@5 : 0,632 → **0,917** ; multi-hop 0,440 → **0,869** ; p ≈ 0,004, mais questions favorables au graphe par construction |
| Qualité des réponses | **Pas de gain mesurable** : mêmes scores, échecs différents (le graphe gagne une question, en perd une) |
| Classification thématique | **Pas de gain** : la régression logistique (0,420, et 0,537 avec la moyenne de l'acte) reste devant le meilleur GNN (0,357) |

**En une phrase :** le graphe aide là où la structure compte (trouver l'article qui modifie un acte, suivre un renvoi), pas partout.

---

## 1. Vue d'ensemble : deux projets, une comparaison

Le travail est réparti en **deux dépôts**, pour que la référence reste intacte :

| Dépôt | Contenu | Rôle |
|---|---|---|
| `..\Advenced-Rag-Agents-main\Advenced-Rag-Agents-main` | RAG vectoriel, agents, API FastAPI, données collectées | **Référence (baseline)**, livrable autonome |
| `GraphRAG-juridique` (ce dépôt) | Graphe Neo4j, clusters, Graph RAG, GNN, évaluation, interface commune | Ce qu'apporte le graphe |

Ce dépôt **lit** les données du projet vectoriel par leur chemin (`VECTOR_RAG_DIR`) et **importe** son code (`config.use_vector_rag()`) :
même découpage, mêmes embeddings, même prompt, même LLM, mêmes métriques. Rien n'est recopié, donc rien ne peut diverger.

```
                         ┌─────────────── projet vectoriel ───────────────┐
 Cellar (EUR-Lex) ──►    │ articles.jsonl ─► chunks ─► e5 ─► ChromaDB       │
                         │                  └► BM25                         │
                         │ recherche vector / hybrid / advanced ─► LLM      │
                         └───────────────▲──────────────────────────────────┘
                                         │ mêmes données, même code de génération
                         ┌───────────────┴──── ce dépôt ───────────────────┐
                         │ Neo4j : Document, Article, EuroVoc, Cluster      │
                         │ Louvain ─► 25 clusters                           │
                         │ Graph RAG : candidats advanced + graphe ─► LLM   │
                         │ GNN : GCN, GAT, Graph Transformer                │
                         │ Évaluation, interface Streamlit                  │
                         └──────────────────────────────────────────────────┘
```

---

## 2. Les données

### 2.1 Collecte (`scripts/collect_eurlex.py` du projet vectoriel)

Le site eur-lex.europa.eu bloque les scripts. On passe donc par **Cellar**, le dépôt de l'Office des publications de l'UE :

| Source | Ce qu'on récupère |
|---|---|
| API REST Cellar (`http://publications.europa.eu/resource/celex/<CELEX>`) | Texte officiel en français (XHTML), découpé en articles |
| SPARQL Cellar | Métadonnées (titre, date, type), descripteurs **EuroVoc**, relations entre actes : `CITES`, `AMENDS`, `REPEALS`, `BASED_ON` |
| Expressions régulières sur le texte | Renvois d'article à article (« l'article 33 du règlement (UE) 2016/679 ») |
| `scripts/collect_external_relations.py` (ce dépôt) | Relations vers les actes **hors corpus** et leur titre officiel |

Un **CELEX** est l'identifiant officiel d'un acte : `32016R0679` = 3 (législation) + 2016 + R (règlement) + 0679 = le RGPD.

### 2.2 Le corpus

| Élément | Nombre |
|---|---|
| Actes du corpus (volet « protection des données et numérique ») | 100 (RGPD, AI Act, DSA, DMA, NIS2, Data Act, DORA, CRA, ePrivacy…) |
| Articles | 3 941 (3 143 de règlements, 782 de directives, 16 de décisions) |
| Chunks | 9 290 |
| Renvois article → article extraits | 8 023 (7 047 arêtes une fois gardés ceux qui visent un article du corpus) |
| Relations entre actes du corpus | 639 |
| Relations vers des actes hors corpus | 732, vers 668 actes |

**Périmètre :** seuls les articles du dispositif sont indexés, ni les considérants ni les annexes (les annexes de l'AI Act sont absentes).

### 2.3 Fichiers (dans `..\Advenced-Rag-Agents-main\Advenced-Rag-Agents-main\data\eurlex\`)

| Fichier | Contenu |
|---|---|
| `articles.jsonl` | 3 941 articles : `id` (`<celex>_art_<num>`), `celex`, `doc_title`, `doc_type`, `date`, `article_num`, `article_title`, `text`, `eurovoc` |
| `documents.csv` | 100 actes : `celex`, `type`, `date`, `title`, `eurovoc`, `n_articles` |
| `relations.csv` | Relations acte → acte (Cellar) |
| `article_refs.csv` | Renvois article → article ; seuls ceux avec `target_in_corpus = True` deviennent des arêtes |

Dans ce dépôt : `data/eurlex_external/` (relations et actes hors corpus).

---

## 3. Le RAG vectoriel (la référence)

Un **RAG** (*Retrieval-Augmented Generation*) cherche d'abord des passages pertinents, puis demande à un LLM de répondre **à partir
de ces passages seulement**. Le LLM ne répond donc pas de mémoire et peut citer ses sources.

### 3.1 Découpage (`app/rag/chunking.py`)

- **L'article est l'unité de découpage**, pas la page : c'est l'unité naturelle d'un texte de loi.
- Un article de 1 500 caractères au plus donne 1 chunk (2 025 articles sur 3 941). Un article plus long est sous-découpé en morceaux
  d'environ 1 500 caractères avec **200 caractères de recouvrement**, en coupant de préférence avant un paragraphe numéroté, un point
  `a)`, une fin de phrase.
- **Chaque chunk commence par un en-tête** : `Règlement (UE) 2016/679 (RGPD) — Article 83 — Conditions générales…`. Sans lui, un
  morceau qui commence par « 2. Le paragraphe 1 ne s'applique pas… » serait impossible à rattacher à son acte.
- Chaque chunk reste sous la limite de 512 tokens du modèle d'embeddings.

### 3.2 Embeddings et index

- Modèle `intfloat/multilingual-e5-base`, 768 dimensions, exécuté en local sur CPU. Multilingue, donc adapté au français juridique.
- e5 attend les préfixes `query: ` (questions) et `passage: ` (documents), ajoutés automatiquement.
- Vecteurs normalisés, index **ChromaDB** en distance cosinus (`app/vectorstore_juridique/`).

### 3.3 Trois modes de recherche

| Mode | Fonctionnement |
|---|---|
| `vector` | Les chunks les plus proches de la question (cosinus) |
| `hybrid` | 20 candidats vectoriels + 20 candidats **BM25**, fusionnés par **RRF** |
| `advanced` (**baseline**) | Question + 3 reformulations (**MultiQuery**) + un article fictif (**HyDE**), chacune en recherche vectorielle (10 résultats), + BM25 (10) → fusion RRF → 20 candidats → **reranker** cross-encoder |

Les briques, une par une :
- **BM25** : recherche par mots-clés (fréquence des termes, pondérée par leur rareté). Elle retrouve ce que l'embedding rate : « article 83 »,
  « NIS2 », « 72 heures ». Tokenisation française, en gardant les numéros d'actes comme `2016/679`.
- **RRF** (*Reciprocal Rank Fusion*) : chaque liste vote pour ses documents selon leur rang, `score(d) = Σ 1 / (60 + rang(d))`.
  Aucune normalisation de scores hétérogènes (cosinus contre BM25) n'est nécessaire.
- **MultiQuery** : le LLM reformule la question avec le vocabulaire des textes (« responsable du traitement », « entité essentielle »).
- **HyDE** (*Hypothetical Document Embeddings*) : le LLM rédige un faux article de loi qui répondrait à la question ; ce texte ressemble
  davantage aux articles du corpus que la question elle-même, donc il les retrouve mieux.
- **Reranker** (cross-encoder) : il lit la question **et** le passage ensemble et donne un score de pertinence. Plus précis qu'un
  embedding, mais trop lent pour tout le corpus : il ne reclasse que les candidats.
- **Cache des candidats** : MultiQuery et HyDE varient d'une exécution à l'autre. Les 20 candidats de chaque question sont mis en
  cache ; le Graph RAG relit les mêmes, ce qui garantit qu'il part exactement des mêmes graines que la baseline.

### 3.4 Budget de contexte et génération

- **5 articles distincts** dans le contexte, pour tous les modes (`pipeline.top_articles`) : sans cette règle, 3 morceaux du même article
  pouvaient occuper 3 places.
- **Articles complets** (`retriever.full_article_context`, limite 6 000 caractères, 95 % des articles tiennent dedans) : au début, seul le
  meilleur chunk de chaque article était envoyé. Or 47 des 59 articles attendus sont découpés en plusieurs chunks (4 en moyenne) :
  le bon article était trouvé, mais sans le paragraphe qui répond. La sélection ne change pas, seul le texte envoyé au LLM change.
- **Prompt juridique partagé** (`app/rag/prompts.py:ANSWER_PROMPT`) : citer `[n]` et nommer l'acte et l'article ; reprendre
  fidèlement conditions, délais, montants ; dire « Les articles fournis ne permettent pas de répondre à cette question » plutôt que
  d'inventer ; ne pas ajouter d'intitulé d'acte tiré de ses connaissances ; guillemets réservés aux citations exactes.

### 3.5 Ce que le projet vectoriel contient aussi

API FastAPI, mémoire de conversation (une relance est reformulée en question autonome), extraction structurée (Instructor,
schéma `LegalProvision`), et des **agents** : agent ReAct avec 3 outils (`search_legal_corpus`, `get_article`, `find_legal_acts`),
routeur LangGraph, superviseur multi-agents. Voir son README. L'agent est disponible dans l'interface de ce dépôt.

---

## 4. Les modèles et LLM utilisés

| Modèle | Rôle | Exécution |
|---|---|---|
| `intfloat/multilingual-e5-base` | Embeddings des chunks, des articles et des questions (768 dimensions) | Local, CPU |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder multilingue : reclasse les passages | Local, CPU (~2,3 Go) |
| BM25 (`rank-bm25`) | Recherche par mots-clés | Local |
| `openai/gpt-oss-120b` | **Génère les réponses**, température 0 | Groq (API) |
| `openai/gpt-oss-20b` | MultiQuery, HyDE, reformulation, **juge RAGAS** | Groq (API) |

- `gpt-oss` est un modèle à raisonnement : les tâches simples tournent avec `reasoning_effort="low"`, sinon le juge RAGAS épuisait ses tokens.
- **Quota Groq gratuit :** 8 000 tokens par minute et 200 000 par jour, **par modèle**. Une réponse avec articles complets consomme
  6 000 à 8 000 tokens : 25 à 30 réponses par jour. C'est la raison des échantillons réduits pour l'évaluation des réponses.
- **Conditions identiques pour tous les modes comparés :** mêmes chunks, embeddings, LLM, prompt, reranker et 5 articles. Seule la recherche change.

---

## 5. Le graphe de connaissances (Neo4j)

Construit par `graphrag/build_graph.py` (vide la base puis recharge tout en ~30 s). Comptes mesurés dans la base le 10 novembre 2026.

### 5.1 Schéma

```
(:Document {celex, title, short_name, label, type, date, year, in_corpus, url, aliases})   768 (100 du corpus, 668 hors corpus)
(:Article  {id, celex, num, title, text, position, chunk_ids, n_chunks, embedding})        3 941
(:EuroVoc  {label})                                                                        354
(:Cluster  {id, label, size, n_acts})                                                      25
```

| Arête | De → vers | Nombre | Comment elle est obtenue |
|---|---|---|---|
| `HAS_ARTICLE` | Document → Article | 3 941 | Découpage du corpus |
| `REFERS_TO` | Article → Article | 7 047 (629 entre actes différents) | Renvois extraits du texte par regex |
| `NEXT` | Article → Article | 3 841 | Ordre des articles dans l'acte |
| `MENTIONS` | Article → Document | 1 525 | Actes cités dans le texte de l'article |
| `AMENDS` / `REPEALS` | Article → Document | 109 / 30 | L'article précis qui opère la modification ou l'abrogation |
| `CITES` / `BASED_ON` | Document → Document | 619 / 495 | Cellar |
| `AMENDS` / `REPEALS` | Document → Document | 175 / 62 | Cellar |
| `HAS_SUBJECT` | Document → EuroVoc | 861 | Cellar |
| `IN_CLUSTER` | Article → Cluster | 3 941 | Louvain (§6) |

(Dans la base, `AMENDS` totalise 284 et `REPEALS` 92 : ce sont les sommes des deux niveaux.)

### 5.2 Comment les liens « article → acte » sont repérés (`graphrag/acts.py`)

Cellar dit qu'un acte en modifie un autre, mais pas **quel article** opère la modification. Pour le retrouver :
1. on repère les actes cités dans le titre et le texte de l'article (« directive (UE) 2016/1148 » → CELEX `32016L1148`) ;
2. on ne garde que les actes que Cellar relie déjà à l'acte de l'article par `AMENDS` ou `REPEALS` ;
3. on exige un verbe (« modifie », « abroge »…) dans le titre, ou une formule dispositive (« est modifié », « est abrogée ») dans le texte.

Exemple : l'article 66 du Cyber Resilience Act, titré « Modification du règlement (UE) 2019/1020 », reçoit
`(:Article {id:'32024R2847_art_66'})-[:AMENDS]->(:Document {celex:'32019R1020'})`.

Une règle a été corrigée en cours de route : « … restent en vigueur jusqu'à ce qu'elles soient modifiées, remplacées ou abrogées »
(RGPD, art. 45 et 46) créait à tort une abrogation de la directive 95/46/CE.

### 5.3 Embeddings des articles

L'embedding d'un article est la **moyenne normalisée des embeddings de ses chunks**, repris de Chroma. Réencoder l'article entier le
tronquerait à 512 tokens. Il est indexé dans Neo4j (index vectoriel `article_embedding`, cosinus).

### 5.4 Contrôle qualité (`graphrag/sample_edges.py`, `results/edge_sample.csv`)

50 arêtes extraites par règles, tirées au hasard (graine 42), vérifiées à la main :

| Type | Précision | Erreurs constatées |
|---|---|---|
| `REFERS_TO` | 28/30 = 93 % | Renvoi rattaché au mauvais acte (numéros proches) ; article d'un acte modifié attribué à l'acte modificateur |
| `AMENDS` article → acte | 11/11 = 100 % | Aucune |
| `REPEALS` article → acte | 8/9 = 89 % | Abrogation indirecte (« la Commission abroge… ») |
| **Total** | **47/50 = 94 %** | |

---

## 6. Les clusters thématiques (Louvain)

Script `graphrag/clusters.py`, résultats `results/clusters.json`, écrits dans Neo4j (`IN_CLUSTER`).

### 6.1 Le problème

91 % des renvois restent **à l'intérieur d'un même acte**. Un algorithme de communautés appliqué à ces seuls renvois retrouve donc les
actes, pas des thèmes transversaux comme « la cybersécurité » ou « l'IA ».

### 6.2 La méthode

- **Louvain** : algorithme qui regroupe les nœuds pour maximiser la **modularité**, c'est-à-dire la part des liens à l'intérieur des
  clusters par rapport à ce qu'on attendrait dans un graphe aléatoire de mêmes degrés. Graine 42.
- Deux graphes comparés :

| Graphe | Clusters | Modularité | NMI avec l'acte | Articles dans un cluster multi-actes |
|---|---|---|---|---|
| Structure seule (`REFERS_TO` + `NEXT`) | 57 | 0,917 | 0,920 | 58 % |
| **Thèmes** (`REFERS_TO` + 10 plus proches voisins sémantiques), **retenu** | **25** | 0,814 | 0,738 | **86 %** |

- Graphe « thèmes » : chaque article est relié à ses 10 plus proches voisins (cosinus des embeddings, poids = similarité), et chaque
  renvoi ajoute un poids de 1. Le **NMI** (information mutuelle normalisée) avec l'acte baisse de 0,920 à 0,738 : les clusters
  ne recopient plus les actes.
- **Étiquette** : les descripteurs EuroVoc sur-représentés, score = couverture × log(1 + lift), couverture d'au moins 15 %.
  Si au moins 40 % des articles d'un cluster ont un titre de clause type (« Entrée en vigueur », « Comité »…), il est étiqueté « Clauses formelles ».

### 6.3 Résultat

25 clusters, **0 article sans cluster**, dont 23 thématiques et 2 de clauses formelles. Les plus grands :

| Cluster | Articles | Actes |
|---|---|---|
| Vie privée · données personnelles · protection des données | 457 | 20 |
| Aide financière · budget général · contrat public | 359 | 17 |
| Contrat · rapprochement des législations · droit d'auteur | 231 | 33 |
| Marché unique numérique · prestation de services | 186 | 6 |
| Sécurité des systèmes d'information (cybersécurité) | 176 | 6 |
| Intelligence artificielle | 162 | 7 |

**Limites :** 6 clusters ne couvrent qu'un seul acte ; EuroVoc décrit l'acte, pas l'article, donc l'étiquette d'un cluster reflète ses actes.

---

## 7. Le Graph RAG

Code : `graphrag/retriever.py`. Le Graph RAG **ne remplace pas** la recherche vectorielle : il part de ses résultats et ajoute la structure.

### 7.1 La chaîne, étape par étape

| Étape | Ce qui se passe | Propre au graphe ? |
|---|---|---|
| 1. Candidats | Les 20 candidats du mode advanced, relus dans son cache (mêmes graines que la baseline) | Non |
| 2. Entités | Articles cités dans la question (« article 17 du RGPD ») ; si la question parle de modification ou d'abrogation, articles qui l'opèrent (`AMENDS` / `REPEALS`), 5 au plus | **Oui** |
| 3. Expansion | 1 saut depuis les 10 premiers articles, **5 voisins au plus par article**, par ordre : renvois sortants, renvois entrants, articles qui modifient l'acte, article voisin (`NEXT`). Si la question parle de modification, les articles modificateurs passent en premier. À type égal, le plus proche de la question | **Oui** |
| 4. Classement | Même reranker que la baseline, sur candidats + voisins (chaque voisin y entre avec son chunk le plus proche de la question) | Non |
| 5. Sélection | 5 articles distincts ; en mode `graph_linked`, 2 places réservées (§7.3) | **Oui** |
| 6. Contexte | Articles complets ; plus une ligne « Information du graphe (pas un extrait du texte) : » avec au plus 2 renvois suivis et 3 faits de modification | Ajout du graphe |
| 7. Réponse | Même prompt, même LLM (`pipeline.generate_answer`) | Non |

### 7.2 Les modes

| Mode | Description |
|---|---|
| `graph` | Baseline + voisins du graphe, tout passe par le reranker |
| `graph_norerank` | Comme `graph`, sans reranker (fusion RRF), pour isoler l'effet du reranker |
| `graph_linked` | Comme `graph`, mais les 2 dernières places vont aux articles **d'autres actes** cités par l'article classé 1er. **Mode retenu**, utilisé dans l'interface |

### 7.3 Pourquoi `graph_linked`

Sur les 8 questions multi-hop de test, le pool du mode `graph` contenait **les 18 articles attendus**, contre 14 pour la baseline : le
graphe trouvait les bons articles. Mais le reranker les classait **de la 7e à la 38e place**, car il note la ressemblance avec la
question, et un article lié par un renvoi ne ressemble pas forcément à la question.

Règle de `graph_linked` : on garde les 3 premiers articles du classement ; les 2 dernières places vont aux articles d'autres actes
vers lesquels renvoie l'article classé 1er, départagés par le reranker. Sans tel renvoi, le résultat est celui de `graph`.
**La règle a été figée avant l'évaluation finale et n'a plus été modifiée**, y compris après le test à l'aveugle.

**Limite :** si l'article classé 1er est hors sujet, ses renvois le sont aussi et peuvent évincer un bon article. C'est ce qui fait perdre
l'article 26 du DSA sur la question de la publicité ciblée (§9.5).

---

## 8. La classification : texte seul contre GNN

Script `graphrag/classify.py`, résultats `results/classification.json`.

### 8.1 La tâche

- **Multi-étiquettes** : prédire, pour chaque article, les 10 descripteurs EuroVoc présents dans au moins 10 actes (règle fixée avant
  les essais) : Internet, données personnelles, harmonisation des normes, marché unique, marché unique numérique, prestation de services,
  protection des données, sécurité des systèmes d'information, technologie numérique, échange d'information.
- Les étiquettes existent au niveau de l'acte : chaque article hérite de celles de son acte.
- Entrée de tous les modèles : l'embedding e5 de l'article (768 dimensions).

### 8.2 Validation

- **GroupKFold à 5 plis par acte** : aucun article d'un acte de test n'est vu à l'entraînement. Sans cela, le modèle n'aurait qu'à
  reconnaître l'acte.
- Dans l'entraînement, 15 % des actes servent de **validation** : arrêt anticipé, choix du seuil de décision par étiquette (grille 0,05
  à 0,95, celui qui maximise le F1) et du paramètre C de la régression logistique (1, 10 ou 100). **Le test n'intervient dans aucun choix.**
- Cadre transductif : toutes les arêtes et tous les embeddings sont visibles, jamais les étiquettes des actes de test.

### 8.3 Les modèles

| Modèle | Principe |
|---|---|
| Régression logistique | Une régression par étiquette (One-vs-Rest), texte seul |
| LogReg + moyenne de l'acte | Embedding de l'article concaténé à la moyenne des embeddings de son acte : témoin du « contexte d'acte » sans graphe |
| MLP | 2 couches cachées de 256, ReLU, dropout 0,3, texte seul |
| **GCN** | `H' = σ(D^-1/2 (A + I) D^-1/2 · H · W)` : chaque couche fait une moyenne normalisée des représentations des voisins (et de soi). 2 propagations |
| **GAT** | Attention apprise par voisin : `e_ij = LeakyReLU(aᵀ [W h_i ‖ W h_j])`, softmax sur les voisins, 4 têtes. 2 couches |
| **Graph Transformer** | Attention requête-clé limitée aux voisins, `softmax(q_i · k_j / √d)`, connexions résiduelles, LayerNorm, couche feed-forward ; encodage de position par probabilités de retour d'une marche aléatoire (8 pas) |

Deux jeux d'arêtes : `all` (`REFERS_TO` + `NEXT`) et `cross` (seulement les renvois entre actes différents, ablation).
Entraînement : Adam (lr 0,002, weight decay 1e-4), entropie croisée binaire, 300 époques au plus, patience 30. Écrits en PyTorch pur
(torch_geometric n'était pas installable).

### 8.4 Résultats (moyenne ± écart-type sur 5 plis)

| Modèle | Macro-AP | Micro-F1 | Macro-F1 | F1 par acte |
|---|---|---|---|---|
| LogReg + moyenne de l'acte (sans graphe) | **0,537 ± 0,162** | 0,432 | 0,387 | 0,446 |
| LogReg, texte seul | 0,420 ± 0,128 | 0,370 | 0,320 | 0,419 |
| Graph Transformer, inter-actes | 0,357 ± 0,104 | 0,352 | 0,294 | 0,416 |
| GAT, toutes arêtes | 0,343 ± 0,133 | 0,326 | 0,285 | 0,383 |
| Graph Transformer, toutes arêtes | 0,273 ± 0,093 | 0,275 | 0,214 | 0,352 |
| GAT, inter-actes | 0,253 ± 0,105 | 0,299 | 0,262 | 0,366 |
| MLP, texte seul | 0,208 ± 0,052 | 0,275 | 0,238 | 0,355 |
| GCN, inter-actes | 0,208 ± 0,059 | 0,275 | 0,231 | 0,347 |
| GCN, toutes arêtes | 0,170 ± 0,054 | 0,269 | 0,234 | 0,350 |

**Lecture :**
- Le graphe n'apporte pas de gain au-delà d'un classifieur texte bien réglé. Le meilleur modèle n'utilise aucune arête.
- À entraînement identique, le graphe aide un réseau de neurones : Graph Transformer inter-actes +0,149 et GAT +0,135 face au MLP, sur les 5 plis. Le GCN, non.
- Explication probable, **non testée** : les étiquettes sont au niveau de l'acte et 91 % des renvois restent dans l'acte ; les arêtes transportent surtout du contexte d'acte, que la moyenne de l'acte fournit directement.
- Limites : une graine, hyperparamètres des réseaux non optimisés, fort écart-type entre plis.

---

## 9. L'évaluation : métriques, protocole, résultats

### 9.1 Les métriques

**Recherche** (au niveau de l'article : la liste des chunks est dédoublonnée en articles) :

| Métrique | Formule | Lecture |
|---|---|---|
| Recall@5 | articles attendus parmi les 5 retenus / articles attendus | Part des bons articles retrouvés |
| Hit@5 | 1 si au moins un article attendu est retenu | La recherche a-t-elle trouvé quelque chose d'utile ? |
| MRR | moyenne de 1 / rang du premier article attendu | Le bon article est-il en tête ? |

**Réponses** (RAGAS, juge `gpt-oss-20b`) :

| Métrique | Ce qu'elle mesure |
|---|---|
| Faithfulness | Le juge découpe la réponse en affirmations et vérifie chacune dans le contexte : part des affirmations appuyées |
| Answer relevancy | Le juge régénère une question à partir de la réponse ; similarité (embeddings) avec la vraie question. `strictness=1`, car Groq refuse `n>1` |
| Context recall | Part des éléments de la réponse de référence qu'on retrouve dans le contexte |

**Classification et clusters** :

| Métrique | Ce qu'elle mesure |
|---|---|
| Macro-AP | Pour chaque étiquette, aire sous la courbe précision-rappel ; moyenne des étiquettes. Indépendante du seuil : mesure principale |
| Micro-F1 / Macro-F1 | F1 sur toutes les décisions / moyenne des F1 par étiquette (sensible aux étiquettes rares) |
| F1 par acte | Probabilités moyennées sur les articles d'un acte, comparées aux étiquettes de l'acte |
| Modularité | Densité des liens à l'intérieur des clusters par rapport au hasard (−0,5 à 1) |
| NMI | Information partagée entre les clusters et les actes (0 = indépendants, 1 = identiques) |

**Test statistique :** test des signes bilatéral sur les questions où deux modes diffèrent (meilleur / moins bon), sans hypothèse sur la distribution.

### 9.2 Règles de comparaison

Mêmes questions, mêmes métriques, mêmes chunks, mêmes embeddings, même LLM (température 0), même prompt, même reranker,
**5 articles distincts** dans le contexte pour tous les modes. Le RAG vectoriel n'a pas été modifié pour favoriser l'un ou l'autre.

### 9.3 Jeux de questions

| Jeu | Fichier | Contenu |
|---|---|---|
| 60 questions | `data/eval_60.jsonl` | 34 factuelles, 6 relationnelles, 20 multi-hop, avec articles attendus vérifiés dans le corpus |
| 21 questions à l'aveugle | `data/questions_utilisateur_21.jsonl` | Rédigées par l'équipe le 10 novembre, jamais utilisées pour un réglage, lancées une seule fois |

Origine des 60 questions : `B-` 24 questions de départ (ont servi à concevoir le retriever), `T-` 28 questions de test (leurs résultats ont
motivé `graph_linked`), `U-R` 2 relationnelles, `N-` 6 multi-hop neuves rédigées à partir de renvois inter-actes tirés au hasard (graine 2026).

Les 21 questions à l'aveugle se répartissent en 4 titres :
- **A**, réponse cachée loin dans l'article (5 questions) ;
- **B**, multi-hop entre actes (7) ;
- **C**, relationnelles inverses, dont la réponse n'est pas dans l'acte modifié (5) ;
- **D**, pièges anti-hallucination (4).

Avant toute exécution, chaque article attendu a été vérifié dans le texte de loi, et les sous-ensembles à générer ont été tirés au sort
(graine 20261110, `scripts/build_user_questions.py`).

### 9.4 Résultats de recherche

**60 questions** (`results/eval_graph_20261108_215757.json`)

| | `vector` | `advanced` | `graph` | **`graph_linked`** |
|---|---|---|---|---|
| Recall@5 global | 0,811 | 0,858 | 0,869 | **0,922** |
| MRR global | 0,727 | 0,866 | 0,873 | 0,870 |
| Factuelles (34) | 0,941 | 1,000 | 1,000 | 1,000 |
| Relationnelles (6) | 0,833 | 0,833 | 1,000 | 1,000 |
| Multi-hop (20) | 0,583 | 0,625 | 0,608 | **0,767** |
| Articles attendus retrouvés | 64/85 | 68/85 | 68/85 | **75/85** |

| Origine | n | `advanced` | `graph` | `graph_linked` |
|---|---|---|---|---|
| `B-` | 24 | 29/32 | 30/32 | 29/32 |
| `T-` | 28 | 29/38 | 28/38 | 33/38 |
| `U-R` | 2 | 3/3 | 3/3 | 3/3 |
| `N-` (jamais réglées) | 6 | 7/12 | 7/12 | **10/12** |

`graph_linked` contre la baseline : meilleur sur 9 questions, moins bon sur 2 (B-M06, T-M04) ; **p ≈ 0,065, non significatif au seuil de 5 %**.

**17 questions à l'aveugle avec articles attendus** (`results/eval_graph_20261110_005139.json`)

| | `vector` | `advanced` | `graph` | **`graph_linked`** |
|---|---|---|---|---|
| Recall@5 global | 0,515 | 0,632 | 0,691 | **0,917** |
| MRR global | 0,721 | 0,853 | 0,912 | 0,912 |
| Hit@5 global | 0,824 | 0,941 | 1,000 | 1,000 |
| A – réponse loin dans l'article (5) | 0,533 | 0,733 | 0,733 | 0,900 |
| B – multi-hop (7) | 0,440 | 0,440 | 0,440 | **0,869** |
| C – relationnelles inverses (5) | 0,600 | 0,800 | 1,000 | 1,000 |

`graph_linked` contre la baseline : meilleur sur 9, moins bon sur 0, égal sur 8 ; p ≈ 0,004. **À nuancer :** dans les titres A et B, le second
article attendu est souvent un article cité par le premier, exactement ce que `graph_linked` va chercher.
Exemple marquant, C2 « Quel acte du corpus modifie le règlement 2019/1020 ? » : la baseline ne trouve aucun bon article, le graphe le place 1er.

**Latence de recherche** (CPU, indicative) : environ 12 à 19 s pour `advanced`, 36 à 38 s pour `graph`.

### 9.5 Qualité des réponses

Les réponses ont été générées avec articles complets, dans les deux modes, par `scripts/answer_user_questions.py`
(`results/answers_utilisateur_20261110_010908.json` et `…_015420.json`), puis **relues à la main contre le texte des articles**.

| Question | Baseline (`advanced`) | Graph RAG (`graph_linked`) |
|---|---|---|
| P1 – publicité ciblée et données sensibles (cas signalé) | Juste : DSA art. 26 §3 | **Incomplète** : ne répond que pour la publicité politique, DSA art. 26 perdu |
| P2 – analyse d'impact AI Act et AIPD (cas signalé) | Juste : art. 27 §4 | Juste |
| A2 – IA à haut risque et biais (tirée) | Juste : art. 10 §5 | Juste |
| A3 – Data Act et droits du RGPD (tirée) | Juste | Juste, plus complète (seule à citer l'art. 15 et « complètent ») ; 1 erreur de référence |
| B4 – mineurs et profilage (tirée) | Juste : DSA art. 28 | Juste |
| B5 – bac à sable réglementaire (tirée) | Juste ; 1 erreur de numéro d'acte | Juste |
| B6 – objet connecté et tiers (tirée) | Juste | Juste |
| C3 – modificateurs de 2017/2394 (tirée) | Juste | Juste ; 1 erreur de détail |
| C5 – modificateur de 2001/29 (tirée) | Juste | Juste |
| C2 – modificateur de 2019/1020 (hors tirage) | « Ne permet pas de répondre » | **Juste : CRA art. 66** |
| D1 – article 200 du RGPD (piège, hors tirage) | Refus | Refus |
| D2 – amende CNIL contre Google (piège, tirée) | Refus | Refus |
| D3 – cookie selon la loi française (piège, tirée) | Refus | Refus |
| D4 – fausse prémisse sur l'AI Act (piège, hors tirage) | Refus | Refus |

**Bilan :** sur les 10 questions qui ont une réponse, 9 justes pour chaque système, avec des échecs différents. Sur les 4 pièges, aucune
invention ; en revanche, aucun système ne corrige explicitement la prémisse (« le RGPD n'a que 99 articles », « les systèmes à haut
risque sont encadrés, pas interdits »).

**Erreurs de détail, vérifiées dans le texte :**
- Graph RAG, C3 : attribue la modification du règlement 2017/2394 à l'art. 48 du Data Act, qui modifie la directive 2020/1828.
- Graph RAG, A3 : écrit « articles 9-11 du Data Act » pour l'art. 5 §9-11.
- Baseline, B5 : écrit « règlement 2021/1725 » au lieu de 2018/1725.

**Citations entre guillemets** (contrôle automatique) : 32 fragments sur 37 (baseline) et 29 sur 34 (Graph RAG) figurent mot pour mot dans
les articles transmis ; les autres sont des écarts de ponctuation, des étiquettes ou des reformulations, sans contenu inventé.

**RAGAS** (`scripts/ragas_user_answers.py`, `results/ragas_utilisateur_20261110_013124.json`) sur B4, C3 et C5 seulement, choisies parce
que leur contexte est court (une requête du juge doit tenir dans 8 000 tokens par minute) :

| Moyenne sur 3 questions | Baseline | Graph RAG |
|---|---|---|
| Faithfulness | 1,000 | 0,944 |
| Answer relevancy | 0,872 | 0,872 |
| Context recall | 1,000 | 1,000 |

La seule faithfulness inférieure à 1 (C3, Graph RAG, 0,833) correspond à l'erreur sur l'article 48 trouvée à la main.

**Conclusion :** le gain de recherche **ne se traduit pas** en meilleures réponses sur cet échantillon, car l'article principal cite souvent
déjà le texte lié (B5, B6 : la baseline cite l'article 35 du RGPD ou la directive ePrivacy sans les avoir sélectionnés).

### 9.6 Une leçon de méthode

Le Recall@5 a d'abord surestimé la qualité : deux réponses fausses étaient comptées comme des succès, parce que le bon article était
retrouvé sans le bon paragraphe. D'où l'envoi des articles complets et la relecture systématique des réponses.

---

## 10. Comment les deux objectifs sont atteints

### Objectif 1 : un graphe de relations entre articles de loi avec clusters thématiques

1. **Articles comme nœuds** : les 3 941 articles des 100 actes, avec leur texte et un embedding (§5.1, §5.3).
2. **Relations entre articles** : 7 047 renvois `REFERS_TO` extraits du texte, dont 629 entre actes ; ordre `NEXT` (§5.1).
3. **Relations entre actes** et **article qui opère une modification** : Cellar + règles (§5.2).
4. **Qualité contrôlée** : 94 % de liens corrects sur un échantillon tiré au hasard (§5.4).
5. **Clusters thématiques** : Louvain sur renvois + voisins sémantiques, 25 clusters étiquetés par EuroVoc, tous les articles rattachés,
   86 % dans un cluster multi-actes (§6).
6. **Visible** dans Neo4j Browser et dans l'interface (§12).

### Objectif 2 : un modèle capable de répondre à des questions spécifiques sur la législation

1. **Recherche** : baseline vectorielle avancée (§3), puis Graph RAG qui suit renvois et modifications (§7).
2. **Génération contrôlée** : articles complets, prompt qui impose de citer l'acte et l'article et d'avouer quand le contexte ne suffit pas (§3.4).
3. **Vérifié** sur 60 + 21 questions : Recall@5 de 0,922 et 0,917 ; réponses relues à la main, 9 justes sur 10 ; pièges refusés (§9).
4. **Utilisable** : interface Streamlit avec sources cliquables vers EUR-Lex et chemin suivi dans le graphe (§12).

---

## 11. Limites

1. **Échantillons réduits** : 60 et 21 questions ; RAGAS sur 3 questions ; une seule graine pour les GNN.
2. **Significativité** : sur les 60 questions, le gain de recherche n'est pas significatif (p ≈ 0,065).
3. **Questions favorables au graphe** : les `N-` et une partie des questions à l'aveugle associent un article et un article qu'il cite.
4. **Recall et justesse** : le Recall@5 mesure la sélection d'articles, pas la justesse de la réponse.
5. **`graph_linked` dépend de l'article classé 1er** (cas de l'article 26 du DSA).
6. **Extraction par règles** : environ 6 % d'erreurs sur l'échantillon d'arêtes.
7. **Pièges** : refus corrects, mais prémisse fausse jamais corrigée explicitement.
8. **Coût** : recherche ~3 fois plus lente avec le graphe sur CPU ; quota Groq gratuit (25 à 30 réponses par jour).
9. **Périmètre** : ni considérants ni annexes.

Pistes : questions rédigées par des juristes, reranker qui tienne compte des liens, extraction des renvois par un modèle, règle
de places réservées conditionnée à la pertinence de l'article classé 1er.

---

## 12. Démonstration

**Avant la séance :** démarrer la base dans Neo4j Desktop, puis `python -m graphrag.check`, puis lancer l'interface
(`python -m streamlit run streamlit_app.py --server.port 8502 --server.headless true`) et ouvrir http://localhost:8502.
Poser à l'avance les questions de la démo dans l'onglet ouvert : elles restent affichées (plan B si Groq sature).

Les candidats des questions de test ont été copiés dans le cache de l'interface (`data/cache/streamlit_candidates.json`) : en démo, ces
questions sélectionnent exactement les mêmes articles qu'à l'évaluation, sans appel MultiQuery ni HyDE.

| Temps | Où | Action | Attendu |
|---|---|---|---|
| 0:30 | Interface | Barre latérale | 100 actes, 3 941 articles, 7 047 renvois, 25 clusters |
| 1:15 | Mode « Graph RAG » | « Quel acte du corpus modifie le règlement 2019/1020 ? » | Règlement (UE) 2024/2847, art. 66 ; source marquée « ajouté par le graphe » |
| 0:45 | Mode « RAG avancé » | Même question | « Les articles fournis ne permettent pas de répondre » |
| 0:45 | Mode « Graph RAG » | « Quel montant d'amende la CNIL a-t-elle infligé à Google en 2022 ? » | Refus, rien d'inventé |
| 1:30 | Neo4j Browser (http://localhost:7474) | Requêtes ci-dessous | Renvoi DSA art. 28 → RGPD art. 4 ; cluster IA ; article modificateur |

```cypher
// Renvoi entre actes : DSA art. 28 -> RGPD art. 4 (définition du profilage)
MATCH p=(a:Article {id:'32022R2065_art_28'})-[:REFERS_TO]->(b:Article)<-[:HAS_ARTICLE]-(d:Document) RETURN p

// Cluster « intelligence artificielle » : 305 renvois internes, 5 actes
MATCH (c:Cluster {id:10})<-[:IN_CLUSTER]-(a:Article)-[r:REFERS_TO]->(b:Article)-[:IN_CLUSTER]->(c) RETURN a, r, b LIMIT 60

// L'article qui modifie le règlement 2019/1020
MATCH p=(a:Article)-[:AMENDS]->(d:Document {celex:'32019R1020'}) RETURN p

// Les clusters
MATCH (c:Cluster) RETURN c.id, c.label, c.size, c.n_acts ORDER BY c.size DESC
```

**Précautions :** éviter le mode « Comparer » sur une question longue (deux réponses en moins d'une minute peuvent dépasser 8 000 tokens par
minute) ; si « Le modèle de langage est momentanément indisponible » s'affiche, attendre une minute et reposer la question. Questions à
contexte court, sûres en démo : B4, C2, C3, C5, D2.

---

## 13. Installation et commandes

Prérequis : le projet vectoriel installé et indexé (environnement conda `rag-juridique`, Python 3.11, torch CPU) et **Neo4j Desktop**
(APOC et Graph Data Science), base démarrée.

```powershell
conda activate rag-juridique
cd C:\Users\hazem\5eme\GraphRAG-juridique
pip install -r requirements.txt
copy .env.example .env               # renseigner NEO4J_PASSWORD
python -m graphrag.check             # vérifie les données et la connexion Neo4j
```

Sur ce poste : garder `pyarrow==26.0.0` (Smart App Control bloque un fichier de la 26.0.1) et `pandas<3`.

| Action | Commande |
|---|---|
| **Interface** (http://localhost:8502) | `python -m streamlit run streamlit_app.py --server.port 8502 --server.headless true` |
| Construire le graphe (**vide la base**, ~30 s) | `python -m graphrag.build_graph` |
| Échantillon et précision des arêtes | `python -m graphrag.sample_edges [--score]` |
| Clusters (à refaire après `build_graph`) | `python -m graphrag.clusters` |
| Classification (5 plis, ~10 min) | `python -u -m graphrag.classify` |
| Recherche, 60 questions | `python -u -m graphrag.evaluate --file data\eval_60.jsonl --modes vector advanced graph graph_linked` |
| Recherche, questions à l'aveugle | `python -u -m graphrag.evaluate --file data\questions_utilisateur_21.jsonl --types A_article_long B_multi_hop C_relationnelle --modes vector advanced graph graph_linked` |
| Réponses des questions à l'aveugle (quota Groq) | `$env:PYTHONPATH="."; python -u scripts\answer_user_questions.py [--ids C2 D1 D4]` |
| RAGAS sur réponses enregistrées (juge 20b) | `$env:PYTHONPATH="."; python -u scripts\ragas_user_answers.py --answers results\answers_utilisateur_XXX.json` |
| Comparer les réponses de deux modes | `python -u -m graphrag.compare_answers --results results\eval_graph_20261108_215757.json` |
| Reconstruire les jeux de questions | `$env:PYTHONPATH="."; python scripts\build_eval_set.py` ; `python scripts\build_user_questions.py` |
| Régénérer les notebooks avec leurs sorties | `$env:PYTHONPATH="."; python scripts\build_notebooks.py [01 02 …]` (Neo4j démarré, aucun appel au LLM) |

### Notebooks (`notebooks/`)

| Notebook | Contenu | Vérification incluse |
|---|---|---|
| `01_donnees_et_graphe` | Corpus, relations, comptes Neo4j, exemples de chemins, contrôle qualité | 47/50 liens corrects |
| `02_clusters_thematiques` | Louvain recalculé sur les deux graphes, 25 clusters | Recalcul identique au fichier enregistré |
| `03_rag_vectoriel_et_graph_rag` | Sélection des articles par mode, contexte complet, prompt, réponses enregistrées | Sélections identiques à l'évaluation |
| `04_classification_gnn` | Résultats des 5 plis, architectures GCN/GAT/GT, pli 0 rejoué | Macro-AP rejouées = enregistrées |
| `05_evaluation` | Recall@5, MRR, tests des signes, réponses, citations, RAGAS | p = 0,0654 et 0,0039 |

Jupyter n'étant pas installé sur le poste, `scripts/build_notebooks.py` exécute chaque cellule avec le code du projet et écrit ses
sorties réelles dans les fichiers `.ipynb`. Les notebooks s'ouvrent et se relancent normalement dans Jupyter ou VS Code.

Pour les scripts de `scripts\` qui génèrent des réponses, le cache des candidats est `data/cache/user_test_candidates.json` (variable `SEED_CACHE_PATH`) :
rien n'est écrit dans le projet vectoriel.

---

## 14. Organisation du dépôt

```
graphrag/
  config.py           chemins, Neo4j, accès au code du projet vectoriel
  db.py               connexion Neo4j
  acts.py             repérage des actes et articles cités, relations opérées par un article
  build_graph.py      construction du graphe et contrôles
  check.py            vérification de l'installation
  sample_edges.py     échantillon d'arêtes à contrôler à la main
  retriever.py        Graph RAG : modes graph, graph_norerank, graph_linked ; contexte en articles complets
  evaluate.py         évaluation des modes (métriques du projet vectoriel)
  compare_answers.py  comparaison des réponses de deux modes
  clusters.py         clusters Louvain, écrits dans Neo4j
  classify.py         LogReg, MLP, GCN, GAT, Graph Transformer
streamlit_app.py      interface commune RAG vectoriel et Graph RAG
notebooks/            5 notebooks exécutés (données, clusters, RAG, GNN, évaluation)
docs/captures/        captures de l'interface et de Neo4j utilisées dans la présentation
scripts/
  collect_external_relations.py   relations et titres des actes hors corpus
  sample_cross_act_refs.py        tirage des renvois inter-actes (questions N-)
  build_eval_set.py               jeu de 60 questions
  build_user_questions.py         21 questions à l'aveugle et tirage des réponses à générer
  answer_user_questions.py        réponses baseline et Graph RAG, même chaîne que l'interface
  ragas_user_answers.py           RAGAS sur des réponses enregistrées
  build_notebooks.py              exécution des notebooks et écriture de leurs sorties
data/
  eurlex_external/                relations et actes hors corpus
  questions_test_28.jsonl, questions_neuves_6.jsonl, eval_60.jsonl
  questions_utilisateur_21.jsonl  questions à l'aveugle
  cache/                          caches des candidats (interface, tests) et conversations de l'agent
results/
  edge_sample.csv                          arêtes annotées
  eval_graph_20261108_215757.json          recherche, 60 questions
  eval_graph_20261110_005139.json          recherche, 17 questions à l'aveugle
  answers_utilisateur_20261110_010908.json réponses : 2 cas signalés + 9 tirées
  answers_utilisateur_20261110_015420.json réponses : C2, D1, D4 (hors tirage)
  ragas_utilisateur_20261110_013124.json   RAGAS sur B4, C3, C5
  clusters.json, classification.json
  eval_graph_20261108_203929.json, eval_graph_20261108_210819.json   comparaisons intermédiaires
```

---

## 15. Glossaire

| Terme | Définition |
|---|---|
| Chunk | Morceau de texte indexé (ici un article ou une partie d'article) |
| Embedding | Vecteur qui représente le sens d'un texte ; deux textes proches ont des vecteurs proches (cosinus) |
| BM25 | Score de recherche par mots-clés, fondé sur la fréquence des termes et leur rareté dans le corpus |
| RRF | Fusion de plusieurs classements : `Σ 1 / (60 + rang)` |
| MultiQuery | Plusieurs reformulations de la question, chacune cherchée séparément |
| HyDE | Recherche avec un document hypothétique rédigé par le LLM |
| Cross-encoder / reranker | Modèle qui lit question et passage ensemble pour noter leur pertinence |
| Graine (recherche) | Article de départ de l'expansion dans le graphe |
| Multi-hop | Question qui demande de combiner plusieurs articles ou actes |
| Louvain | Algorithme de détection de communautés qui maximise la modularité |
| Modularité | Densité des liens internes aux clusters comparée au hasard |
| NMI | Information mutuelle normalisée entre deux partitions |
| GCN / GAT / Graph Transformer | Réseaux de neurones sur graphe : moyenne des voisins / attention apprise / attention requête-clé |
| GroupKFold | Validation croisée où un groupe (ici un acte) n'est jamais à la fois en entraînement et en test |
| Macro-AP | Moyenne, sur les étiquettes, de l'aire sous la courbe précision-rappel |
| CELEX | Identifiant officiel d'un acte de l'UE |
| EuroVoc | Thésaurus officiel de l'UE, utilisé pour indexer les actes par thème |

---

## 16. Questions probables du jury

**Pourquoi deux projets séparés ?** Pour que la baseline reste intacte : le Graph RAG importe son code au lieu de le recopier, donc les
conditions de comparaison sont identiques par construction.

**Votre comparaison est-elle équitable ?** Oui : mêmes candidats de départ (cache), mêmes embeddings, reranker, LLM, prompt et 5 articles.
Seules les étapes 2, 3 et 5 du Graph RAG diffèrent (§7.1).

**Le gain est-il significatif ?** Sur les 60 questions, non (p ≈ 0,065). Sur les 17 questions à l'aveugle, oui (p ≈ 0,004), mais ces
questions favorisent le graphe par construction. Nous ne concluons pas à un gain général.

**Pourquoi les GNN font-ils moins bien qu'une régression logistique ?** Les étiquettes décrivent l'acte, 91 % des renvois restent dans
l'acte : les arêtes apportent surtout du contexte d'acte, que la moyenne de l'acte donne plus directement. Avec 3 941 exemples, un
modèle linéaire bien régularisé est difficile à battre. Hypothèse non testée.

**Pourquoi pas torch_geometric ?** Non installable sur ce poste ; les trois couches sont écrites en PyTorch, ce qui montre aussi leur fonctionnement.

**Comment savez-vous que le graphe est correct ?** 50 arêtes tirées au hasard, vérifiées à la main : 94 % correctes, erreurs décrites (§5.4).

**Pourquoi Louvain sur un graphe enrichi de similarités ?** Sur les renvois seuls, les clusters recopient les actes (NMI 0,920). Les
voisins sémantiques font apparaître des thèmes transversaux (NMI 0,738, 86 % d'articles dans des clusters multi-actes).

**Le modèle peut-il halluciner ?** Oui, ponctuellement sur des détails (3 erreurs de référence relevées sur 28 réponses). Il refuse en
revanche les 4 pièges. Le prompt impose les citations et les guillemets exacts ; nous avons vérifié automatiquement les citations.

**Pourquoi Groq et ce modèle ?** Accès gratuit à un grand modèle ouvert (gpt-oss-120b) ; contrepartie : 8 000 tokens par minute et
200 000 par jour, d'où des échantillons réduits pour l'évaluation des réponses.

**Que feriez-vous avec plus de temps ?** Plus de questions écrites par des juristes, RAGAS sur l'ensemble, un reranker qui tienne compte
des liens, plusieurs graines pour les GNN.

---

## 17. État d'avancement

| Élément | État |
|---|---|
| Collecte et découpage | Fait |
| RAG vectoriel (baseline) | Fait |
| Graphe Neo4j et contrôle qualité | Fait |
| Clusters thématiques | Fait |
| Graph RAG | Fait |
| Classification (témoins et GNN) | Fait |
| Évaluation de la recherche (60 + 17 questions) | Fait |
| Évaluation des réponses (14 questions relues, RAGAS sur 3) | Fait, échantillon réduit |
| Interface Streamlit | Faite |
| Présentation de validation | Faite |
| Notebooks à déposer sur Blackboard | Faits (`notebooks/`) |
