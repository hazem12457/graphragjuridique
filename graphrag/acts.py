"""Repérage des actes juridiques cités dans un texte (article ou question).

Deux usages :
  - à la construction du graphe : « la directive (UE) 2016/1148 est abrogée » -> 32016L1148 ;
  - à la recherche : « Comment le DSA modifie-t-il la directive 2000/31/CE ? » -> {32022R2065, 32000L0031}.
"""
import re

from graphrag.config import use_vector_rag

use_vector_rag()
from app.rag.corpus import get_corpus, normalize, short_act_name  # noqa: E402  (code du projet RAG vectoriel)


def short_act_name_safe(title: str) -> str:
    """Nom court d'un acte hors corpus ('Directive 95/46/CE du Parlement…' -> 'Directive 95/46/CE')."""
    try:
        return short_act_name(title or "")
    except Exception:
        return ""

# « directive (UE) 2016/1148 », « règlement (CE) n° 45/2001 », « règlement (UE) n o 182/2011 », « directive 95/46/CE »
ACT_RE = re.compile(
    r"(r[èe]glement|directive|d[ée]cision)\s+(?:d[’']ex[ée]cution\s+|d[ée]l[ée]gu[ée]e?\s+)?"
    r"(?:\((?:UE|CE|CEE|Euratom)\)\s+)?(?:n\s*[°o]\s*)?(\d{2,4})\s*/\s*(\d{1,4})(?:/(?:CEE|CE|UE))?",
    re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\d/])(\d{2,4})\s*/\s*(\d{1,4})(?![\d])")

AMEND_RE = re.compile(r"modifi|supprim|remplac|ajout|insere|amend(?!es?\b)")   # « amende(s) » = sanction
REPEAL_RE = re.compile(r"abrog")
# Règle 3 (titre sans verbe) : formule dispositive exigée. Sans elle, « … restent en vigueur jusqu'à ce qu'elles
# soient modifiées, remplacées ou abrogées » (RGPD, art. 45 et 46) devenait une abrogation de la directive 95/46/CE.
STRONG_RE = {
    "REPEALS": re.compile(r"\b(est|sont) abroge"),
    "AMENDS": re.compile(r"\b(est|sont) (modifie|supprime|remplace|ajoute|insere)|\bmodifie comme suit"),
}


def _year_ok(x: str) -> bool:
    return len(x) == 4 and 1950 <= int(x) <= 2035


def act_to_celex(kind: str, a: str, b: str) -> str:
    """('directive', '95', '46') -> '31995L0046' ; ('règlement', '45', '2001') -> '32001R0045'."""
    kind = kind.lower()
    letter = "R" if kind.startswith("r") else ("L" if kind.startswith("di") else "D")
    if _year_ok(a) and not _year_ok(b):
        year, num = a, b
    elif _year_ok(b) and not _year_ok(a):
        year, num = b, a
    elif len(a) == 2 and letter in "LD":          # ancienne numérotation : directive 95/46/CE
        year, num = ("19" if int(a) > 50 else "20") + a, b
    else:
        year, num = a, b
    return f"3{year}{letter}{int(num):04d}"


def mentioned_acts(text: str) -> dict[str, str]:
    """CELEX -> forme rencontrée dans le texte (« directive (UE) 2016/1148 »)."""
    out = {}
    for m in ACT_RE.finditer(text or ""):
        out.setdefault(act_to_celex(*m.groups()), re.sub(r"\s+", " ", m.group(0)).strip())
    return out


def act_number(title: str) -> str | None:
    """'Directive 95/46/CE du Parlement…' -> '95/46' (numéro tel qu'on le cite)."""
    m = NUMBER_RE.search(title or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;:])\s+(?=[A-ZÀ-Ý0-9«])", text or "") if s]


def article_relations(title: str, text: str, own_celex: str,
                      doc_relations: dict[str, set[str]]) -> dict[tuple[str, str], str]:
    """Relations AMENDS / REPEALS opérées PAR cet article : {(type, celex cible): preuve}.

    `doc_relations` : celex cible -> types de relation connus entre l'acte de l'article et la cible
    (relations.csv + relations_external.csv, issues de Cellar).

    Règles, de la plus sûre à la moins sûre :
      1. titre : « Abrogation de la directive 95/46/CE », « Modification de la directive (UE) 2019/1937 »
         -> la relation porte sur les actes nommés dans le titre ;
      2. titre « Abrogation » / « Modifications » sans acte nommé -> phrase du texte qui contient à la
         fois le verbe (abroge, modifie, supprime, remplace…) et l'acte ;
      3. titre sans verbe -> même règle de phrase, mais seulement si Cellar confirme la relation entre actes.
    Les passages entre guillemets (texte inséré dans l'acte modifié) sont ignorés.
    Preuve enregistrée sur l'arête : "titre" (règle 1), "phrase" (règle 2), "phrase_cellar" (règle 3, la plus bruitée).
    """
    found: dict[tuple[str, str], str] = {}
    title_types = title_relation_types(title)
    title_acts = {c for c in mentioned_acts(title) if c != own_celex}

    if title_types and title_acts:                                       # règle 1
        for t in title_types:
            for c in title_acts:
                found[(t, c)] = "titre"
        return found

    body = re.sub(r"«[^»]*»", " ", text or "")
    for sent in sentences(body):
        sent_n = normalize(sent)
        types = ({"REPEALS"} if REPEAL_RE.search(sent_n) else set()) | \
                ({"AMENDS"} if AMEND_RE.search(sent_n) else set())
        if title_types:
            # « La directive 97/7/CE, telle que modifiée par…, est abrogée » : dans un article « Abrogation »,
            # seule l'abrogation compte (la modification décrit l'historique de l'acte)
            types &= title_types
        if not types:
            continue
        for c in mentioned_acts(sent):
            if c == own_celex:
                continue
            for t in types:
                if title_types:                                          # règle 2
                    found.setdefault((t, c), "phrase")
                elif t in doc_relations.get(c, set()) and STRONG_RE[t].search(sent_n):   # règle 3
                    found.setdefault((t, c), "phrase_cellar")
    return found


_TITLE_HEAD = re.compile(r"^(abrogations?|modifications?)(?:\s+(\S+))?")
_TITLE_NEXT_OK = {"et", "de", "du", "des", "au", "aux", "a", "apportees", "d", "l", "la", "le", "les"}


def title_relation_types(title: str) -> set[str]:
    """« Abrogation », « Modifications du règlement… », « Abrogation et dispositions transitoires » -> types ;
    « Modifications substantielles après agrément » -> aucun (le titre décrit autre chose qu'une modification d'acte)."""
    t = normalize(title).replace("'", " ")
    m = _TITLE_HEAD.match(t)
    if not m or (m.group(2) and m.group(2) not in _TITLE_NEXT_OK):
        return set()
    types = {"REPEALS"} if m.group(1).startswith("abrog") else {"AMENDS"}
    if REPEAL_RE.search(t) and AMEND_RE.search(t):        # « Modification et abrogation de… »
        types = {"REPEALS", "AMENDS"}
    return types


_ARTICLE_NUM = r"\d+(?:\s*(?:bis|ter|quater))?"
ARTICLE_MENTION_RE = re.compile(rf"\barticles?\s+({_ARTICLE_NUM}(?:\s*(?:,|et|ou)\s*{_ARTICLE_NUM})*)")
_MAX_GAP_TO_ACT = 30      # « article 17 du RGPD » : l'acte suit de près la mention de l'article


def question_acts(question: str) -> dict[str, int]:
    """Actes nommés dans une question : CELEX -> position dans la question normalisée.

    Trois formes : « directive (UE) 2016/1148 » (y compris hors corpus), numéro seul (« 2000/31/CE », corpus
    seulement car le type d'acte est inconnu), alias (« RGPD », « DSA »).
    """
    q = normalize(question)
    corpus = get_corpus()
    found: dict[str, int] = {}
    for m in ACT_RE.finditer(q):
        found.setdefault(act_to_celex(*m.groups()), m.start())
    for m in NUMBER_RE.finditer(q):
        celex = corpus.resolve_act(m.group(0))
        if celex:
            found.setdefault(celex, m.start())
    for celex, doc in corpus.documents.items():
        for alias in doc.aliases:
            alias_n = normalize(alias)
            m = re.search(rf"\b{re.escape(alias_n)}\b", q) if len(alias_n) >= 3 else None
            if m:
                found[celex] = min(found.get(celex, m.start()), m.start())
    return found


def question_articles(question: str, acts: dict[str, int]) -> list[str]:
    """Articles cités explicitement (« article 17 du RGPD », « articles 6 et 7 du RGPD ») -> identifiants du corpus.

    Acte retenu : celui qui suit de près la mention, sinon le dernier nommé avant elle, sinon le suivant.
    """
    q = normalize(question)
    corpus = get_corpus()
    out: list[str] = []
    for m in ARTICLE_MENTION_RE.finditer(q):
        after = sorted((p, c) for c, p in acts.items() if p >= m.end())
        before = sorted((p, c) for c, p in acts.items() if p < m.start())
        if after and after[0][0] - m.end() <= _MAX_GAP_TO_ACT:
            celex = after[0][1]
        elif before:
            celex = before[-1][1]
        elif after:
            celex = after[0][1]
        else:
            continue
        for num in re.findall(_ARTICLE_NUM, m.group(1)):
            art = corpus.get_article(celex, num)
            if art and art.id not in out:
                out.append(art.id)
    return out


def relation_intents(question: str) -> set[str]:
    """Types de relation demandés par la question (« abroge » -> REPEALS, « modifie » -> AMENDS…)."""
    q = normalize(question)
    intents = set()
    if REPEAL_RE.search(q) or "remplace" in q:
        intents.add("REPEALS")
    if re.search(r"modifi|amend(?!es?\b)", q):       # « amende(s) » = sanction, pas modification
        intents.add("AMENDS")
    if re.search(r"base juridique|fondement|fonde sur", q):
        intents.add("BASED_ON")
    return intents
