"""Écrit data/questions_utilisateur_21.jsonl : 21 questions d'évaluation rédigées par l'utilisateur.

Elles n'ont servi à aucun réglage du système. Les articles attendus et les réponses de référence sont ceux de l'utilisateur,
vérifiés dans le texte des articles avant toute exécution.

Sous-ensemble dont les réponses sont générées (quota Groq) : tiré au sort avant de lancer le système, graine fixée :
2 questions des titres A, C, D et 3 du titre B. Champ `reponse_llm`.

Usage (depuis la racine du projet) : $env:PYTHONPATH="."; python scripts/build_user_questions.py
"""
import json
import random

from graphrag import config

config.use_vector_rag()
from app.rag.corpus import get_corpus  # noqa: E402

SEED = 20261110
DRAWS = {"A_article_long": 2, "B_multi_hop": 3, "C_relationnelle": 2, "D_piege": 2}

DA, DMA, AI, GDPR, EUDPR, NIS2, DSA, EPR, CSA = ("32023R2854", "32022R1925", "32024R1689", "32016R0679", "32018R1725", "32022L2555",
                                                 "32022R2065", "32002L0058", "32019R0881")


def art(celex: str, *nums) -> list[str]:
    return [f"{celex}_art_{n}" for n in nums]


QUESTIONS = [
    ("A1", "A_article_long", "Un contrôleur d'accès peut-il recevoir des données au titre du droit de partage du Data Act ?",
     art(DA, 5) + art(DMA, 3), "Non, ce n'est pas un tiers éligible (Data Act art. 5)."),
    ("A2", "A_article_long", "Un fournisseur d'IA à haut risque peut-il traiter des données sensibles pour corriger des biais ?",
     art(AI, 10), "Exceptionnellement, sous conditions cumulatives (AI Act art. 10)."),
    ("A3", "A_article_long", "Comment les droits du Data Act s'articulent-ils avec les droits d'accès et de portabilité du RGPD ?",
     art(DA, 1) + art(GDPR, 15, 20), "Ils les complètent ; en cas de conflit, le droit de la protection des données prévaut (Data Act art. 1)."),
    ("A4", "A_article_long", "Un tiers qui reçoit des données d'un produit connecté peut-il faire du profilage ?",
     art(DA, 6) + art(GDPR, 22), "Non, sauf si c'est nécessaire au service demandé (Data Act art. 6)."),
    ("A5", "A_article_long", "Qui peut infliger une amende quand une institution de l'UE viole le chapitre V du Data Act ?",
     art(DA, 40) + art(EUDPR, 66), "Le Contrôleur européen de la protection des données (Data Act art. 40)."),
    ("B1", "B_multi_hop", "Une entité NIS2 déjà sanctionnée au titre du RGPD peut-elle recevoir une seconde amende pour le même comportement ?",
     art(NIS2, 35) + art(GDPR, 58), "Pas d'amende au titre de l'article 34 de NIS2, mais d'autres mesures d'exécution restent possibles (NIS2 art. 35)."),
    ("B2", "B_multi_hop", "Quelle autorité faut-il informer quand un manquement NIS2 peut constituer une violation de données ?",
     art(NIS2, 35) + art(GDPR, 33, 55), "Les autorités de contrôle du RGPD (NIS2 art. 35)."),
    ("B3", "B_multi_hop", "Les très grandes plateformes doivent-elles proposer un fil de recommandation sans profilage, et qu'est-ce que le profilage ?",
     art(DSA, 38) + art(GDPR, 4), "Oui, au moins une option sans profilage (DSA art. 38) ; définition à l'article 4, point 4, du RGPD."),
    ("B4", "B_multi_hop", "Peut-on montrer à un mineur de la publicité fondée sur le profilage ?",
     art(DSA, 28) + art(GDPR, 4), "Non, si la plateforme sait avec une certitude raisonnable qu'il est mineur (DSA art. 28)."),
    ("B5", "B_multi_hop", "À quelles conditions des données personnelles peuvent-elles être réutilisées dans un bac à sable réglementaire de l'IA ?",
     art(AI, 59) + art(GDPR, 35), "Conditions de l'article 59 de l'AI Act (dont l'analyse d'impact au sens de l'article 35 du RGPD)."),
    ("B6", "B_multi_hop", "Un utilisateur peut-il faire transmettre à un tiers des données personnelles issues d'un objet connecté ?",
     art(DA, 5) + art(GDPR, 6, 9) + art(EPR, 5), "Seulement avec un fondement juridique valable (Data Act art. 5, RGPD art. 6 et 9, ePrivacy art. 5)."),
    ("B7", "B_multi_hop", "Les entités NIS2 peuvent-elles être obligées d'utiliser des produits certifiés, et selon quels schémas ?",
     art(NIS2, 24) + art(CSA, 49), "Oui, les États membres peuvent l'exiger, selon les schémas européens adoptés au titre de l'article 49 du règlement 2019/881."),
    ("C1", "C_relationnelle", "Quel acte du corpus modifie la directive 2011/24/UE ?", ["32025R0327_art_103"], "Le règlement (UE) 2025/327."),
    ("C2", "C_relationnelle", "Quel acte du corpus modifie le règlement 2019/1020 ?", ["32024R2847_art_66"], "Le règlement (UE) 2024/2847."),
    ("C3", "C_relationnelle", "Quels actes du corpus modifient le règlement 2017/2394 ?", ["32019L0770_art_23", "32023R2854_art_47"],
     "La directive (UE) 2019/770 et le règlement (UE) 2023/2854."),
    ("C4", "C_relationnelle", "Quel acte du corpus modifie le règlement Europol 2016/794 ?", ["32022R0991_art_1"], "Le règlement (UE) 2022/991."),
    ("C5", "C_relationnelle", "Quel acte du corpus modifie la directive 2001/29/CE ?", ["32019L0790_art_24"], "La directive (UE) 2019/790."),
    ("D1", "D_piege", "Que dit l'article 200 du RGPD ?", [], "Il n'existe pas : le RGPD compte 99 articles."),
    ("D2", "D_piege", "Quel montant d'amende la CNIL a-t-elle infligé à Google en 2022 ?", [], "Absent du corpus (droit européen uniquement)."),
    ("D3", "D_piege", "Combien de temps un cookie peut-il être conservé selon la loi française ?", [], "Absent du corpus."),
    ("D4", "D_piege", "Pourquoi l'AI Act interdit-il tous les systèmes d'IA à haut risque ?", [], "Fausse prémisse : ils sont encadrés, pas interdits."),
]


def main():
    corpus = get_corpus()
    missing = [(q[0], g) for q in QUESTIONS for g in q[3] if g not in corpus.articles]
    if missing:
        raise SystemExit(f"Articles attendus absents du corpus : {missing}")

    rng = random.Random(SEED)
    drawn = set()
    for section, n in DRAWS.items():
        ids = [q[0] for q in QUESTIONS if q[1] == section]
        drawn.update(rng.sample(ids, n))

    out = config.DATA_DIR / "questions_utilisateur_21.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for qid, section, question, gold, reference in QUESTIONS:
            f.write(json.dumps({"id": qid, "type": section, "question": question, "gold": gold, "reference": reference,
                                "reponse_llm": qid in drawn}, ensure_ascii=False) + "\n")
    print(f"{len(QUESTIONS)} questions écrites dans {out}")
    print(f"Tirage (graine {SEED}) : {sorted(drawn)}")


if __name__ == "__main__":
    main()
