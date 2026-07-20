from __future__ import annotations

from canar.app.response_strategy import ResponseMode
from canar.app.retrieval.models import RetrievalHit

SYSTEM_PROMPT_FR = """
Tu es “CoachR”, un formateur R très pédagogue pour un public venant majoritairement de SAS et peu 
familier des langages de programmation.
Objectif: expliquer R clairement, donner des exemples courts, et guider vers de bonnes pratiques 
utilisées en statistiques (nettoyage, indicateurs, pondération, contrôles).

Règles de pédagogie:
1) Commence par une explication simple (2–5 phrases), puis un exemple minimal exécutable.
2) Fais systématiquement un parallèle SAS→R quand cela aide (DATA step vs dplyr, PROC SQL vs 
dplyr/dbplyr, formats vs factors/labels, macro vs fonctions).
3) Privilégie des exemples inspirés d’enquêtes (variables, modalités, indicatrices, 
pondération) sans inventer de données sensibles.
4) Si l’utilisateur est bloqué, pose 1–2 questions maximum, sinon propose une hypothèse 
et avance.
5) Toujours inclure:
   - “Exemple” (code R)
   - “À retenir” (3 bullets)
   - “Pièges fréquents (SAS→R)” (1–3 bullets) quand pertinent
6) Style:
   - Base R si nécessaire, sinon tidyverse pour la lisibilité.
   - Donne des noms d’objets parlants (df, individus, poids, etc.)
7) Ne donne pas d’infos non fondées; indique clairement ce qui est hypothèse.

Sortie: Markdown, blocs ```r```.
"""


def assemble_context(citations: list[RetrievalHit]) -> tuple[str, list[dict]]:
    """
    Map citations to labels [S1].. and return (context_text, source_list_for_ui)
    """
    lines = []
    srcs = []
    for i, hit in enumerate(citations, 1):
        label = f"S{i}"
        lines.append(f"[{label}] {hit.section}\n{hit.generation_text or hit.text}\n")
        srcs.append(
            {
                "label": label,
                "url": hit.source_url,
                "section": hit.section,
                "collection": hit.collection,
            }
        )
    context = "\n---\n".join(lines)
    return context, srcs


def build_messages(
    query: str,
    citations: list[RetrievalHit],
    response_mode: ResponseMode | None = None,
) -> tuple[list[dict], list[dict]]:
    if response_mode is ResponseMode.GENERAL_KNOWLEDGE_ONLY:
        context_text, src_list = "", []
    else:
        context_text, src_list = assemble_context(citations)

    if response_mode is None:
        user_msg = (
            f"Question: {query}\n\nContexte (extraits documentaires):\n{context_text}\n\n"
            "Consigne: Utilise uniquement les extraits pertinents. "
            "Cite [S1], [S2] si utilisés."
        )
    elif response_mode is ResponseMode.RAG_ONLY:
        user_msg = (
            f"Question: {query}\n\nContexte (extraits documentaires):\n{context_text}\n\n"
            "Consigne: Réponds entièrement et uniquement à partir des extraits pertinents. "
            "N'ajoute aucune information issue de tes connaissances générales. "
            "Cite [S1], [S2] pour chaque information utilisée."
        )
    elif response_mode is ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE:
        user_msg = (
            f"Question: {query}\n\nContexte (extraits documentaires):\n{context_text}\n\n"
            "Consigne: Réponds dans la langue de la question en séparant clairement la réponse "
            "en deux sections avec des titres dans cette même langue. La première section, "
            "« Informations issues de la base documentaire », doit utiliser les extraits et "
            "citer [S1], [S2] pour chaque information. La seconde section, « Complément fondé "
            "sur les connaissances générales du modèle », peut compléter les lacunes avec tes "
            "connaissances générales et ne doit contenir aucune citation documentaire. "
            "En cas de contradiction, les extraits documentaires prévalent et la divergence "
            "doit être signalée explicitement."
        )
    else:
        user_msg = (
            f"Question: {query}\n\n"
            "Consigne: Réponds dans la langue de la question uniquement à partir de tes "
            "connaissances générales. N'utilise aucun document récupéré, n'invente aucune "
            "source et n'ajoute aucune citation [S1], [S2]. L'avertissement sur l'absence "
            "de fondement documentaire est ajouté séparément par l'application : ne le répète pas."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": user_msg},
    ]
    return messages, src_list
