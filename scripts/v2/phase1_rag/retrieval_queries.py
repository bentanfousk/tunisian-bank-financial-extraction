from __future__ import annotations

from typing import Any


def build_retrieval_queries(
    bank: str,
    fiscal_year: int,
) -> list[dict[str, Any]]:
    """
    Build the two fixed retrieval queries used for every
    report. Only the bank and fiscal year vary.

    The wording mirrors canonical financial-statement
    headings and row labels commonly present on the
    authoritative table pages.
    """

    if not bank.strip():
        raise ValueError("Bank cannot be empty.")

    if fiscal_year < 1900:
        raise ValueError(
            "Fiscal year is not valid."
        )

    balance_sheet_query = (
        f"Banque {bank}. Exercice {fiscal_year}. "
        "Page du bilan individuel arrêté au "
        f"31 décembre {fiscal_year}. Rubriques ACTIF, "
        "PASSIF et CAPITAUX PROPRES : total actif, "
        "créances sur la clientèle, dépôts et avoirs "
        "de la clientèle et total des capitaux propres."
    )

    income_statement_query = (
        f"Banque {bank}. Exercice {fiscal_year}. "
        "Page de l'état de résultat individuel pour "
        f"l'exercice clos le 31 décembre {fiscal_year}. "
        "Produits et charges d'exploitation bancaire, "
        "produit net bancaire, résultat d'exploitation "
        "et résultat net de l'exercice."
    )

    return [
        {
            "query_id": "balance_sheet",
            "text": balance_sheet_query,
        },
        {
            "query_id": "income_statement",
            "text": income_statement_query,
        },
    ]