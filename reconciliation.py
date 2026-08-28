from dataclasses import dataclass
from typing import Any


@dataclass
class SourceClaim:
    source: str
    source_id: str
    claim: str


@dataclass
class ReconciliationResult:
    agreements: list[str]
    conflicts: list[dict[str, Any]]
    unresolved: list[str]


def reconcile_claims(
    claims: list[SourceClaim],
) -> ReconciliationResult:

    # V1 of reconciliation:
    # keep source claims together and explicitly flag
    # potentially conflicting claims.
    #
    # We will make this smarter after the basic flow works.

    grouped: dict[str, list[SourceClaim]] = {}

    for claim in claims:
        grouped.setdefault(claim.claim, []).append(claim)

    agreements = []
    conflicts = []
    unresolved = []

    for claim_text, sources in grouped.items():

        if len(sources) > 1:
            agreements.append(
                claim_text
            )

    return ReconciliationResult(
        agreements=agreements,
        conflicts=conflicts,
        unresolved=unresolved,
    )