from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Evidence:
    source: str
    source_id: str
    content: str
    metadata: dict[str, Any]


def create_evidence(
    source: str,
    source_id: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = Evidence(
        source=source,
        source_id=source_id,
        content=content,
        metadata=metadata or {},
    )

    return asdict(evidence)