from __future__ import annotations

from app.models import Citation
from app.store import Passage


def validate_citations(citations: list[Citation], passages: list[Passage]) -> list[str]:
    """Return human-readable integrity errors for citations against indexed passages."""
    by_id = {passage.passage_id: passage for passage in passages}
    errors: list[str] = []

    for citation in citations:
        passage = by_id.get(citation.passage_id)
        if passage is None:
            errors.append(f"unknown passage_id: {citation.passage_id}")
            continue

        if citation.document_id != passage.document_id:
            errors.append(
                f"document mismatch for {citation.passage_id}: "
                f"{citation.document_id} != {passage.document_id}"
            )

        if citation.title != passage.title:
            errors.append(f"title mismatch for {citation.passage_id}")

        excerpt = citation.excerpt.strip()
        if excerpt and excerpt not in passage.text:
            errors.append(f"excerpt mismatch for {citation.passage_id}")

    return errors
