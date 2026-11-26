from app.models import Citation, DocumentIn
from app.store import DocumentStore
from app.validation import validate_citations


def _store() -> DocumentStore:
    store = DocumentStore()
    store.add(DocumentIn(
        document_id="evidence",
        title="Evidence Policy",
        text="Every grounded answer should cite a stable source passage. "
             "Unsupported questions should abstain.",
    ))
    return store


def test_valid_citation_matches_indexed_passage():
    store = _store()
    passage = store.passages()[0]
    citation = Citation(
        document_id=passage.document_id,
        passage_id=passage.passage_id,
        title=passage.title,
        score=0.1,
        excerpt="Every grounded answer should cite a stable source passage.",
    )

    assert validate_citations([citation], store.passages()) == []


def test_unknown_passage_is_rejected():
    store = _store()
    citation = Citation(
        document_id="evidence",
        passage_id="evidence:missing",
        title="Evidence Policy",
        score=0.1,
        excerpt="Every grounded answer should cite a stable source passage.",
    )

    errors = validate_citations([citation], store.passages())

    assert errors == ["unknown passage_id: evidence:missing"]


def test_document_and_excerpt_mismatches_are_reported():
    store = _store()
    passage = store.passages()[0]
    citation = Citation(
        document_id="wrong-document",
        passage_id=passage.passage_id,
        title=passage.title,
        score=0.1,
        excerpt="This text never appeared in the source.",
    )

    errors = validate_citations([citation], store.passages())

    assert any("document mismatch" in error for error in errors)
    assert any("excerpt mismatch" in error for error in errors)
