import pytest

from app.models import DocumentIn
from app.retrieval import SearchHit
from app.store import DocumentStore
from app.synthesis import GroundedSynthesizer


@pytest.fixture
def document_store():
    store = DocumentStore()
    store.add(DocumentIn(
        document_id="sequencing",
        title="Sequencing quality",
        text="RNA sequencing quality includes read quality and adapter contamination.",
    ))
    store.add(DocumentIn(
        document_id="evidence",
        title="Evidence logging",
        text="Evidence assistants should log source passages and request identifiers.",
    ))
    return store


@pytest.mark.parametrize("document_id, question", [
    ("sequencing", "What affects RNA sequencing quality?"),
    ("evidence", "What should evidence assistants log?"),
])
def test_answer_citations_match_current_store(document_store, document_id, question):
    passage = next(
        passage for passage in document_store.passages()
        if passage.document_id == document_id
    )
    hit = SearchHit(
        passage=passage,
        score=0.5,
        confidence=0.8,
        lexical_rank=1,
        tfidf_rank=1,
    )
    _, citations, abstained = GroundedSynthesizer().answer(question, [hit])

    assert abstained is False
    assert citations, "Every non-abstained answer must include a citation"

    passages = {passage.passage_id: passage for passage in document_store.passages()}
    for citation in citations:
        assert citation.passage_id in passages
        passage = passages[citation.passage_id]
        assert citation.document_id == passage.document_id
        assert citation.excerpt == passage.text[:240].strip()


def test_answer_without_hits_abstains_without_citations():
    _, citations, abstained = GroundedSynthesizer().answer("unknown question", [])

    assert abstained is True
    assert citations == []


def test_low_confidence_answer_abstains_without_citations(document_store):
    passage = document_store.passages()[0]
    low_confidence_hit = SearchHit(
        passage=passage,
        score=0.5,
        confidence=0.01,
        lexical_rank=1,
        tfidf_rank=1,
    )

    _, citations, abstained = GroundedSynthesizer().answer(
        "unrelated question", [low_confidence_hit]
    )

    assert abstained is True
    assert citations == []
