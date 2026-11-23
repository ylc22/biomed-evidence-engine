import pytest

from evals.benchmark import (
    build_synthetic_store,
    check_regression,
    percentile_summary,
    run_benchmark,
)


def test_synthetic_corpus_is_deterministic_and_has_expected_size():
    first = build_synthetic_store(8)
    second = build_synthetic_store(8)

    assert len(first.passages()) == 8
    assert [
        (p.document_id, p.passage_id, p.text) for p in first.passages()
    ] == [
        (p.document_id, p.passage_id, p.text) for p in second.passages()
    ]


def test_percentile_summary_reports_p50_and_p95():
    result = percentile_summary([1, 2, 3, 4, 5])

    assert result == {"p50_ms": 3.0, "p95_ms": 4.8}


def test_percentile_summary_rejects_empty_samples():
    with pytest.raises(ValueError, match="at least one"):
        percentile_summary([])


def test_regression_threshold_is_optional_and_strict():
    assert check_regression(10.0, None) is False
    assert check_regression(10.0, 10.0) is False
    assert check_regression(10.01, 10.0) is True


def test_benchmark_smoke_run_returns_latency_metrics():
    result = run_benchmark(document_count=8, iterations=3, warmups=1)

    assert result["corpus_documents"] == 8
    assert result["corpus_passages"] == 8
    assert result["iterations"] == 3
    assert result["latency"]["p50_ms"] >= 0
    assert result["latency"]["p95_ms"] >= result["latency"]["p50_ms"]


def test_benchmark_rejects_invalid_iteration_count():
    with pytest.raises(ValueError, match="iterations"):
        run_benchmark(iterations=0)
