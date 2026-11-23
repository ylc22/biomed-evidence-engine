from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from app.models import DocumentIn
from app.retrieval import HybridRetriever
from app.store import DocumentStore


# Fixed inputs keep corpus construction reproducible and avoid network/API calls.
_TOPICS: tuple[tuple[str, str], ...] = (
    (
        "sequencing quality control",
        "RNA sequencing quality control reviews read quality, adapter contamination, "
        "mapping rate, duplication, and sample acceptance thresholds.",
    ),
    (
        "model evaluation",
        "Scientific model evaluation uses a held-out dataset, calibration checks, "
        "minority-class metrics, and explicit release criteria.",
    ),
    (
        "evidence citations",
        "An evidence assistant links each answer to stable source passage identifiers "
        "and abstains when retrieved evidence is insufficient.",
    ),
    (
        "assay operations",
        "Assay operations require versioned protocols, acceptance criteria, and "
        "traceable deviations with human review for ambiguous results.",
    ),
    (
        "data governance",
        "Scientific data governance records provenance, access decisions, retention "
        "rules, and audit events without exposing sensitive document contents.",
    ),
    (
        "retrieval monitoring",
        "Retrieval monitoring tracks query latency, recall, ranking changes, and "
        "service errors across a fixed evaluation set.",
    ),
)

_QUERIES: tuple[str, ...] = (
    "RNA sequencing adapter contamination and read quality",
    "held out scientific model calibration evaluation",
    "stable evidence citations and abstention",
    "assay acceptance criteria and protocol deviations",
    "data provenance audit and retention",
    "retrieval query latency ranking monitoring",
)


def build_synthetic_store(document_count: int = 24) -> DocumentStore:
    """Build a deterministic, local corpus with relevant and distractor passages."""
    if document_count < 1:
        raise ValueError("document_count must be at least 1")

    store = DocumentStore()
    for index in range(document_count):
        topic, evidence = _TOPICS[index % len(_TOPICS)]
        # The fixed suffix varies documents without introducing randomness.
        text = (
            f"{evidence}\n\n"
            f"Benchmark record {index:04d} covers {topic}. "
            "Review the documented evidence and preserve reproducible decisions."
        )
        store.add(DocumentIn(
            document_id=f"synthetic-{index:04d}",
            title=f"Synthetic {topic.title()} Record {index:04d}",
            text=text,
        ))
    return store


def percentile_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """Return nearest-rank-style NumPy percentile summaries for measured samples."""
    if not samples_ms:
        raise ValueError("at least one latency sample is required")
    values = np.asarray(samples_ms, dtype=float)
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }


def run_benchmark(
    *,
    document_count: int = 24,
    iterations: int = 30,
    warmups: int = 5,
) -> dict[str, int | float | dict[str, float]]:
    """Measure repeated HybridRetriever.search calls; corpus setup is excluded."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")

    store = build_synthetic_store(document_count)
    retriever = HybridRetriever(store)

    for index in range(warmups):
        retriever.search(_QUERIES[index % len(_QUERIES)], top_k=3)

    samples_ms: list[float] = []
    for index in range(iterations):
        query = _QUERIES[index % len(_QUERIES)]
        started = time.perf_counter_ns()
        retriever.search(query, top_k=3)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    return {
        "corpus_documents": document_count,
        "corpus_passages": len(store.passages()),
        "iterations": iterations,
        "warmups": warmups,
        "query_count": len(_QUERIES),
        "latency": percentile_summary(samples_ms),
    }


def check_regression(p95_ms: float, threshold_ms: float | None) -> bool:
    """Return True when the measured p95 exceeds a configured threshold."""
    return threshold_ms is not None and p95_ms > threshold_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure local hybrid retrieval latency on a deterministic synthetic corpus."
    )
    parser.add_argument("--documents", type=int, default=24, help="synthetic document count")
    parser.add_argument("--iterations", type=int, default=30, help="measured retrieval calls")
    parser.add_argument("--warmups", type=int, default=5, help="unmeasured warm-up calls")
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        default=None,
        help="optional regression limit; exits non-zero when measured p95 exceeds it",
    )
    args = parser.parse_args()

    threshold = args.max_p95_ms
    if threshold is None:
        configured = os.getenv("RETRIEVAL_P95_THRESHOLD_MS")
        threshold = float(configured) if configured is not None else None
    if threshold is not None and threshold < 0:
        parser.error("--max-p95-ms / RETRIEVAL_P95_THRESHOLD_MS must be non-negative")

    result = run_benchmark(
        document_count=args.documents,
        iterations=args.iterations,
        warmups=args.warmups,
    )
    p95_ms = float(result["latency"]["p95_ms"])  # type: ignore[index]
    result["regression"] = {
        "threshold_p95_ms": threshold,
        "exceeded": check_regression(p95_ms, threshold),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if check_regression(p95_ms, threshold) else 0


if __name__ == "__main__":
    raise SystemExit(main())
