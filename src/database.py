from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _connect(db_path: Path | str):
    """Open a SQLite connection and always close it.

    sqlite3.Connection's own context-manager commits/rolls back transactions,
    but it does not close the connection. That can keep temporary .db files
    locked on Windows and causes WinError 32 during test cleanup.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database(db_path: Path | str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(doc_type)")


def replace_source_chunks(db_path: Path | str, source: str, doc_type: str, records: list[dict]) -> None:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        for record in records:
            vector = np.asarray(record["embedding"], dtype=np.float32)
            conn.execute(
                """
                INSERT INTO chunks(source, doc_type, page, chunk_index, content, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    doc_type,
                    record.get("page"),
                    int(record["chunk_index"]),
                    record["content"],
                    vector.tobytes(),
                    int(vector.size),
                ),
            )


def count_sources(db_path: Path | str) -> int:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(DISTINCT source) AS n FROM chunks").fetchone()
        return int(row["n"])


def count_chunks(db_path: Path | str) -> int:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])


def list_sources(db_path: Path | str) -> list[dict]:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT source, doc_type, COUNT(*) AS chunk_count
            FROM chunks
            GROUP BY source, doc_type
            ORDER BY doc_type, source
            """
        ).fetchall()
        return [dict(row) for row in rows]


def delete_source(db_path: Path | str, source: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chunks WHERE source = ?", (source,))


def reset_database(db_path: Path | str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chunks")


def source_names(db_path: Path | str, doc_type: str | None = None) -> list[str]:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        if doc_type:
            rows = conn.execute(
                "SELECT DISTINCT source FROM chunks WHERE doc_type = ? ORDER BY source",
                (doc_type,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT DISTINCT source FROM chunks ORDER BY source").fetchall()
        return [str(row["source"]) for row in rows]


def _merge_overlapping_chunks(chunks: list[str], max_overlap_words: int = 80) -> str:
    """Reconstruct readable source text without repeating chunk overlap."""
    if not chunks:
        return ""

    merged_words = chunks[0].split()
    for chunk in chunks[1:]:
        current = chunk.split()
        max_check = min(max_overlap_words, len(merged_words), len(current))
        overlap = 0
        for size in range(max_check, 0, -1):
            if merged_words[-size:] == current[:size]:
                overlap = size
                break
        merged_words.extend(current[overlap:])
    return " ".join(merged_words)


def get_source_text(db_path: Path | str, source: str, max_chars: int = 14000) -> str:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE source = ? ORDER BY chunk_index",
            (source,),
        ).fetchall()
    text = _merge_overlapping_chunks([str(row["content"]) for row in rows])
    return text[:max_chars]


def search_chunks(
    db_path: Path | str,
    query_embedding: list[float] | np.ndarray,
    top_k: int = 4,
    doc_types: list[str] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    initialize_database(db_path)
    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        return []

    where: list[str] = []
    params: list[str] = []
    if doc_types:
        where.append("doc_type IN (%s)" % ",".join("?" for _ in doc_types))
        params.extend(doc_types)
    if sources:
        where.append("source IN (%s)" % ",".join("?" for _ in sources))
        params.extend(sources)

    sql = "SELECT source, doc_type, page, chunk_index, content, embedding, embedding_dim FROM chunks"
    if where:
        sql += " WHERE " + " AND ".join(where)

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    scored: list[dict] = []
    for row in rows:
        if int(row["embedding_dim"]) != int(query.size):
            continue
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        denom = query_norm * float(np.linalg.norm(vector))
        score = float(np.dot(query, vector) / denom) if denom else 0.0
        scored.append(
            {
                "source": row["source"],
                "doc_type": row["doc_type"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: max(int(top_k), 1)]


def search_chunks_text(
    db_path: Path | str,
    query_text: str,
    top_k: int = 4,
    doc_types: list[str] | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    """Fast lexical retrieval for interactive Q&A.

    Query-time embedding model loads were the main latency source on CPU. The
    documents are still embedded with Foundry Local at ingestion, but interactive
    questions use a lightweight local token/phrase ranker before generation.
    """
    from .text_utils import tokenize_keywords

    initialize_database(db_path)
    where: list[str] = []
    params: list[str] = []
    if doc_types:
        where.append("doc_type IN (%s)" % ",".join("?" for _ in doc_types))
        params.extend(doc_types)
    if sources:
        where.append("source IN (%s)" % ",".join("?" for _ in sources))
        params.extend(sources)

    sql = "SELECT source, doc_type, page, chunk_index, content FROM chunks"
    if where:
        sql += " WHERE " + " AND ".join(where)

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    q_tokens = tokenize_keywords(query_text)
    q_set = set(q_tokens)
    q_norm = " ".join(q_tokens)
    scored: list[dict] = []

    for row in rows:
        content = str(row["content"])
        c_tokens = tokenize_keywords(content)
        c_set = set(c_tokens)
        overlap = q_set & c_set

        # Coverage rewards chunks that contain the user's important terms, while
        # density keeps giant generic chunks from dominating.
        coverage = len(overlap) / max(len(q_set), 1) if q_set else 0.0
        density = len(overlap) / max(min(len(c_set), 18), 1) if overlap else 0.0
        phrase_bonus = 0.0
        content_lower = content.lower()
        if q_norm and len(q_tokens) >= 2 and q_norm in " ".join(c_tokens):
            phrase_bonus = 0.22
        elif any(len(tok) >= 4 and tok in content_lower for tok in q_set):
            phrase_bonus = 0.06

        score = min(1.0, 0.78 * coverage + 0.22 * min(density * 3.0, 1.0) + phrase_bonus)
        scored.append(
            {
                "source": row["source"],
                "doc_type": row["doc_type"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "content": content,
                "score": float(score),
            }
        )

    scored.sort(key=lambda item: (item["score"], -int(item["chunk_index"])), reverse=True)

    # For broad questions (e.g. "CV'de neler geliştirilmeli?") there may be no
    # literal overlap. Return early chunks deterministically rather than loading
    # an embedding model just to rescue the query.
    if scored and scored[0]["score"] <= 0.0:
        scored.sort(key=lambda item: (item["source"], int(item["chunk_index"])))

    return scored[: max(int(top_k), 1)]
