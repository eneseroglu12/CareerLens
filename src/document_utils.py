from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader

from .text_utils import normalize_text


def _chunk_words(text: str, max_words: int = 180, overlap_words: int = 30) -> list[str]:
    """Kelime bazlı overlap uygular; başlık/satır yapısını mümkün olduğunca korur."""
    text = normalize_text(text)
    if not text:
        return []

    # Satır sonlarını ayrı token olarak tut. Böylece "GEREKSİNİMLER\n- SQL..."
    # yapısı embedding ve daha sonraki deterministic parser için kaybolmaz.
    tokens = re.findall(r"\n+|[^\s]+", text)
    word_token_positions = [i for i, token in enumerate(tokens) if not token.startswith("\n")]
    word_count = len(word_token_positions)
    if word_count == 0:
        return []
    if word_count <= max_words:
        return [text]

    def render(token_slice: list[str]) -> str:
        out = ""
        for token in token_slice:
            if token.startswith("\n"):
                out = out.rstrip() + "\n"
            else:
                if out and not out.endswith("\n"):
                    out += " "
                out += token
        return normalize_text(out)

    chunks: list[str] = []
    step = max(max_words - overlap_words, 1)
    for start_word in range(0, word_count, step):
        end_word = min(start_word + max_words, word_count)
        token_start = word_token_positions[start_word]
        token_end = (
            word_token_positions[end_word]
            if end_word < word_count
            else len(tokens)
        )
        part = render(tokens[token_start:token_end])
        if part:
            chunks.append(part)
        if end_word >= word_count:
            break
    return chunks


def _decode_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def extract_chunks(filename: str, raw_bytes: bytes, doc_type: str) -> list[dict]:
    extension = Path(filename).suffix.lower()
    chunks: list[dict] = []
    chunk_index = 0

    if extension == ".pdf":
        reader = PdfReader(BytesIO(raw_bytes))
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            for part in _chunk_words(text):
                chunks.append(
                    {
                        "source": filename,
                        "doc_type": doc_type,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "content": part,
                    }
                )
                chunk_index += 1

    elif extension == ".docx":
        document = Document(BytesIO(raw_bytes))
        text = normalize_text("\n".join(p.text for p in document.paragraphs if p.text.strip()))
        for part in _chunk_words(text):
            chunks.append(
                {
                    "source": filename,
                    "doc_type": doc_type,
                    "page": None,
                    "chunk_index": chunk_index,
                    "content": part,
                }
            )
            chunk_index += 1

    elif extension in {".txt", ".md"}:
        text = normalize_text(_decode_text(raw_bytes))
        for part in _chunk_words(text):
            chunks.append(
                {
                    "source": filename,
                    "doc_type": doc_type,
                    "page": None,
                    "chunk_index": chunk_index,
                    "content": part,
                }
            )
            chunk_index += 1
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {extension or 'bilinmiyor'}")

    return chunks
