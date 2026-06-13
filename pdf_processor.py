from typing import List
from io import BytesIO
from PyPDF2 import PdfReader


def extract_text_from_pdf(file) -> str:
    """Extracts text from a PDF file-like object."""
    if hasattr(file, "read"):
        data = file.read()
        bio = BytesIO(data)
        reader = PdfReader(bio)
    else:
        reader = PdfReader(file)

    texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        texts.append(txt)
    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200, max_chunks: int = 1000) -> List[str]:
    """Split large text into overlapping chunks for embedding/searching.

    Defensive: limit the maximum number of chunks to avoid runaway memory usage
    when processing malformed or extremely large inputs. Returns a list of
    text chunks (up to `max_chunks`).
    """
    if not text:
        return []

    try:
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length and len(chunks) < max_chunks:
            end = min(start + chunk_size, length)
            chunk = text[start:end]
            chunks.append(chunk)
            # compute next start with overlap
            next_start = end - overlap
            if next_start <= start:
                # no forward progress possible (overlap >= chunk_size or small text)
                break
            start = next_start
            if start < 0:
                start = 0
            if start >= length:
                break

        return chunks
    except MemoryError:
        # Defensive: if memory is exhausted during chunking, return empty list
        return []
