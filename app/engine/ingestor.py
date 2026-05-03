"""End-to-end ingestion pipeline: PDF bytes → structured chunks → vector store.

Chunking strategy
-----------------
When ``use_hybrid_chunker=True`` (default) the pipeline uses Docling's
``HybridChunker``, which respects the document's structural hierarchy
(headings, paragraphs, tables, captions) and merges undersized sibling chunks.
This reliably produces 20–40 meaningful chunks per 10–15 page document versus
the 1–3 coarse chunks produced by a naive character-based splitter applied to
raw Markdown.

When ``use_hybrid_chunker=False`` the pipeline falls back to LangChain's
``RecursiveCharacterTextSplitter`` for simple or non-PDF sources.
"""
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.core.config import get_settings
from app.engine.document_loader import DoclingPDFLoader, _build_converter

settings = get_settings()


def _hybrid_chunks(raw_path: Path, filename: str) -> list[Document]:
    """Parse with Docling and chunk with HybridChunker.

    Tokenizer strategy
    ------------------
    We load the SentenceTransformer model (already cached on disk) and extract its
    BertTokenizer, then wrap it in ``HuggingFaceTokenizer``.  This avoids any
    HuggingFace network download — the tokenizer bytes are already present in the
    local sentence-transformers cache.

    Fallback
    --------
    If the HybridChunker cannot be initialised for any reason, the function raises
    so that the caller (``DocumentIngestor.ingest_bytes``) can catch it and invoke
    ``_splitter_chunks`` instead.
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from sentence_transformers import SentenceTransformer

    converter = _build_converter()
    result    = converter.convert(str(raw_path))
    doc       = result.document

    # Extract BertTokenizer from locally-cached SentenceTransformer — no network.
    _st_model      = SentenceTransformer(settings.hybrid_chunker_tokenizer)
    _bert_tokenizer = _st_model.tokenizer
    hft = HuggingFaceTokenizer(
        tokenizer=_bert_tokenizer,
        max_tokens=settings.hybrid_chunker_max_tokens,
    )

    chunker = HybridChunker(
        tokenizer=hft,
        merge_peers=settings.hybrid_chunker_merge_peers,
    )

    lc_docs: list[Document] = []
    for idx, chunk in enumerate(chunker.chunk(doc)):
        text = chunk.text or ""
        if not text.strip():
            continue

        # Extract heading breadcrumb and page number from DocMeta
        headings: list[str] = getattr(chunk.meta, "headings", []) or []
        page_no = None
        try:
            page_no = chunk.meta.doc_items[0].prov[0].page_no
        except Exception:
            pass

        lc_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source":    filename,
                    "filename":  filename,
                    "chunk_idx": idx,
                    "headings":  " > ".join(headings) if headings else "",
                    "page":      page_no,
                    "chars":     len(text),
                },
            )
        )

    logger.info(
        f"[HybridChunker] {len(lc_docs)} chunks  "
        f"(max_tokens={settings.hybrid_chunker_max_tokens}, "
        f"merge_peers={settings.hybrid_chunker_merge_peers})"
    )
    return lc_docs


def _splitter_chunks(docs: list[Document]) -> list[Document]:
    """Fallback: LangChain RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        f"[RecursiveCharacterTextSplitter] {len(chunks)} chunks  "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks


class DocumentIngestor:
    def __init__(self, vector_store) -> None:
        self._vector_store = vector_store
        self._loader       = DoclingPDFLoader()

    async def ingest_bytes(self, data: bytes, filename: str) -> int:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            if settings.use_hybrid_chunker:
                try:
                    chunks = _hybrid_chunks(tmp_path, filename)
                except Exception as hybrid_exc:
                    logger.warning(
                        f"[DocumentIngestor] HybridChunker failed ({hybrid_exc}) — "
                        "falling back to RecursiveCharacterTextSplitter"
                    )
                    docs   = self._loader.load(tmp_path)
                    chunks = _splitter_chunks(docs)
                    for c in chunks:
                        c.metadata["filename"] = filename
            else:
                docs   = self._loader.load(tmp_path)
                chunks = _splitter_chunks(docs)
                for c in chunks:
                    c.metadata["filename"] = filename

            self._vector_store.add_documents(chunks)
            logger.info(f"[DocumentIngestor] Stored {len(chunks)} chunks for '{filename}'")
            return len(chunks)
        finally:
            tmp_path.unlink(missing_ok=True)
