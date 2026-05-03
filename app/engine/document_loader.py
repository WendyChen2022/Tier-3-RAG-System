"""PDF ingestion via Docling with configurable parsing profiles.

Two pipeline modes
------------------
FAST (default, ``use_ocr=False``)
    Extracts the embedded text layer directly.  Typical throughput: 2–5 s for
    a 13-page document.  Correct choice for born-digital / text-based PDFs.

FULL (``use_ocr=True``)
    Runs RapidOCR on every page image.  Required for scanned / image-only PDFs
    but adds ~5–7 s per page (≈ 71 s for 13 pages).  Set ``USE_OCR=true`` in
    your ``.env`` to activate.
"""
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


def _build_pipeline_options() -> PdfPipelineOptions:
    """Return PdfPipelineOptions driven by the Settings object."""
    opts = PdfPipelineOptions()
    opts.do_ocr              = settings.use_ocr
    opts.do_table_structure  = True          # always keep – needed for RAG accuracy
    opts.generate_page_images = settings.generate_page_images
    return opts


def _build_converter() -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_build_pipeline_options())
        }
    )


class DoclingPDFLoader:
    """Converts PDFs to LangChain Documents using Docling's layout-aware parser.

    The converter is instantiated lazily so that heavy model weights (table
    detector, OCR engine) are only loaded on first use.
    """

    def __init__(self) -> None:
        self._converter: DocumentConverter | None = None

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            mode = "FULL (OCR)" if settings.use_ocr else "FAST (text-layer)"
            logger.info(f"[DoclingPDFLoader] Initialising converter — mode={mode}")
            self._converter = _build_converter()
        return self._converter

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        logger.info(f"[DoclingPDFLoader] Loading: {path.name}")

        result = self._get_converter().convert(str(path))
        markdown_text = result.document.export_to_markdown()
        num_pages = len(result.document.pages)

        logger.info(
            f"[DoclingPDFLoader] Done — pages={num_pages}  chars={len(markdown_text):,}"
        )
        return [
            Document(
                page_content=markdown_text,
                metadata={
                    "source":    str(path),
                    "filename":  path.name,
                    "num_pages": num_pages,
                    "ocr_used":  settings.use_ocr,
                },
            )
        ]
