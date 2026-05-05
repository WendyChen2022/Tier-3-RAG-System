"""
Tier-3 Agentic RAG System — Streamlit Dashboard
=================================================
Professional UI featuring:
  • System health sidebar   (ChromaDB, Redis live status)
  • PDF ingestion panel     (upload → Docling → ChromaDB)
  • Agentic RAG query panel (self-correction loop, source chunks)
  • Collection inspector    (browse indexed chunks as a table)
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Config  (app runs locally; ChromaDB + Redis are in Docker)
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_URL      = "http://localhost:8001"
CHROMA_API      = f"{CHROMA_URL}/api/v1"
REDIS_HOST      = "localhost"
REDIS_PORT      = 6379
COLLECTION_NAME = "tier3_rag_system_spec"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 5
RELEVANCE_THRESHOLD       = 0.70
MAX_CORRECTION_ITERATIONS = 3

# ── Anthropic API key resolution (checked fresh on every LLM call) ───────────
# Priority:
#   1. ANTHROPIC_API_KEY environment variable  (set in shell / .env / Docker)
#   2. ANTHROPIC_API_KEY=  line in .env.example
#   3. Legacy OPENAI_API_KEY= line in .env.example  (backwards-compat)
_ENV_FILE = Path(__file__).parent / ".env.example"

def _get_api_key() -> str:
    """Return the Anthropic API key, reading os.environ fresh on every call.

    Raises ValueError with a clear message if the key is absent or is still
    the placeholder value so the UI can surface a useful error instead of a
    cryptic 401 AuthenticationError.
    """
    # 1. Environment variable (highest priority — works in all deployment modes)
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # 2 & 3. Fall back to .env.example file
    if not key and _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip()
                break
            if line.startswith("OPENAI_API_KEY="):   # legacy key name in this project
                key = line.split("=", 1)[1].strip()
                # keep scanning — a real ANTHROPIC_API_KEY line later takes precedence

    # Validate
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found.  "
            "Set it as an environment variable or add it to .env.example."
        )
    if key.startswith("sk-ant-api") is False and key.startswith("sk-") is False:
        # Catches placeholder strings like "your-key-here" or empty quotes
        if len(key) < 20:
            raise ValueError(
                f"ANTHROPIC_API_KEY looks like a placeholder ({key!r}).  "
                "Replace it with your real Anthropic API key."
            )
    return key

# Module-level convenience — used only for the sidebar "key present?" check.
# All LLM calls use _get_api_key() directly so they always get the live value.
ANTHROPIC_API_KEY: str = ""
try:
    ANTHROPIC_API_KEY = _get_api_key()
except ValueError:
    pass

LLM_MODEL            = "claude-haiku-4-5"
INGEST_TIMEOUT_SECS  = 600          # watchdog: 10 minutes (OCR on large PDFs needs ~5–8 min)
POLL_INTERVAL_SECS   = 0.6          # UI refresh cadence while worker runs
_EXECUTOR            = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

# ── Ingestion job state machine ──────────────────────────────────────────────
# States: "idle" | "running" | "done" | "error" | "timeout"
# Stored in st.session_state["ingest_job"] as a plain dict so Streamlit can
# survive across reruns.  Heavy Python objects (Queue, Event, Thread) live
# inside the dict — session_state is in-process memory, never serialised.

_JOB_DEFAULTS: dict = {
    "status":      "idle",
    "steps":       [],          # list[dict]  {msg, ts, level, elapsed}
    "result":      None,        # final metrics dict on success
    "error":       None,        # str on failure / timeout
    "start_time":  None,        # float – wall-clock start
    "_queue":      None,        # queue.Queue  worker → main
    "_cancel":     None,        # threading.Event
    "_future":     None,        # concurrent.futures.Future
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tier-3 Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main gradient header */
.rag-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.rag-header h1 { color: #e8f4fd; margin: 0; font-size: 2rem; }
.rag-header p  { color: #90caf9; margin: 0.4rem 0 0; font-size: 0.95rem; }

/* Status pills */
.status-ok   { background:#0d7f4a; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.status-fail { background:#b71c1c; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* Metric cards */
.metric-card {
    background:#1e2a3a; border:1px solid #2d3e50;
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
}
.metric-card .value { font-size:1.8rem; font-weight:700; color:#64b5f6; }
.metric-card .label { font-size:0.8rem; color:#90a4ae; margin-top:4px; }

/* Chunk cards */
.chunk-card {
    background:#1a2332; border-left:4px solid #0f3460;
    border-radius:6px; padding:0.9rem 1rem; margin-bottom:0.7rem;
    font-family: monospace; font-size: 0.82rem; color: #cdd9e5;
    white-space: pre-wrap; word-break: break-word;
}
.chunk-header {
    font-size:0.78rem; color:#64b5f6; margin-bottom:6px;
    display:flex; justify-content:space-between;
}
.sim-bar-wrap { background:#2d3e50; border-radius:4px; height:6px; margin-top:4px; }
.sim-bar      { background:#0f3460; border-radius:4px; height:6px; }

/* Answer box */
.answer-box {
    background: linear-gradient(135deg,#0d2137,#0f3460);
    border:1px solid #1565c0; border-radius:10px;
    padding:1.4rem 1.6rem; color:#e3f2fd; line-height:1.7;
    font-size:0.95rem;
}

/* Iteration log */
.iter-log {
    background:#111c2b; border:1px solid #263545;
    border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem;
    font-size:0.8rem; color:#b0bec5;
}
.iter-log .label { color:#64b5f6; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#111827; }

/* ── Thread-isolated ingestion step tracker ── */
.step-tracker {
    background:#0d1b2a; border:1px solid #1e3a5f;
    border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;
}
.step-row {
    display:flex; align-items:flex-start; gap:10px;
    padding:5px 0; border-bottom:1px solid #162030; font-size:0.85rem;
}
.step-row:last-child { border-bottom:none; }
.step-icon  { font-size:1rem; min-width:22px; margin-top:1px; }
.step-msg   { flex:1; color:#cdd9e5; }
.step-ts    { color:#546e7a; font-size:0.75rem; white-space:nowrap; }
.step-info    .step-icon::before { content: "⟳"; color:#64b5f6; }
.step-success .step-icon::before { content: "✓"; color:#4caf50; }
.step-warning .step-icon::before { content: "⚠"; color:#ff9800; }
.step-error   .step-icon::before { content: "✕"; color:#ef5350; }

/* Watchdog progress bar */
.watchdog-bar-wrap { background:#1e2a3a; border-radius:4px; height:8px; margin:6px 0 2px; }
.watchdog-bar      { border-radius:4px; height:8px; transition:width 0.5s; }

/* Disabled-button visual cue */
div[data-testid="stButton"] button:disabled {
    opacity:0.45; cursor:not-allowed;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embed_model():
    # Singleton pattern — model is loaded once and reused across all requests
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_docling_converter_v2():
    """Force-fresh DocumentConverter — pypdfium2 backend, OCR and table-structure OFF.

    Renamed to _v2 so Streamlit's cache_resource treats this as a brand-new
    resource and discards any previously cached converter instance.

    Backend: PyPdfiumDocumentBackend (pypdfium2) is set on PdfFormatOption —
    that is the correct Docling API; pdf_backend is NOT a PdfPipelineOptions
    field.  Falls back to Docling's default backend if the import is absent.

    OCR / table-structure are disabled because both trigger heavy ML model loads
    (RapidOCR, table-transformer) that cause 'bad allocation' crashes on machines
    with limited RAM.  Text-layer extraction via pypdfium2 is sufficient for
    born-digital PDFs and avoids the memory pressure entirely.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # ── Pipeline options ────────────────────────────────────────────────────
    # OCR and table-structure detection disabled — both are heavy ML passes
    # that trigger a 'bad allocation' memory crash on constrained machines.
    # pypdfium2 backend still used for reliable native text-layer extraction.
    opts = PdfPipelineOptions()
    opts.do_ocr               = False   # RapidOCR OFF — avoids memory crash
    opts.do_table_structure   = False   # table detector OFF — avoids memory crash
    opts.generate_page_images = False   # no page rasterisation needed

    # ── Backend: pypdfium2 for fast, low-memory native text extraction ──────
    backend_cls = None
    try:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        backend_cls = PyPdfiumDocumentBackend
        print("[Docling v2] pypdfium2 backend loaded ✓", flush=True)
    except ImportError:
        print("[Docling v2] pypdfium2 backend not found — using Docling default", flush=True)

    fmt_option = (
        PdfFormatOption(backend=backend_cls, pipeline_options=opts)
        if backend_cls is not None
        else PdfFormatOption(pipeline_options=opts)
    )

    print(
        f"[Docling v2] Initialising DocumentConverter — "
        f"OCR={opts.do_ocr}  table_structure={opts.do_table_structure}  "
        f"backend={'pypdfium2' if backend_cls else 'default'}",
        flush=True,
    )
    logger.info(
        f"[Docling v2] DocumentConverter ready — "
        f"OCR=off  table_structure=off  "
        f"backend={'pypdfium2' if backend_cls else 'default'}"
    )
    return DocumentConverter(format_options={InputFormat.PDF: fmt_option})


@st.cache_resource(show_spinner=False)
def load_hybrid_chunker():
    """Cache the HybridChunker, reusing the already-cached SentenceTransformer tokenizer.

    Token → character mapping (all-MiniLM-L6-v2, typical English prose):
        128 tokens ≈  512 chars  (lower bound of 500–800 target)
        200 tokens ≈  800 chars  (upper bound of 500–800 target)

    We extract the BertTokenizer from the locally-cached SentenceTransformer model
    and wrap it in HuggingFaceTokenizer — this avoids any HuggingFace network call.

    Returns None if HybridChunker is unavailable; _ingest_worker falls back to
    RecursiveCharacterTextSplitter in that case.
    """
    try:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        # Reuse the BertTokenizer already loaded inside the cached SentenceTransformer —
        # zero network access, model weights already on disk.
        bert_tokenizer = load_embed_model().tokenizer
        hft = HuggingFaceTokenizer(tokenizer=bert_tokenizer, max_tokens=200)
        chunker = HybridChunker(tokenizer=hft, merge_peers=True)
        logger.info("[HybridChunker] Initialised offline using cached MiniLM tokenizer")
        return chunker
    except Exception as exc:
        logger.warning(
            f"[HybridChunker] Failed to initialise ({exc}) — "
            "worker will use RecursiveCharacterTextSplitter fallback"
        )
        return None


def embed(texts: list[str]) -> list[list[float]]:
    model = load_embed_model()
    vecs = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def chroma_get(path: str) -> dict | list:
    r = requests.get(f"{CHROMA_API}{path}", timeout=5)
    r.raise_for_status()
    return r.json()


def chroma_post(path: str, body: dict) -> dict | list:
    r = requests.post(f"{CHROMA_API}{path}", json=body, timeout=15)
    r.raise_for_status()
    return r.json()


def check_chroma() -> tuple[bool, str]:
    try:
        hb = chroma_get("/heartbeat")
        return True, f"v0.5.x · ns={hb.get('nanosecond heartbeat', '?')}"
    except Exception as e:
        return False, str(e)


def check_redis() -> tuple[bool, str]:
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        r.ping()
        info = r.info("server")
        return True, f"v{info.get('redis_version','?')} · uptime {info.get('uptime_in_seconds','?')}s"
    except Exception as e:
        return False, str(e)


def get_collection_id(name: str) -> str | None:
    try:
        colls = chroma_get("/collections")
        for c in colls:
            if c["name"] == name:
                return c["id"]
        return None
    except Exception:
        return None


def list_all_collections() -> list[dict]:
    try:
        return chroma_get("/collections")  # type: ignore
    except Exception:
        return []


def fetch_all_chunks(coll_id: str) -> list[dict]:
    """Pull every stored chunk for the inspector table."""
    try:
        r = requests.post(
            f"{CHROMA_API}/collections/{coll_id}/get",
            json={"include": ["documents", "metadatas"]},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        ids   = data.get("ids", [])
        docs  = data.get("documents", [])
        metas = data.get("metadatas", [])
        return [
            {"id": i, "text": d, "metadata": m}
            for i, d, m in zip(ids, docs, metas)
        ]
    except Exception:
        return []


def grade_relevance(question: str, context: str) -> float:
    import anthropic
    client = anthropic.Anthropic(api_key=_get_api_key())
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": (
                "You are a relevance grader. Given a question and retrieved context, "
                "output ONLY a JSON object: {\"score\": <float 0-1>}\n\n"
                f"Question: {question}\n\nContext (first 800 chars):\n{context[:800]}"
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    try:
        return float(json.loads(raw).get("score", 0.0))
    except Exception:
        m = re.search(r"0\.\d+|1\.0", raw)
        return float(m.group()) if m else 0.5


def rewrite_query(question: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_get_api_key())
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": (
                "Rewrite the following question to improve document retrieval. "
                "Output ONLY the rewritten question, no explanation.\n\n"
                f"Original: {question}"
            ),
        }],
    )
    return msg.content[0].text.strip()


def generate_answer(question: str, context: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_get_api_key())
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "You are a senior systems architect. Answer the question using ONLY "
                "the provided context. Show your calculations explicitly with exact numbers. "
                "Format the response in clean Markdown.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            ),
        }],
    )
    return msg.content[0].text.strip()


def agentic_query(question: str, progress_placeholder) -> dict:
    """Self-correction loop: retrieve → grade → rewrite → answer."""
    current_q   = question
    chunks      = []
    iterations  = []
    converged   = False

    for i in range(1, MAX_CORRECTION_ITERATIONS + 1):
        with progress_placeholder.container():
            st.markdown(f"**⟳ Iteration {i}/{MAX_CORRECTION_ITERATIONS}** — embedding query…")

        q_vec  = embed([current_q])[0]
        coll_id = get_collection_id(COLLECTION_NAME)
        if not coll_id:
            return {"error": f"Collection '{COLLECTION_NAME}' not found. Run ingestion first."}

        resp = chroma_post(f"/collections/{coll_id}/query", {
            "query_embeddings": [q_vec],
            "n_results": TOP_K,
            "include": ["documents", "metadatas", "distances"],
        })
        ids       = resp.get("ids", [[]])[0]
        docs      = resp.get("documents", [[]])[0]
        metas     = resp.get("metadatas", [[]])[0]
        distances = resp.get("distances", [[]])[0]

        chunks = [
            {"id": cid, "text": doc, "metadata": meta,
             "similarity": round(1.0 - dist, 4)}
            for cid, doc, meta, dist in zip(ids, docs, metas, distances)
        ]
        context = "\n\n".join(c["text"] for c in chunks)

        with progress_placeholder.container():
            st.markdown(f"**⟳ Iteration {i}/{MAX_CORRECTION_ITERATIONS}** — grading relevance…")

        score = grade_relevance(current_q, context)
        iter_info = {
            "iteration": i,
            "query":     current_q,
            "score":     score,
            "action":    "",
        }

        if score >= RELEVANCE_THRESHOLD:
            iter_info["action"] = f"✅ Score {score:.2f} ≥ {RELEVANCE_THRESHOLD} — generating answer"
            iterations.append(iter_info)
            converged = True
            break

        if i < MAX_CORRECTION_ITERATIONS:
            with progress_placeholder.container():
                st.markdown(f"**⟳ Iteration {i}/{MAX_CORRECTION_ITERATIONS}** — rewriting query…")
            new_q = rewrite_query(current_q)
            iter_info["action"] = f"⚡ Score {score:.2f} < {RELEVANCE_THRESHOLD} — query rewritten"
            iterations.append(iter_info)
            current_q = new_q
        else:
            iter_info["action"] = f"⚠️ Max iterations reached (score={score:.2f}) — answering with best context"
            iterations.append(iter_info)

    with progress_placeholder.container():
        st.markdown("**✍️ Generating grounded answer…**")

    context = "\n\n".join(c["text"] for c in chunks)
    answer  = generate_answer(question, context)

    return {
        "answer":     answer,
        "chunks":     chunks,
        "iterations": iterations,
        "converged":  converged,
    }


def ingest_pdf_bytes(data: bytes, filename: str) -> dict:
    """Full ingestion pipeline: bytes → Docling → chunks → ChromaDB."""
    from docling.document_converter import DocumentConverter
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        converter = DocumentConverter()
        t0        = time.perf_counter()
        result    = converter.convert(str(tmp_path))
        parse_sec = time.perf_counter() - t0
        markdown  = result.document.export_to_markdown()
        pages     = len(result.document.pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64,
            separators=["\n\n", "\n", " ", ""],
        )
        raw_doc = Document(
            page_content=markdown,
            metadata={"source": filename, "filename": filename},
        )
        chunks    = splitter.split_documents([raw_doc])
        all_texts = [c.page_content for c in chunks]

        t_emb       = time.perf_counter()
        embeddings  = embed(all_texts)
        embed_sec   = time.perf_counter() - t_emb

        # Drop + recreate collection
        try:
            requests.delete(f"{CHROMA_API}/collections/{COLLECTION_NAME}", timeout=5)
        except Exception:
            pass

        coll = chroma_post("/collections", {
            "name": COLLECTION_NAME,
            "metadata": {"hnsw:space": "cosine"},
        })
        coll_id = coll["id"]

        batch = 50
        for s in range(0, len(chunks), batch):
            e = min(s + batch, len(chunks))
            chroma_post(f"/collections/{coll_id}/upsert", {
                "ids":        [f"chunk_{i:04d}" for i in range(s, e)],
                "documents":  all_texts[s:e],
                "embeddings": embeddings[s:e],
                "metadatas":  [c.metadata for c in chunks[s:e]],
            })

        return {
            "filename":   filename,
            "pages":      pages,
            "chunks":     len(chunks),
            "parse_sec":  round(parse_sec, 2),
            "embed_sec":  round(embed_sec, 2),
            "markdown":   markdown,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-isolated ingestion worker  (runs in _EXECUTOR, never on main thread)
# ─────────────────────────────────────────────────────────────────────────────
def _ingest_worker(
    pdf_bytes: bytes,
    filename: str,
    progress_q: "queue.Queue[dict]",
    cancel_evt: threading.Event,
    result_bag: dict,          # mutated in-place; main thread reads after done
) -> None:
    """
    4-step pipeline.  Between every step the worker checks cancel_evt so the
    watchdog can signal a clean stop without force-killing the thread (Python
    doesn't support that).  All UI updates go through progress_q — the worker
    never touches st.* directly.
    """
    t0 = time.perf_counter()

    def push(msg: str, level: str = "info") -> None:
        # Swallow Windows handle errors (WinError 6) that occur when the OS
        # invalidates a thread handle between Streamlit's internal poll and the
        # queue put.  The message is dropped but the worker keeps running.
        try:
            progress_q.put({
                "msg":     msg,
                "level":   level,
                "elapsed": round(time.perf_counter() - t0, 1),
            })
        except OSError:
            pass   # [WinError 6] The handle is invalid — safe to ignore
        try:
            logger.debug(f"[WORKER:{threading.current_thread().name}] {msg}")
        except OSError:
            pass   # logger may also hit a closed handle on Windows

    def cancelled() -> bool:
        if cancel_evt.is_set():
            push("🛑 Cancelled by watchdog — step aborted.", "warning")
            result_bag["status"] = "timeout"
            return True
        return False

    try:
        # ── STEP 1: Parse ────────────────────────────────────────────────────
        # Docling is bypassed entirely — its layout-detection model loads ~500 MB
        # of ML weights even with OCR disabled, which causes 'bad allocation'
        # crashes on memory-constrained machines.
        #
        # Parsing chain (no ML weights loaded at any level):
        #   1. PyMuPDF (fitz)  — fastest, preserves layout, table text included
        #   2. PyPDF2           — pure-Python fallback if fitz not installed
        #   3. pypdfium2        — second pure fallback via pdfium bindings
        #
        # The extracted plain text is passed straight to RecursiveCharacterTextSplitter
        # in Step 2 — same chunk_size=512 / overlap=64 as before.
        push("🔍 Step 1 / 4 — Parsing PDF  [lightweight engine: PyMuPDF → PyPDF2 → pypdfium2]…")
        from langchain_core.documents import Document as LCDocument

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        markdown = ""
        pages    = 0
        engine_used = ""

        try:
            t_parse = time.perf_counter()

            # ── Primary: PyMuPDF (fitz) ──────────────────────────────────────
            try:
                import fitz  # PyMuPDF
                pdf_doc = fitz.open(str(tmp_path))
                pages   = pdf_doc.page_count
                parts: list[str] = []
                for page in pdf_doc:
                    text = page.get_text("text")   # native text layer, no ML
                    if text.strip():
                        parts.append(text.strip())
                pdf_doc.close()
                markdown    = "\n\n".join(parts)
                engine_used = "PyMuPDF (fitz)"

            except ImportError:
                push("⚠️  PyMuPDF not found — trying PyPDF2…", "warning")

                # ── Fallback 1: PyPDF2 ───────────────────────────────────────
                try:
                    import PyPDF2
                    with open(tmp_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        pages  = len(reader.pages)
                        parts  = []
                        for pg in reader.pages:
                            text = pg.extract_text() or ""
                            if text.strip():
                                parts.append(text.strip())
                    markdown    = "\n\n".join(parts)
                    engine_used = "PyPDF2"

                except ImportError:
                    push("⚠️  PyPDF2 not found — trying pypdfium2…", "warning")

                    # ── Fallback 2: pypdfium2 ────────────────────────────────
                    import pypdfium2 as pdfium
                    pdf_doc = pdfium.PdfDocument(str(tmp_path))
                    pages   = len(pdf_doc)
                    parts   = []
                    for i in range(pages):
                        page      = pdf_doc[i]
                        text_page = page.get_textpage()
                        text      = text_page.get_text_range()
                        if text.strip():
                            parts.append(text.strip())
                    pdf_doc.close()
                    markdown    = "\n\n".join(parts)
                    engine_used = "pypdfium2"

            parse_sec = round(time.perf_counter() - t_parse, 2)

        finally:
            tmp_path.unlink(missing_ok=True)

        if not markdown.strip():
            raise ValueError(
                f"'{filename}' yielded no extractable text. "
                "The PDF may be image-only (needs OCR) or corrupt."
            )

        push(
            f"✅ Parse done — {pages} page(s), {len(markdown):,} chars, {parse_sec}s  "
            f"[engine={engine_used}]",
            "success",
        )
        if cancelled():
            return

        # ── STEP 2: Chunk ────────────────────────────────────────────────────
        # HybridChunker requires a Docling document object — since Step 1 now
        # uses a lightweight text extractor (PyMuPDF / PyPDF2 / pypdfium2),
        # we go directly to RecursiveCharacterTextSplitter.
        # chunk_size=512 / overlap=64 targets ~93 chunks for a 13-page doc.
        push(
            "✂️  Step 2 / 4 — Chunking with RecursiveCharacterTextSplitter  "
            "[chunk_size=512, overlap=64]…"
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64,
            separators=["\n\n", "\n", " ", ""],
        )
        raw_doc = LCDocument(
            page_content=markdown,
            metadata={"source": filename, "filename": filename},
        )
        chunks: list[LCDocument] = splitter.split_documents([raw_doc])
        for idx, c in enumerate(chunks):
            c.metadata.update({"chunk_idx": idx, "chars": len(c.page_content)})

        # ── Guard: never proceed to embedding with 0 chunks ──────────────────
        if not chunks:
            raise ValueError(
                f"Chunking produced 0 chunks for '{filename}'. "
                "The PDF may be blank, image-only (needs OCR), or corrupt."
            )

        all_texts = [c.page_content for c in chunks]
        avg_chars = round(sum(len(t) for t in all_texts) / max(len(all_texts), 1))
        in_range  = sum(500 <= len(t) <= 800 for t in all_texts)
        push(
            f"✅ {len(chunks)} chunks created  "
            f"(avg {avg_chars} chars · {in_range}/{len(chunks)} in 500–800 range)",
            "success",
        )
        if cancelled():
            return

        # ── STEP 3: Embed ────────────────────────────────────────────────────
        push(f"🧮 Step 3 / 4 — Embedding {len(chunks)} chunks with {EMBED_MODEL}…")
        t_emb      = time.perf_counter()
        embeddings = embed(all_texts)          # uses @st.cache_resource model
        embed_sec  = round(time.perf_counter() - t_emb, 2)
        push(
            f"✅ Embeddings ready — dim={len(embeddings[0])}, {embed_sec}s",
            "success",
        )
        if cancelled():
            return

        # ── STEP 4: Upsert ───────────────────────────────────────────────────
        push("💾 Step 4 / 4 — Upserting to ChromaDB…")
        try:
            requests.delete(f"{CHROMA_API}/collections/{COLLECTION_NAME}", timeout=5)
        except Exception:
            pass

        coll    = chroma_post("/collections", {
            "name":     COLLECTION_NAME,
            "metadata": {"hnsw:space": "cosine"},
        })
        coll_id = coll["id"]

        batch_sz = 50
        for s in range(0, len(chunks), batch_sz):
            if cancelled():
                return
            e = min(s + batch_sz, len(chunks))
            chroma_post(f"/collections/{coll_id}/upsert", {
                "ids":        [f"chunk_{i:04d}" for i in range(s, e)],
                "documents":  all_texts[s:e],
                "embeddings": embeddings[s:e],
                "metadatas":  [c.metadata for c in chunks[s:e]],
            })
            push(
                f"   ↳ batch [{s}:{e}] upserted ({e}/{len(chunks)})",
                "info",
            )

        push(
            f"✅ ChromaDB upsert complete — collection `{COLLECTION_NAME}`",
            "success",
        )

        # Brief pause so Windows finishes flushing file handles opened by the
        # ChromaDB REST server and the queue drain on the main thread has time
        # to read the final messages before the future is marked done.
        time.sleep(0.15)

        # ── Write result bag (main thread reads this after future.done()) ────
        result_bag.update({
            "status":    "done",
            "filename":  filename,
            "pages":     pages,
            "chunks":    len(chunks),
            "parse_sec": parse_sec,
            "embed_sec": embed_sec,
            "markdown":  markdown,
        })
        push("🎉 Ingestion finished successfully!", "success")

    except Exception as exc:
        # Log first — if logger itself raises a handle error, still record status.
        try:
            logger.exception(f"[WORKER] Unhandled exception: {exc}")
        except OSError:
            pass
        push(f"❌ Unhandled error: {exc}", "error")
        result_bag["status"] = "error"
        result_bag["error"]  = str(exc)


# ── Job helpers (all called from main thread only) ───────────────────────────
def _init_job() -> None:
    """Ensure session_state has a clean job dict."""
    if "ingest_job" not in st.session_state:
        st.session_state["ingest_job"] = dict(_JOB_DEFAULTS)


def _start_ingest_job(pdf_bytes: bytes, filename: str) -> None:
    """Spin up the worker thread and record the job in session_state."""
    # Resource management — ThreadPoolExecutor reuses threads to avoid overhead
    q          = queue.Queue()
    cancel_evt = threading.Event()
    result_bag: dict = {}

    future: Future = _EXECUTOR.submit(
        _ingest_worker, pdf_bytes, filename, q, cancel_evt, result_bag
    )
    st.session_state["ingest_job"] = {
        "status":     "running",
        "steps":      [],
        "result":     None,
        "error":      None,
        "start_time": time.time(),
        "_queue":     q,
        "_cancel":    cancel_evt,
        "_future":    future,
        "_result_bag": result_bag,
    }


def _poll_ingest_job() -> None:
    """
    Called once per Streamlit rerun while status == 'running'.
    Drains the progress queue, checks the watchdog, and transitions
    the job to a terminal state when the future completes.
    """
    job = st.session_state["ingest_job"]
    q: queue.Queue     = job["_queue"]
    future: Future     = job["_future"]
    cancel_evt         = job["_cancel"]
    result_bag: dict   = job["_result_bag"]
    elapsed            = time.time() - job["start_time"]

    # Drain all pending messages from the worker
    while not q.empty():
        try:
            job["steps"].append(q.get_nowait())
        except queue.Empty:
            break

    # ── Watchdog ─────────────────────────────────────────────────────────────
    if elapsed >= INGEST_TIMEOUT_SECS and not future.done():
        cancel_evt.set()                   # signal worker to stop at next checkpoint
        job["status"] = "timeout"
        job["error"]  = (
            f"Watchdog: job exceeded {INGEST_TIMEOUT_SECS}s limit and was cancelled. "
            "The background thread will stop at its next checkpoint."
        )
        logger.warning(f"[WATCHDOG] Ingestion timed out after {elapsed:.0f}s")
        return

    # ── Terminal state ───────────────────────────────────────────────────────
    if future.done():
        # One final queue drain after the thread exits
        while not q.empty():
            try:
                job["steps"].append(q.get_nowait())
            except queue.Empty:
                break

        try:
            future.result()             # re-raises any exception from the worker
        except Exception as exc:
            job["status"] = "error"
            job["error"]  = str(exc)
            return

        if result_bag.get("status") == "done":
            job["status"] = "done"
            job["result"] = result_bag
        elif result_bag.get("status") == "timeout":
            job["status"] = "timeout"
            job["error"]  = "Cancelled by watchdog inside worker."
        else:
            job["status"] = "error"
            job["error"]  = result_bag.get("error", "Unknown worker error")


def _reset_ingest_job() -> None:
    st.session_state["ingest_job"] = dict(_JOB_DEFAULTS)


# ── Initialise job state once per session ────────────────────────────────────
_init_job()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛠️ System Status")

    chroma_ok, chroma_msg = check_chroma()
    redis_ok,  redis_msg  = check_redis()

    pill = lambda ok: (
        '<span class="status-ok">● LIVE</span>'
        if ok else
        '<span class="status-fail">● DOWN</span>'
    )
    st.markdown(
        f"**ChromaDB** &nbsp; {pill(chroma_ok)}<br>"
        f"<small style='color:#78909c'>{chroma_msg}</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Redis** &nbsp; {pill(redis_ok)}<br>"
        f"<small style='color:#78909c'>{redis_msg}</small>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("## ⚙️ RAG Config")
    top_k_cfg   = st.slider("Top-K retrieval",   1, 10, TOP_K)
    thresh_cfg  = st.slider("Relevance threshold", 0.0, 1.0, RELEVANCE_THRESHOLD, 0.05)
    iters_cfg   = st.slider("Max self-correction iterations", 1, 5, MAX_CORRECTION_ITERATIONS)
    TOP_K                     = top_k_cfg
    RELEVANCE_THRESHOLD       = thresh_cfg
    MAX_CORRECTION_ITERATIONS = iters_cfg

    st.divider()
    st.markdown("## 📚 Collections")
    colls = list_all_collections()
    if colls:
        for c in colls:
            dim = c.get("dimension") or "—"
            st.markdown(
                f"**{c['name']}**  \n"
                f"<small style='color:#78909c'>dim={dim}  id={c['id'][:8]}…</small>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No collections yet.")

    st.divider()
    st.caption("Tier-3 Agentic RAG System  •  v0.1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>🧠 Tier-3 Agentic RAG System</h1>
    <p>Hybrid Search · Self-Correction Agent · Docling PDF Parsing · RAGAS Evaluation</p>
</div>
""", unsafe_allow_html=True)

tab_query, tab_ingest, tab_inspect = st.tabs([
    "🔍 Agentic Query",
    "📄 PDF Ingestion",
    "🗄️ Collection Inspector",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Agentic Query
# ─────────────────────────────────────────────────────────────────────────────
with tab_query:
    st.markdown("### Ask the RAG Agent")
    st.caption(
        "The self-correction loop retrieves top-K chunks, grades relevance with Claude, "
        "rewrites the query if needed, then generates a grounded answer."
    )

    default_q = (
        "Based on the performance table, what would be the Latency improvement "
        "and Throughput increase if we migrate our services from AF-South-1 to US-West-2?"
    )
    question = st.text_area(
        "Your question",
        value=default_q,
        height=90,
        placeholder="Type your question here…",
    )

    col_run, col_clear = st.columns([1, 5])
    run_clicked = col_run.button("▶ Run Query", type="primary", use_container_width=True)
    if col_clear.button("✕ Clear", use_container_width=False):
        st.session_state.pop("query_result", None)
        st.rerun()

    if run_clicked and question.strip():
        try:
            _get_api_key()   # validate key before spinning up the query loop
            _key_ok = True
        except ValueError as _e:
            st.error(f"❌ {_e}")
            _key_ok = False

        if not _key_ok:
            pass   # error already displayed above
        elif not chroma_ok:
            st.error("❌ ChromaDB is not reachable. Start it with `docker compose up chromadb -d`.")
        else:
            progress = st.empty()
            t_start  = time.perf_counter()
            with st.spinner("Running agentic retrieval…"):
                result = agentic_query(question.strip(), progress)
            progress.empty()
            result["total_sec"] = round(time.perf_counter() - t_start, 2)
            st.session_state["query_result"] = result

    if "query_result" in st.session_state:
        res = st.session_state["query_result"]

        if "error" in res:
            st.error(res["error"])
        else:
            # ── Metrics row ──
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Iterations", res["iterations"].__len__())
            col2.metric("Converged", "✅ Yes" if res["converged"] else "⚠️ No")
            col3.metric("Chunks used", len(res["chunks"]))
            col4.metric("Total time", f"{res['total_sec']}s")

            st.divider()

            # ── Self-correction log ──
            with st.expander("🔄 Self-Correction Iteration Log", expanded=False):
                for it in res["iterations"]:
                    st.markdown(
                        f'<div class="iter-log">'
                        f'<span class="label">Iter {it["iteration"]}</span> &nbsp; '
                        f'Score: <b>{it["score"]:.4f}</b> &nbsp;|&nbsp; {it["action"]}<br>'
                        f'<span style="color:#546e7a">Query: {it["query"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Source chunks ──
            with st.expander(f"📂 Source Chunks ({len(res['chunks'])} retrieved)", expanded=True):
                for i, chunk in enumerate(res["chunks"]):
                    sim_pct = int(chunk["similarity"] * 100)
                    sim_color = (
                        "#4caf50" if chunk["similarity"] >= 0.6 else
                        "#ff9800" if chunk["similarity"] >= 0.35 else
                        "#ef5350"
                    )
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div class="chunk-header">'
                        f'<span>📎 <b>{chunk["id"]}</b> &nbsp;·&nbsp; '
                        f'{chunk["metadata"].get("filename","?")}</span>'
                        f'<span style="color:{sim_color}">sim {chunk["similarity"]:.4f}</span>'
                        f'</div>'
                        f'<div class="sim-bar-wrap">'
                        f'<div class="sim-bar" style="width:{sim_pct}%; background:{sim_color}"></div>'
                        f'</div>'
                        f'<div style="margin-top:8px">{chunk["text"][:500]}'
                        f'{"…" if len(chunk["text"]) > 500 else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Final answer ──
            st.markdown("#### 🤖 Agent Answer")
            st.markdown(
                f'<div class="answer-box">{res["answer"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Download button
            export = {
                "question":   question,
                "answer":     res["answer"],
                "iterations": res["iterations"],
                "chunks":     [
                    {"id": c["id"], "similarity": c["similarity"], "text": c["text"]}
                    for c in res["chunks"]
                ],
            }
            st.download_button(
                "⬇ Download result as JSON",
                data=json.dumps(export, indent=2),
                file_name="rag_result.json",
                mime="application/json",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PDF Ingestion  (thread-isolated, watchdog-protected)
# ─────────────────────────────────────────────────────────────────────────────
with tab_ingest:
    job = st.session_state["ingest_job"]

    # ── Poll: if a job is running, drain the queue then decide whether to
    #    schedule the next rerun.  This is the only place st.rerun() is called
    #    so we never loop outside the normal Streamlit execution model.
    if job["status"] == "running":
        _poll_ingest_job()
        job = st.session_state["ingest_job"]   # re-read after poll

    is_running = (job["status"] == "running")

    st.markdown("### Upload & Ingest a PDF")
    st.caption(
        "The ingestion pipeline runs in an **isolated background thread** — "
        "the UI stays responsive and a 5-minute watchdog prevents runaway jobs."
    )

    # ── File uploader (hidden while a job is running to avoid state conflicts)
    uploaded = None
    if not is_running:
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded:
        st.info(
            f"📄 **{uploaded.name}** — {uploaded.size / 1024:.1f} KB  "
            f"| ready to ingest into `{COLLECTION_NAME}`"
        )

    # ── Action buttons row ────────────────────────────────────────────────────
    col_btn, col_cancel, col_reset, _ = st.columns([2, 2, 2, 4])

    ingest_clicked = col_btn.button(
        "🚀 Ingest Document",
        type="primary",
        disabled=(is_running or uploaded is None),
        use_container_width=True,
        help="Disabled while a job is already running" if is_running else "",
    )
    cancel_clicked = col_cancel.button(
        "⏹ Cancel",
        disabled=not is_running,
        use_container_width=True,
    )
    reset_clicked = col_reset.button(
        "↺ Reset",
        disabled=is_running,
        use_container_width=True,
    )

    # ── Button actions ────────────────────────────────────────────────────────
    if ingest_clicked and uploaded and not is_running:
        if not chroma_ok:
            st.error("❌ ChromaDB is unreachable. Start it with `docker compose up chromadb -d`.")
        else:
            _start_ingest_job(uploaded.read(), uploaded.name)
            st.rerun()

    if cancel_clicked and is_running:
        job["_cancel"].set()
        job["status"] = "timeout"
        job["error"]  = "Manually cancelled by user."
        st.rerun()

    if reset_clicked:
        _reset_ingest_job()
        st.rerun()

    # ── Live status panel ─────────────────────────────────────────────────────
    if job["status"] != "idle":
        st.markdown("---")

        # Elapsed + watchdog bar
        elapsed      = (time.time() - job["start_time"]) if job["start_time"] else 0
        pct          = min(int(elapsed / INGEST_TIMEOUT_SECS * 100), 100)
        bar_color    = "#4caf50" if pct < 70 else "#ff9800" if pct < 90 else "#ef5350"
        status_label = {
            "running": "⟳ Running…",
            "done":    "✅ Complete",
            "error":   "❌ Failed",
            "timeout": "⏱ Timed Out",
        }.get(job["status"], job["status"])

        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**Status:** {status_label}")
        c2.markdown(
            f"<div style='text-align:right;color:#90a4ae;font-size:0.82rem'>"
            f"{'⏱ ' if is_running else ''}{elapsed:.1f}s / {INGEST_TIMEOUT_SECS}s</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="watchdog-bar-wrap">'
            f'<div class="watchdog-bar" style="width:{pct}%;background:{bar_color}"></div>'
            f'</div>'
            f'<div style="font-size:0.72rem;color:#546e7a;margin-bottom:6px">'
            f'Watchdog: {pct}% of {INGEST_TIMEOUT_SECS}s limit used</div>',
            unsafe_allow_html=True,
        )

        # Step-tracker widget
        if job["steps"]:
            rows_html = ""
            for step in job["steps"]:
                level  = step.get("level", "info")
                icon   = {"success": "✓", "warning": "⚠", "error": "✕"}.get(level, "⟳")
                color  = {
                    "success": "#4caf50", "warning": "#ff9800",
                    "error":   "#ef5350", "info":    "#64b5f6",
                }.get(level, "#64b5f6")
                rows_html += (
                    f'<div class="step-row">'
                    f'<span class="step-icon" style="color:{color}">{icon}</span>'
                    f'<span class="step-msg">{step["msg"]}</span>'
                    f'<span class="step-ts">+{step["elapsed"]}s</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="step-tracker">{rows_html}</div>',
                unsafe_allow_html=True,
            )

        # ── Terminal state renderers ──────────────────────────────────────────
        if job["status"] == "done" and job["result"]:
            r = job["result"]
            st.success(f"✅ **{r['filename']}** ingested successfully!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pages",       r["pages"])
            col2.metric("Chunks",      r["chunks"])
            col3.metric("Parse time",  f"{r['parse_sec']}s")
            col4.metric("Embed time",  f"{r['embed_sec']}s")
            with st.expander("📄 Docling Markdown Output (first 3 000 chars)", expanded=False):
                st.code(r["markdown"][:3000], language="markdown")

        elif job["status"] == "error":
            st.error(f"❌ Ingestion failed: {job['error']}")

        elif job["status"] == "timeout":
            st.warning(
                f"⏱ **Watchdog fired** — job exceeded {INGEST_TIMEOUT_SECS}s. "
                "The background thread will stop at its next checkpoint. "
                "Click **↺ Reset** to start a new ingestion."
            )

        # ── Schedule next poll while still running ────────────────────────────
        #    Important: this must come LAST in the tab so all widgets are
        #    rendered before the rerun is triggered.
        if is_running:
            time.sleep(POLL_INTERVAL_SECS)
            st.rerun()

    # ── Quick-generate helper (uses subprocess so it doesn't block UI) ────────
    st.divider()
    st.markdown("#### 🧪 Quick-Generate `system_spec.pdf`")
    st.caption("Regenerate the professional benchmark PDF used in earlier tests.")

    if st.button("🔧 Generate & Ingest system_spec.pdf", disabled=is_running):
        import subprocess, sys as _sys
        with st.spinner("Generating PDF and running ingestion script…"):
            proc = subprocess.run(
                [_sys.executable, "-X", "utf8", "debug_ingest.py"],
                capture_output=True, text=True,
                cwd=str(Path(__file__).parent),
            )
        if proc.returncode == 0:
            st.success("✅ system_spec.pdf generated and ingested!")
            with st.expander("Script output"):
                st.code(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)
        else:
            st.error("Ingestion script failed.")
            st.code(proc.stderr[-2000:])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Collection Inspector
# ─────────────────────────────────────────────────────────────────────────────
with tab_inspect:
    st.markdown("### Collection Inspector")
    st.caption("Browse every indexed chunk stored in ChromaDB.")

    colls = list_all_collections()
    if not colls:
        st.warning("No collections found. Ingest a document first.")
    else:
        coll_names = [c["name"] for c in colls]
        selected   = st.selectbox("Select collection", coll_names)
        coll_id    = next(c["id"] for c in colls if c["name"] == selected)

        if st.button("🔄 Load Chunks"):
            with st.spinner("Fetching chunks from ChromaDB…"):
                chunks = fetch_all_chunks(coll_id)
            st.session_state["inspector_chunks"] = chunks
            st.session_state["inspector_coll"]   = selected

        if "inspector_chunks" in st.session_state and st.session_state.get("inspector_coll") == selected:
            chunks = st.session_state["inspector_chunks"]
            st.markdown(f"**{len(chunks)} chunks** in `{selected}`")

            search = st.text_input("🔎 Filter chunks by keyword", placeholder="e.g. latency")
            if search:
                chunks = [c for c in chunks if search.lower() in c["text"].lower()]
                st.caption(f"{len(chunks)} chunks matching '{search}'")

            rows = [
                {
                    "ID":       c["id"],
                    "File":     c["metadata"].get("filename", "—"),
                    "Preview":  c["text"][:120].replace("\n", " ") + "…",
                    "Length":   len(c["text"]),
                }
                for c in chunks
            ]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=350)

            if chunks:
                st.markdown("#### 🔬 Chunk Detail")
                selected_id = st.selectbox("Select chunk ID", [c["id"] for c in chunks])
                chunk_detail = next(c for c in chunks if c["id"] == selected_id)
                st.json(chunk_detail["metadata"])
                st.code(chunk_detail["text"], language="markdown")
