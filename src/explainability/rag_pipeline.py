"""
RAG explanation pipeline for fabric defects.

Architecture:
  Knowledge base  → chunked .txt documents in data/knowledge_base/
  Embeddings      → sentence-transformers/all-MiniLM-L6-v2 (local, no API key)
  Vector store    → Qdrant in-memory (no Docker required)
  LLM             → Claude claude-haiku-4-5-20251001 via Anthropic API
                    Falls back to a structured template if ANTHROPIC_API_KEY is absent.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Query
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
COLLECTION_NAME    = "fabric_defects"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE         = 400   # characters
CHUNK_OVERLAP      = 80
TOP_K              = 3

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a fabric quality control expert at a Sri Lankan apparel factory.
    A computer-vision model has detected a defect in fabric on the production line.
    Use the retrieved knowledge base excerpts to give a concise, actionable explanation.
    Structure your response in exactly three numbered points:
      1. What this defect indicates about fabric quality
      2. Most likely root cause (machine, process, or material)
      3. Recommended immediate action for the line operator
    Be specific and practical. Do not repeat the defect name in every sentence.
    Use plain language — the reader is a factory floor supervisor, not an academic.
""")


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries where possible."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # keep overlap from previous chunk
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + "\n\n" + para).strip() if tail else para
    if current:
        chunks.append(current)
    return chunks


def _load_documents() -> list[dict]:
    """Load and chunk all .txt files in the knowledge base directory."""
    docs = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_text(text)):
            docs.append({
                "text":   chunk,
                "source": path.stem,
                "chunk":  i,
            })
    return docs


class FabricDefectExplainer:
    def __init__(self):
        print("Loading embedding model …")
        self._embedder = SentenceTransformer(EMBED_MODEL)
        dim = self._embedder.get_embedding_dimension()

        self._client = QdrantClient(":memory:")
        self._client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._build_index()

        # Claude client — optional
        self._anthropic = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self._anthropic = anthropic.Anthropic(api_key=api_key)
                print("Claude API connected.")
            except ImportError:
                pass
        if not self._anthropic:
            print("No ANTHROPIC_API_KEY — will use structured template fallback.")

    def _build_index(self):
        docs = _load_documents()
        vectors = self._embedder.encode([d["text"] for d in docs], show_progress_bar=False)
        points  = [
            PointStruct(id=i, vector=vec.tolist(), payload=doc)
            for i, (doc, vec) in enumerate(zip(docs, vectors))
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Indexed {len(points)} chunks from {KNOWLEDGE_BASE_DIR.name}/")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Return the top-k most relevant knowledge base chunks for a query."""
        vec  = self._embedder.encode(query).tolist()
        hits = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=k,
            with_payload=True,
        ).points
        return [
            {"text": h.payload["text"], "source": h.payload["source"], "score": h.score}
            for h in hits
        ]

    def explain(
        self,
        defect_class: str,
        confidence: float,
        extra_context: Optional[str] = None,
    ) -> dict:
        """
        Generate a root-cause explanation for a detected defect.

        Returns a dict with keys:
          defect_class, confidence, retrieved_chunks, explanation, source
        """
        query    = f"{defect_class} fabric defect root cause corrective action"
        if extra_context:
            query += f" {extra_context}"
        chunks   = self.retrieve(query)
        context  = "\n\n---\n\n".join(c["text"] for c in chunks)

        if self._anthropic:
            explanation = self._llm_explain(defect_class, confidence, context)
            source = "Claude claude-haiku-4-5-20251001 + RAG"
        else:
            explanation = self._template_explain(defect_class, confidence, context)
            source = "Template + RAG"

        return {
            "defect_class":      defect_class,
            "confidence":        confidence,
            "retrieved_chunks":  chunks,
            "explanation":       explanation,
            "source":            source,
        }

    def _llm_explain(self, defect_class: str, confidence: float, context: str) -> str:
        user_msg = (
            f"Defect detected: **{defect_class.replace('_', ' ').upper()}**\n"
            f"Model confidence: {confidence:.1%}\n\n"
            f"Retrieved knowledge base excerpts:\n\n{context}"
        )
        response = self._anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()

    def _template_explain(self, defect_class: str, confidence: float, context: str) -> str:
        """Structured fallback when no LLM is available.
        Re-retrieves chunks filtered to the defect's own source document first."""
        # Pull chunks that match this defect's source doc
        own_chunks = self.retrieve(
            f"{defect_class} root cause corrective action machine process", k=5
        )
        own_text = " ".join(
            c["text"] for c in own_chunks if c["source"] == defect_class
        )
        # Fall back to full context if no own-source chunks found
        search_text = own_text if own_text else context

        lines = [l.strip() for l in search_text.split("\n")
                 if l.strip() and not l.startswith("---") and not l.startswith("DEFECT:")]

        cause_lines  = [l for l in lines if any(w in l.lower() for w in
                        ["cause", "result", "due to", "misalign", "leak", "worn",
                         "broken", "irregular", "inconsistent", "uneven", "damaged"])]
        action_lines = [l for l in lines if any(w in l.lower() for w in
                        ["stop", "inspect", "check", "replace", "adjust",
                         "quarantine", "halt", "measure", "calibrate"])]

        cause  = cause_lines[0][:220]  if cause_lines  else lines[0][:220] if lines else "Inspect machine and process settings."
        action = action_lines[0][:220] if action_lines else "Stop production line and investigate."

        desc_map = {
            "stain":            "surface contamination — likely machine lubrication failure or dye process deviation",
            "tear":             "physical fabric damage — likely knitting needle wear, yarn break, or foreign object",
            "weave_distortion": "structural deformation — likely loom tension imbalance or stenter misalignment",
            "normal":           "no defect detected — fabric passes automated visual inspection",
        }
        desc = desc_map.get(defect_class, "a fabric quality issue requiring investigation")

        return (
            f"1. This detection ({confidence:.0%} confidence) indicates {desc}.\n\n"
            f"2. Most likely root cause: {cause}\n\n"
            f"3. Recommended action: {action}"
        )
