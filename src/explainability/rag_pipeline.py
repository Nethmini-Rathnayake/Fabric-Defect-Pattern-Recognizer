"""RAG pipeline for root-cause explanation of fabric defects using LangChain + Qdrant + Llama-3."""

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
import os


DEFECT_KNOWLEDGE_BASE = {
    "oil_stain": "Oil stains appear as dark patches caused by machine lubrication leakage near feed rollers or bearings.",
    "dye_stain": "Dye stains or color bleeds result from poor dyeing process control, inconsistent dye bath temperatures, or contaminated dye lots.",
    "hole_snag": "Holes and snags are physical fabric damage caused by yarn breaks, damaged knitting needles, or foreign objects on the conveyor.",
    "drop_stitch": "Drop stitches or runs are missing loops in the knit structure caused by knitting machine malfunction or broken yarn feeders.",
    "weave_distortion": "Weave distortions (skewing, bowing) indicate loom tension misalignment, improper beam winding, or fabric stretching during processing.",
    "slub_nep": "Slubs and neps are thick yarn irregularities caused by spinning process flaws, fiber contamination, or poor carding.",
    "shade_variation": "Shade variation across fabric width results from inconsistent dye bath conditions, temperature gradients, or uneven fabric tension during dyeing.",
    "shrinkage": "Shrinkage indicates dimensional instability from improper heat or moisture treatment during finishing.",
}

PROMPT_TEMPLATE = """You are a fabric quality control expert for a Sri Lankan apparel factory.
A defect has been detected in the fabric on the conveyor belt.

Defect type: {defect_class}
Confidence: {confidence:.1%}

Context from knowledge base:
{context}

Provide a concise explanation in 2-3 sentences:
1. What the defect indicates
2. The most likely root cause
3. Recommended immediate action

Answer:"""


class FabricDefectExplainer:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "fabric_defects",
        llm_model: str = "llama3",
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.llm = Ollama(model=llm_model)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self._populate_knowledge_base()

    def _populate_knowledge_base(self):
        texts = list(DEFECT_KNOWLEDGE_BASE.values())
        metadatas = [{"defect_class": k} for k in DEFECT_KNOWLEDGE_BASE.keys()]
        Qdrant.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            host=self.client._client.host,
            port=self.client._client.port,
            collection_name=self.collection_name,
        )

    def explain(self, defect_class: str, confidence: float) -> str:
        vectorstore = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        prompt = PromptTemplate(
            input_variables=["context", "defect_class", "confidence"],
            template=PROMPT_TEMPLATE,
        )
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
        )
        return chain.run({"query": defect_class, "defect_class": defect_class, "confidence": confidence})
