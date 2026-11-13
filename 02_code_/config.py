"""Central configuration for the GraphRAG + BGE-M3 medical QA pipeline.

All paths are resolved relative to this file, so the pipeline can be run
from any working directory (e.g. `python 02_code/run_experiment.py` or
`python run_experiment.py` from inside `02_code/`).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "01_data" / "medquad.csv"
RESULT_DIR = PROJECT_DIR / "03_result"
ARTIFACT_DIR = BASE_DIR / "artifacts"

RESULT_CSV = RESULT_DIR / "result-rag.csv"
SUMMARY_CSV = RESULT_DIR / "result-rag-summary.csv"

# --- Embedding model -------------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_BATCH_SIZE = 32
EMBED_DEVICE = "cuda"  # falls back to "cpu" automatically if CUDA unavailable

# --- Corpus construction -----------------------------------------------------
MAX_DOCS = -1            # -1 = full cleaned dataset (~16.3k docs, the paper's reported configuration); set a positive int to cap for a faster demo run
RANDOM_SEED = 42
DOC_CHUNK_CHARS = 3000   # max characters of "focus_area + answer" kept per document before embedding (the retrieval "chunk size")

# --- Graph construction (GraphRAG-lite entity graph over `focus_area`) -----
GRAPH_TFIDF_MAX_FEATURES = 4000
GRAPH_TOPK_NEIGHBORS = 5      # neighbors kept per entity node
GRAPH_SIM_THRESHOLD = 0.15    # minimum cosine similarity to draw an edge

# --- Retrieval ---------------------------------------------------------------
DENSE_TOP_K = 20        # candidates pulled by pure dense (BGE-M3) search
GRAPH_HOP_ENTITIES = 2  # number of linked entities used to seed graph expansion
FINAL_TOP_K = 5         # final fused context size handed to the answerer
RRF_K = 60               # Reciprocal Rank Fusion constant (Cormack et al., SIGIR 2009)
FAISS_NUM_THREADS = 1    # single-query search; more threads contend with query encoding (see retriever.py)

# --- Evaluation ---------------------------------------------------------------
NUM_EVAL_QUERIES = 100
ANSWER_CHAR_LIMIT = 600   # truncate long text stored in the result CSV for readability

# --- Optional LLM-based answer synthesis (off by default) --------------------
USE_LLM_GENERATION = False   # overridden by --use-llm CLI flag
OPENAI_MODEL = "gpt-4o-mini"
