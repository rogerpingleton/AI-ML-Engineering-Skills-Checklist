## Semantic chunking in RAG

Every strategy so far — fixed-size, sentence, structure, recursive — uses _structural signals_ to decide where to cut: character counts, punctuation, heading tags. Semantic chunking does something fundamentally different: it uses **meaning**to find boundaries.

The core idea is that a topic shift in text produces a measurable shift in embedding space. You embed consecutive sentences, compute similarity between adjacent pairs, and cut where similarity drops sharply. The result is chunks that each contain one coherent topic — not because you told the splitter where topics begin and end, but because the embeddings revealed it.

---The coral bars are where similarity drops — topic shifts. Drag the threshold up to make the splitter more aggressive (more chunks), down to be more conservative (fewer, larger chunks). The buffer parameter blurs chunk boundaries so context bleeds across cuts.

---

## How it works, step by step---

## Python implementations

### Option 1 — Pure Python (no LangChain, shows the mechanics)

Build it from scratch so you understand exactly what's happening inside every library.

```python
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_chunk(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.5,      # cut when similarity drops below this
    buffer_size: int = 1,        # sentences of context added around each cut
) -> list[dict]:
    """
    Split text into semantically coherent chunks.

    Args:
        text:        Input document.
        model_name:  Sentence-transformers model for embedding.
        threshold:   Cosine similarity below which a cut is made.
        buffer_size: Sentences of overlap added around each boundary.

    Returns:
        List of dicts with 'text', 'sentences', and 'similarities'.
    """
    # Step 1 — sentence tokenisation
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [{"text": text, "sentences": sentences, "similarities": []}]

    # Step 2 — embed every sentence
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Step 3 — compute cosine similarity between consecutive pairs
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Step 4 — find cut points: indices where similarity drops below threshold
    cut_points = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    # Step 5 — group sentences into chunks, with buffer on each side of cuts
    boundaries = [0] + cut_points + [len(sentences)]
    chunks = []

    for idx in range(len(boundaries) - 1):
        start = max(0, boundaries[idx] - buffer_size)
        end   = min(len(sentences), boundaries[idx + 1] + buffer_size)
        group = sentences[start:end]
        chunks.append({
            "text":         " ".join(group),
            "sentences":    group,
            "similarities": similarities[start : end - 1],
        })

    return chunks


# --- Example ---
document = """
RAG stands for Retrieval-Augmented Generation. It combines large language
models with external knowledge retrieval. The model retrieves relevant
passages at inference time rather than relying on learned weights.

Dense vector search is the most common retrieval mechanism. Chunks are
embedded into high-dimensional vectors and stored in a vector database.
At query time the nearest chunks are retrieved via ANN search.

Chunking strategy is one of the most impactful decisions in a RAG pipeline.
Poor chunking leads to irrelevant or incomplete chunks being retrieved.
The quality of retrieval determines the ceiling for answer quality.
"""

chunks = semantic_chunk(document, threshold=0.5, buffer_size=1)

for i, chunk in enumerate(chunks):
    sims = [f"{s:.2f}" for s in chunk["similarities"]]
    print(f"--- Chunk {i+1} ({len(chunk['sentences'])} sentences) ---")
    print(f"Similarities: {sims}")
    print(chunk["text"][:200])
    print()
```

---

### Option 2 — LangChain `SemanticChunker`

LangChain wraps this pattern with three built-in threshold modes and cleaner pipeline integration.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()   # or HuggingFaceEmbeddings(...)

# Three threshold strategies — pick one:

# (a) percentile: cut at the Nth percentile of similarity drops
#     Good default — adapts to the distribution of your document
splitter_percentile = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,   # cut the bottom 15% of similarities
)

# (b) standard_deviation: cut where drop > mean - N*std
#     Better for documents with a few sharp topic shifts
splitter_std = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)

# (c) interquartile: cut where drop is an outlier in the IQR sense
#     Most robust — less sensitive to a single extreme value
splitter_iqr = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=1.5,
)

with open("my_document.txt") as f:
    text = f.read()

chunks = splitter_percentile.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:200])
    print()
```

---

### Option 3 — HuggingFace embeddings (no OpenAI dependency)

Drop-in replacement for the embeddings model — same splitter, fully local.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Runs locally — no API key needed
# good balance of speed and quality for semantic chunking
local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},   # swap "cuda" if available
)

splitter = SemanticChunker(
    embeddings=local_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
)

chunks = splitter.split_text(text)
print(f"{len(chunks)} chunks")
```

---

### Option 4 — Full RAG pipeline with semantic chunking

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Shared embedding model — used for BOTH chunking and indexing
embeddings = OpenAIEmbeddings()

# 1. Load documents
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
raw_docs = loader.load()

# 2. Chunk semantically
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
)

chunked_docs = splitter.split_documents(raw_docs)

# 3. Enrich metadata
for i, doc in enumerate(chunked_docs):
    doc.metadata["chunk_index"]  = i
    doc.metadata["char_count"]   = len(doc.page_content)
    doc.metadata["word_count"]   = len(doc.page_content.split())

print(f"Produced {len(chunked_docs)} semantic chunks")

# 4. Index — note: embeddings are computed AGAIN here for the vector store
#    This is the double-embedding cost of semantic chunking
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory="./chroma_semantic",
)

# 5. Retrieve
query = "How does the retrieval step work in a RAG pipeline?"
results = vectorstore.similarity_search(query, k=3)

for i, result in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Words: {result.metadata['word_count']}  |  "
          f"Source: {result.metadata['source']}")
    print(result.page_content)
```

---

## The double-embedding cost

This is the most important engineering tradeoff to understand with semantic chunking. Every sentence in every document gets embedded **twice** — once during chunking (to find boundaries) and once during indexing (to build the retrieval vector). For a 1,000-document corpus where each document has 50 sentences, that's 50,000 embedding calls just for chunking, on top of the indexing embeddings.

Concrete strategies to manage this:

```python
import hashlib, json, os
from sentence_transformers import SentenceTransformer

# Cache sentence embeddings by content hash — skip re-embedding unchanged docs
class CachedEmbedder:
    def __init__(self, model_name: str, cache_dir: str = ".embed_cache"):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def embed(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            return json.load(open(path))
        vec = self.model.encode(text).tolist()
        json.dump(vec, open(path, "w"))
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
```

---

## Threshold strategy comparison

|Strategy|What it does|When to use|
|---|---|---|
|`percentile`|Cuts the bottom N% of similarities|Best general default — adapts to each document's distribution|
|`standard_deviation`|Cuts where drop > mean − N×std|Documents with a few sharp, obvious topic shifts|
|`interquartile`|Cuts statistical outliers in similarity distribution|Noisy corpora; most robust to extreme values|
|Fixed threshold|Cuts below an absolute similarity value|When you have domain-specific calibration data|

The right threshold is always empirical. Set up a small eval set of 20–50 queries with known relevant passages, run retrieval, and measure recall at different threshold values. A 5-point threshold shift can change chunk count by 30–50%, which directly impacts both retrieval quality and cost.

---

## When semantic chunking earns its cost

It's worth the embedding overhead when your documents have **heterogeneous topics in unstructured prose** — research papers that mix background, methods, results, and discussion; long interview transcripts; multi-topic blog posts; support ticket histories. In these cases, fixed or recursive chunking often buries the relevant passage inside a chunk dominated by surrounding irrelevant content.

If your documents already have clear heading structure, use structure-based chunking instead — it achieves the same semantic coherence at a fraction of the cost, because the author already marked the topic boundaries for you.