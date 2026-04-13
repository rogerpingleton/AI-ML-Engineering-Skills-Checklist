## Chunk overlap in RAG

Overlap is a sliding window applied across chunk boundaries. When you split a document, the last N characters (or tokens, or sentences) of chunk K are repeated as the opening of chunk K+1. The chunk boundaries move forward by `chunk_size - overlap` on each step rather than by the full `chunk_size`.

The problem it solves is fundamental: **any fixed split point is arbitrary**. A sentence that starts at the end of one chunk and finishes at the start of the next will be retrieved by neither, because neither chunk contains it whole. Overlap ensures that content near every boundary appears fully in at least one chunk.

---The amber segments are overlap — content that exists in two adjacent chunks. The step size shrinks as overlap increases, producing more chunks. Click any row to see exactly what text falls in its unique vs. overlap zones.

---

## What happens at a boundary without overlap

The failure mode is concrete. Consider this text split at character 200:

```
...the retrieval step significantly reduces hallucination compared|
to purely parametric generation, which is the key advantage of RAG.
```

The `|` is your chunk boundary. A query asking _"what is the key advantage of RAG?"_ retrieves neither chunk cleanly — the first ends mid-sentence, the second starts with a subordinate clause that only makes sense with the first half. With 50 characters of overlap, both chunks contain the full sentence and either can be retrieved.

---

## Python: overlap across all strategies

### Character overlap (fixed-size and recursive)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,     # 20% of chunk_size — a solid default
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_text(document)

# Verify overlap is working — adjacent chunks should share a tail/head
def check_overlap(chunks: list[str], expected_overlap: int) -> None:
    for i in range(len(chunks) - 1):
        tail = chunks[i][-expected_overlap:]
        head = chunks[i+1][:expected_overlap]
        shared = len(set(tail.split()) & set(head.split()))
        print(f"Chunks {i+1}→{i+2}: ~{shared} shared words in boundary zone")

check_overlap(chunks, expected_overlap=100)
```

### Token overlap (tiktoken)

When your `length_function` counts tokens, the overlap is in tokens too — more precise for LLM context window budgeting.

```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

enc = tiktoken.encoding_for_model("gpt-4o")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # tokens
    chunk_overlap=64,      # tokens — 12.5% of chunk_size
    length_function=lambda t: len(enc.encode(t)),
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_text(document)

for i, chunk in enumerate(chunks):
    toks = len(enc.encode(chunk))
    print(f"Chunk {i+1}: {toks} tokens")
```

### Sentence overlap

For sentence-based chunking, overlap is measured in whole sentences rather than characters, which is cleaner — you never repeat a half-sentence.

```python
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def sentence_chunks_with_overlap(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Group sentences into chunks with sentence-level overlap.
    overlap_sentences=1 means the last sentence of chunk N
    is the first sentence of chunk N+1.
    """
    if overlap_sentences >= sentences_per_chunk:
        raise ValueError("overlap_sentences must be less than sentences_per_chunk")

    sentences = sent_tokenize(text)
    step = sentences_per_chunk - overlap_sentences
    chunks = []
    i = 0

    while i < len(sentences):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append({
            "text": " ".join(group),
            "start_sentence": i,
            "end_sentence": i + len(group) - 1,
            "overlap_head": sentences[i] if i > 0 else None,
        })
        i += step

    return chunks


chunks = sentence_chunks_with_overlap(
    document,
    sentences_per_chunk=5,
    overlap_sentences=1,
)

for c in chunks:
    print(f"[sentences {c['start_sentence']}–{c['end_sentence']}]")
    if c["overlap_head"]:
        print(f"  Overlap head: {c['overlap_head'][:60]}...")
```

### Measuring overlap quality empirically

The real question is whether your overlap is actually preventing retrieval misses at boundaries. This function measures it directly:

```python
def measure_boundary_coverage(
    chunks: list[str],
    test_phrases: list[str],
) -> dict:
    """
    For each test phrase, check whether it appears whole in at least
    one chunk (success) or gets split across a boundary (failure).

    test_phrases should be sentences or clauses that you know span
    or sit near chunk boundaries in your document.
    """
    results = {"covered": [], "split": [], "not_found": []}

    for phrase in test_phrases:
        found_whole = any(phrase.lower() in c.lower() for c in chunks)
        found_partial = any(
            phrase[:len(phrase)//2].lower() in c.lower() or
            phrase[len(phrase)//2:].lower() in c.lower()
            for c in chunks
        )
        if found_whole:
            results["covered"].append(phrase)
        elif found_partial:
            results["split"].append(phrase)
        else:
            results["not_found"].append(phrase)

    total = len(test_phrases)
    results["coverage_rate"] = len(results["covered"]) / total if total else 0
    return results


# Example usage
test_phrases = [
    "reduces hallucination compared to purely parametric generation",
    "nearest chunks are retrieved via approximate nearest-neighbour search",
    "chunking strategy is one of the most impactful decisions",
]

coverage = measure_boundary_coverage(chunks, test_phrases)
print(f"Coverage rate: {coverage['coverage_rate']:.0%}")
print(f"Split phrases: {coverage['split']}")
```

---

## The tradeoffs

Overlap is not free. Every overlapping character gets embedded and stored twice, and retrieved twice. The costs compound across three dimensions:

**Storage** scales with `overlap / (chunk_size - overlap)`. At 10% overlap the overhead is ~11%. At 33% overlap it's 50% — you're storing half again as many tokens as the document contains.

**Retrieval noise** increases because adjacent chunks now share content, and both may score highly for the same query. If your top-k retriever returns chunks 3 and 4 and they share 30% of their content, you've effectively used two context-window slots for 1.4× the unique information. Post-retrieval deduplication helps:

```python
def deduplicate_chunks(
    chunks: list[str],
    similarity_threshold: float = 0.85,
) -> list[str]:
    """
    Remove chunks that are near-duplicates of an already-selected chunk.
    Uses simple character-level Jaccard similarity — swap for embedding
    cosine similarity in production for better accuracy.
    """
    def jaccard(a: str, b: str) -> float:
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        return len(a_words & b_words) / len(a_words | b_words)

    selected = []
    for candidate in chunks:
        if not any(
            jaccard(candidate, kept) >= similarity_threshold
            for kept in selected
        ):
            selected.append(candidate)

    return selected


# After retrieval, deduplicate before passing to LLM
retrieved_chunks = vectorstore.similarity_search(query, k=6)
texts = [r.page_content for r in retrieved_chunks]
unique_texts = deduplicate_chunks(texts, similarity_threshold=0.80)
# Pass unique_texts to your LLM prompt
```

**Embedding cost** at indexing time is linear in total stored characters. For large corpora, a high overlap ratio meaningfully increases your indexing bill.

---

## Choosing overlap size

The right overlap depends on where your content concentrates meaning relative to your chunk size.

```python
def recommend_overlap(
    chunk_size: int,
    content_type: str,  # "prose", "technical", "legal", "code", "faq"
    unit: str = "chars",
) -> dict:
    """
    Returns a recommended overlap range and reasoning.
    These are empirical starting points — always validate with eval.
    """
    profiles = {
        # (min_pct, max_pct, rationale)
        "prose":     (0.10, 0.20, "Ideas often complete within one sentence; moderate overlap covers cross-boundary sentences"),
        "technical": (0.15, 0.25, "Definitions and procedures reference prior context frequently"),
        "legal":     (0.20, 0.30, "Clauses refer back to definitions; high overlap reduces reference loss"),
        "code":      (0.08, 0.15, "Functions are self-contained; small overlap covers multi-line statements"),
        "faq":       (0.00, 0.05, "Each Q&A pair is atomic; overlap adds noise without benefit"),
    }

    lo_pct, hi_pct, rationale = profiles.get(content_type, (0.10, 0.20, "General default"))
    return {
        "chunk_size":      chunk_size,
        "unit":            unit,
        "overlap_min":     int(chunk_size * lo_pct),
        "overlap_max":     int(chunk_size * hi_pct),
        "overlap_pct_range": f"{int(lo_pct*100)}–{int(hi_pct*100)}%",
        "rationale":       rationale,
    }


for content_type in ["prose", "technical", "legal", "code", "faq"]:
    rec = recommend_overlap(512, content_type, unit="tokens")
    print(f"{content_type:12} → {rec['overlap_min']}–{rec['overlap_max']} tokens "
          f"({rec['overlap_pct_range']})  |  {rec['rationale'][:60]}")
```

Output:

```
prose        → 51–102 tokens (10–20%)  |  Ideas often complete within one sentence; moderate...
technical    → 76–128 tokens (15–25%)  |  Definitions and procedures reference prior context...
legal        → 102–153 tokens (20–30%) |  Clauses refer back to definitions; high overlap re...
code         → 40–76 tokens  (8–15%)   |  Functions are self-contained; small overlap covers...
faq          → 0–25 tokens   (0–5%)    |  Each Q&A pair is atomic; overlap adds noise without...
```

---

## The practical rule

**Start at 10–20% of your chunk size.** For a 512-token chunk that's 50–100 tokens. Measure retrieval recall on a sample of boundary-spanning queries. If recall is poor, increase overlap. If your retrieved context is repetitive (the LLM is reading the same sentence twice per answer), reduce it. The sweet spot is the smallest overlap that keeps boundary-spanning content retrievable — beyond that you're paying storage and noise costs for diminishing returns.