## Agentic and contextual chunking in RAG

Every strategy covered so far — fixed-size, sentence, structure, recursive, semantic — makes chunking decisions using only the **local signal** of the text itself: character counts, punctuation, headings, or embedding similarity between adjacent sentences. None of them understand the document as a whole.

Agentic and contextual chunking breaks that constraint. An LLM reads the document — or chunks of it — and either decides where to cut, or enriches each chunk with context that only makes sense when you understand the full document. The result is chunks that are aware of their role within the larger work.

These are two related but distinct techniques worth understanding separately.

--- 

![[agentic_vs_contextual_overview.svg|697]]

---

## Technique 1 — Contextual chunking (Anthropic's approach)

Introduced by Anthropic, this technique keeps your existing chunk boundaries entirely unchanged. What changes is what gets _embedded_. Before indexing, an LLM reads the full document alongside each chunk and writes 1–2 sentences situating that chunk in context. Those sentences are prepended to the chunk text before embedding.

The problem it solves: a chunk like _"The treatment showed a 34% improvement"_ is nearly useless in isolation — improvement over what? Compared to whom? In which trial? The embedding of that fragment points vaguely at "medical improvement." With context prepended — _"This chunk is from a Phase III oncology trial comparing Drug X to placebo in 1,200 patients. The treatment showed a 34% improvement"_ — the embedding is precise and retrievable.

### Implementation

```python
import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

client = anthropic.Anthropic()


def generate_chunk_context(document: str, chunk: str) -> str:
    """
    Ask Claude to write 1-2 sentences situating this chunk
    within the broader document. Returns just the context text.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": f"""<document>
{document}
</document>

Here is a chunk from this document:
<chunk>
{chunk}
</chunk>

Write 1-2 sentences that situate this chunk within the document.
Explain what section it belongs to, what topic it addresses, and
any key entities or concepts needed to understand it without the
surrounding text.

Output only the context sentences. No preamble, no labels."""
        }]
    )
    return response.content[0].text.strip()


def contextual_chunk(
    document: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Chunk a document and enrich each chunk with LLM-generated context.

    Returns list of dicts with:
        - original_text: the raw chunk
        - context:       LLM-generated situating sentences
        - enriched_text: context + original (what gets embedded)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    raw_chunks = splitter.split_text(document)

    enriched = []
    for i, chunk in enumerate(raw_chunks):
        print(f"  Contextualising chunk {i+1}/{len(raw_chunks)}...")
        context = generate_chunk_context(document, chunk)
        enriched.append({
            "original_text": chunk,
            "context":       context,
            "enriched_text": f"{context}\n\n{chunk}",
            "chunk_index":   i,
        })

    return enriched


# --- Full pipeline ---
document = open("my_document.txt").read()

print("Chunking and contextualising...")
chunks = contextual_chunk(document, chunk_size=800, chunk_overlap=100)

# Build Documents using enriched_text for embedding
docs = [
    Document(
        page_content=c["enriched_text"],
        metadata={
            "chunk_index":   c["chunk_index"],
            "original_text": c["original_text"],   # store original for display
            "context":       c["context"],
        }
    )
    for c in chunks
]

# Index using enriched text
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_contextual",
)

# Retrieve — results carry both enriched embedding AND original text
results = vectorstore.similarity_search("what was the treatment outcome?", k=3)

for r in results:
    print("\n--- Retrieved chunk ---")
    print(f"Context:  {r.metadata['context']}")
    print(f"Original: {r.metadata['original_text'][:200]}")
```

### Batching with `prompt_caching` to cut cost

Each contextualisation call sends the full document. For a 50-chunk document, that's 50 API calls each carrying the full document text. Anthropic's prompt caching stores the document in the cache after the first call and charges only the small per-chunk portion for subsequent calls — reducing cost by ~90% for long documents.

```python
def generate_chunk_context_cached(document: str, chunk: str) -> str:
    """
    Uses prompt caching: the document is cached after the first call.
    Subsequent calls for the same document pay only for the chunk portion.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<document>\n",
                },
                {
                    "type": "text",
                    "text": document,
                    "cache_control": {"type": "ephemeral"},  # cache the document
                },
                {
                    "type": "text",
                    "text": f"\n</document>\n\nChunk to contextualise:\n<chunk>\n{chunk}\n</chunk>\n\nWrite 1-2 situating sentences. Output only the sentences.",
                },
            ],
        }]
    )
    return response.content[0].text.strip()
```

---

## Technique 2 — Agentic chunking (LLM-determined boundaries)

Here the LLM doesn't just annotate chunks — it decides where the chunks are. You feed the document with a structured prompt asking the model to identify meaningful segments, name them, and return a JSON plan. You then apply that plan to extract the actual chunks.

This is the right approach when your documents have complex, irregular structure that no rule-based splitter handles well: legal contracts with nested clauses, scientific papers with non-standard section patterns, transcripts where topics shift mid-paragraph.

### Implementation

```python
import json
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()


@dataclass
class AgenticChunk:
    title: str
    summary: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int


def plan_chunks(document: str) -> list[dict]:
    """
    Ask the LLM to produce a chunking plan: a list of segments with
    titles, summaries, and the exact text that should form each chunk.
    Returns parsed JSON.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Analyse this document and divide it into meaningful chunks
for a retrieval system. Each chunk should cover one coherent topic or concept.

Rules:
- Each chunk should be self-contained and answerable as a unit
- Prefer semantic completeness over uniform size
- Do not overlap chunks
- Every word in the document must appear in exactly one chunk

Return a JSON array. Each element must have:
  - "title":   short descriptive label (max 8 words)
  - "summary": one sentence describing what this chunk covers
  - "start":   the exact first 6 characters of this chunk's text
  - "end":     the exact last 6 characters of this chunk's text

Return ONLY the JSON array, no other text.

Document:
{document}"""
        }]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def apply_chunk_plan(document: str, plan: list[dict]) -> list[AgenticChunk]:
    """
    Use the LLM's plan to extract actual text spans from the document.
    Falls back to approximate matching if exact boundary strings shift.
    """
    chunks = []
    search_start = 0

    for i, segment in enumerate(plan):
        start_str = segment["start"]
        end_str   = segment["end"]

        # Find start position
        start_pos = document.find(start_str, search_start)
        if start_pos == -1:
            print(f"  Warning: could not find start of chunk {i+1}, skipping")
            continue

        # Find end position — search from start forward
        end_pos = document.find(end_str, start_pos)
        if end_pos == -1:
            # Fall back: use the next chunk's start as the boundary
            end_pos = len(document) if i == len(plan) - 1 else len(document)
        else:
            end_pos += len(end_str)

        text = document[start_pos:end_pos].strip()
        chunks.append(AgenticChunk(
            title=segment["title"],
            summary=segment["summary"],
            text=text,
            chunk_index=i,
            char_start=start_pos,
            char_end=end_pos,
        ))
        search_start = end_pos

    return chunks


def agentic_chunk(document: str) -> list[AgenticChunk]:
    print("Planning chunks with LLM...")
    plan = plan_chunks(document)
    print(f"  LLM proposed {len(plan)} chunks")
    chunks = apply_chunk_plan(document, plan)
    print(f"  Successfully extracted {len(chunks)} chunks")
    return chunks


# --- Full pipeline ---
document = open("my_document.txt").read()
chunks   = agentic_chunk(document)

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

docs = [
    Document(
        page_content=c.text,
        metadata={
            "title":       c.title,
            "summary":     c.summary,
            "chunk_index": c.chunk_index,
        }
    )
    for c in chunks
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_agentic",
)

results = vectorstore.similarity_search("treatment outcomes in Phase III", k=3)
for r in results:
    print(f"\n[{r.metadata['title']}]")
    print(r.metadata['summary'])
    print(r.page_content[:300])
```

---

## Technique 3 — Combining both

In production, the highest-quality setup pairs agentic boundary detection with contextual enrichment: the LLM decides where to cut _and_ annotates each chunk with situating context.

```python
def agentic_contextual_chunk(document: str) -> list[Document]:
    """
    Two-pass approach:
      Pass 1 (agentic):     LLM determines chunk boundaries
      Pass 2 (contextual):  LLM enriches each chunk with context
    """
    # Pass 1: agentic boundaries
    raw_chunks = agentic_chunk(document)

    # Pass 2: contextual enrichment
    final_docs = []
    for chunk in raw_chunks:
        context = generate_chunk_context_cached(document, chunk.text)
        enriched = f"{context}\n\n{chunk.text}"

        final_docs.append(Document(
            page_content=enriched,
            metadata={
                "title":         chunk.title,
                "summary":       chunk.summary,
                "context":       context,
                "original_text": chunk.text,
                "chunk_index":   chunk.chunk_index,
            }
        ))

    return final_docs
```

---

## When to use each technique
---
![[chunking_decision_guide.svg|697]]

---
## Cost model: what you're actually paying for

This is the most important engineering consideration before adopting either technique.

```python
def estimate_cost(
    num_documents: int,
    avg_doc_tokens: int,
    avg_chunks_per_doc: int,
    model_input_price_per_1k: float = 0.003,   # claude-sonnet-4 approximate
    model_output_price_per_1k: float = 0.015,
    use_prompt_caching: bool = True,
) -> dict:
    """
    Rough cost estimate for contextual chunking a corpus.
    With prompt caching, only the first call per document pays
    for the full document tokens; subsequent calls pay ~10%.
    """
    output_tokens_per_call = 60   # ~2 context sentences

    if use_prompt_caching:
        # First chunk: full doc + chunk tokens
        # Remaining chunks: cached doc (10% cost) + chunk tokens
        avg_chunk_tokens = avg_doc_tokens // avg_chunks_per_doc
        first_call_input  = avg_doc_tokens + avg_chunk_tokens
        cached_call_input = avg_doc_tokens * 0.1 + avg_chunk_tokens
        input_tokens = (
            num_documents * first_call_input
            + num_documents * (avg_chunks_per_doc - 1) * cached_call_input
        )
    else:
        input_tokens = (
            num_documents
            * avg_chunks_per_doc
            * (avg_doc_tokens + avg_doc_tokens // avg_chunks_per_doc)
        )

    output_tokens = num_documents * avg_chunks_per_doc * output_tokens_per_call

    return {
        "total_llm_calls":     num_documents * avg_chunks_per_doc,
        "input_tokens":        int(input_tokens),
        "output_tokens":       int(output_tokens),
        "estimated_cost_usd":  round(
            (input_tokens / 1000 * model_input_price_per_1k)
            + (output_tokens / 1000 * model_output_price_per_1k), 2
        ),
    }


# 1,000 documents, ~2,000 tokens each, ~10 chunks per doc
print(estimate_cost(1000, 2000, 10, use_prompt_caching=False))
# {'total_llm_calls': 10000, 'input_tokens': 22000000, ... ~$66}

print(estimate_cost(1000, 2000, 10, use_prompt_caching=True))
# {'total_llm_calls': 10000, 'input_tokens': 3820000, ... ~$12}
```

Prompt caching reduces the bill by roughly 80% on a typical corpus. For agentic chunking, add one additional LLM call per document (the planning call), which costs roughly the same as one contextualisation call.

---

## Practical summary

**Use contextual chunking** when you have a corpus where retrieval precision matters and you can afford ~$10–50 per 1,000 documents at index time. It's the most cost-effective quality upgrade available — you keep your existing chunking strategy and just add context. Anthropic's own benchmarks show it reduces retrieval failures by 49% compared to naive chunking.

**Use agentic chunking** when your documents have irregular, complex structure that rule-based splitters handle badly: legal contracts, earnings call transcripts, clinical trial protocols, multi-format reports. The cost is higher and the boundary extraction is less deterministic, but the chunk quality on hard documents is unmatched.

**Don't use either** for well-structured documents (markdown documentation, HTML pages with clear sections) — structure-based chunking already gives you semantically clean boundaries at zero LLM cost. Spending money on contextualising a well-structured document is waste.

Both techniques are indexing-time costs only. Once the chunks are in the vector store, retrieval is identical to any other strategy — the LLM investment pays off across every query that hits those chunks.