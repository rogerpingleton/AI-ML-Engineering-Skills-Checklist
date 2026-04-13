## Fixed-size chunking in RAG

Fixed-size chunking is the most straightforward strategy: you divide a document into chunks of a predetermined size, measured in either characters or tokens, with an optional overlap between consecutive chunks. There's no analysis of the content — the splitter doesn't know or care where sentences end or topics shift. It just counts and cuts.

This makes it fast and predictable, which is why it's often the first thing engineers reach for when standing up a new RAG pipeline.

---

## How it works visuallyDrag the sliders to see how chunk size and overlap interact. Notice how increasing overlap reduces the step size (how far each chunk advances), increasing the total chunk count — which is a direct storage and embedding cost tradeoff.

---

## Python implementation

### Option 1 — Pure Python (no dependencies)

Good to understand the mechanics before reaching for a library.

```python
def fixed_size_chunk(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """
    Split text into fixed-size chunks with optional overlap.

    Args:
        text:       The input document string.
        chunk_size: Max characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += step

    return chunks


# Example
doc = """Retrieval-Augmented Generation (RAG) is a technique that combines
large language models with external knowledge retrieval. The model retrieves
relevant passages from a corpus and conditions its output on them."""

chunks = fixed_size_chunk(doc, chunk_size=100, overlap=20)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(repr(chunk))
    print()
```

### Option 2 — LangChain `CharacterTextSplitter`

The most common production path. Splits on a separator character first, then enforces the size limit.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=100,    # characters shared between adjacent chunks
    separator="\n",       # try to split here first; falls back to mid-string
    length_function=len,  # swap for tiktoken to count tokens instead
)

with open("my_document.txt") as f:
    text = f.read()

chunks = splitter.split_text(text)
print(f"{len(chunks)} chunks produced")
print(chunks[0])
```

### Option 3 — Token-based splitting with tiktoken

When you're budgeting against an LLM's context window, you want token counts, not character counts. Characters per token vary (roughly 4 chars/token for English, less for code or other languages).

```python
import tiktoken
from langchain.text_splitter import CharacterTextSplitter

# Use the tokenizer for the model you're targeting
enc = tiktoken.encoding_for_model("gpt-4o")

def token_length(text: str) -> int:
    return len(enc.encode(text))

splitter = CharacterTextSplitter(
    chunk_size=256,           # tokens per chunk
    chunk_overlap=32,         # token overlap
    length_function=token_length,
    separator="\n",
)

chunks = splitter.split_text(document_text)
```

### Option 4 — Full RAG pipeline with fixed-size chunking

Putting it all together — chunk, embed, index, and retrieve.

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Load your document
with open("my_document.txt") as f:
    text = f.read()

# 2. Chunk it
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separator="\n",
)
chunks = splitter.split_text(text)
print(f"Produced {len(chunks)} chunks")

# 3. Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=[{"chunk_index": i} for i in range(len(chunks))],
)

# 4. Retrieve
query = "What is RAG and why is it useful?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
```

---

## The overlap parameter in detail

Overlap is the most important tuning knob for fixed-size chunking. Consider this document split at the `|` boundary:

```
...the model generates an answer based on the retrieved| passages. This step is
called generation and is the second phase of the pipeline...
```

Without overlap, the chunk ending at `retrieved` and the one starting at `passages` are independent. A query about "the generation phase" might retrieve neither cleanly. With an overlap of 50 characters, both chunks contain the phrase "retrieved passages. This step is called generation" — the retrieval succeeds from either side.

**Practical defaults to start with:**

|Chunk size|Overlap|Use case|
|---|---|---|
|256 tokens|32 tokens|Dense technical docs, precise Q&A|
|512 tokens|64 tokens|General knowledge base|
|1024 tokens|128 tokens|Long-form content, summaries|

---

## Tradeoffs to keep in mind

Fixed-size chunking ignores the content entirely — it will happily cut a sentence, a code block, or a table in half. For many corpora this is acceptable (the overlap cushions the worst cases), but for structured content like markdown documentation or code files, you'll quickly find yourself wanting to move to recursive or structure-aware splitting. Think of fixed-size as the baseline you measure other strategies against, not necessarily where you stop.