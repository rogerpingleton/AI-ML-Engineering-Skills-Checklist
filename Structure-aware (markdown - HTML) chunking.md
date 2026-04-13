## Structure-based chunking in RAG

Structure-based chunking uses the document's own organization as the split signal. Rather than counting characters or detecting sentence boundaries, you split on logical units: headings, sections, paragraphs, HTML tags, or code blocks. The document tells you where the boundaries are — you just respect them.

This is the right default for **structured content**: documentation sites, markdown wikis, HTML pages, PDFs with clear section headers, legal documents. The payoff is that each chunk maps to a meaningful, self-contained unit of information, and you get rich metadata (section title, depth, breadth) for free.

---Toggle between Markdown and HTML modes, then click any node to inspect its chunk content and metadata. Notice how structure-based chunking naturally produces a breadcrumb path for each chunk — that's metadata you can filter on during retrieval.

---

## Python implementations

### Option 1 — Markdown: `MarkdownHeaderTextSplitter` (LangChain)

The most practical starting point for markdown documentation.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

markdown_doc = """
# RAG Pipeline Guide

This guide covers the key components of a RAG pipeline.

## 1. Document Ingestion

Ingestion is the first stage. You load raw files and convert them to plain text.

### 1.1 File Loaders

Use format-specific loaders. PDFs need layout-aware parsing. HTML should strip
nav and footer boilerplate.

### 1.2 Cleaning

Remove boilerplate: headers, footers, page numbers. Normalise whitespace.

## 2. Chunking

Chunking splits cleaned text into retrievable units.

### 2.1 Structure-based

Use the document's own headings as split boundaries. Best for docs and wikis.

## 3. Retrieval

Embed chunks and retrieve top-k neighbours at query time.
"""

# Define which heading levels become split boundaries
headers_to_split_on = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # keep heading text inside the chunk
)

chunks = splitter.split_text(markdown_doc)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)   # {"h1": "...", "h2": "...", "h3": "..."}
    print()
```

Each chunk's `.metadata` dict carries the full heading breadcrumb — extremely useful for filtering retrieval to a specific section:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

# Retrieve only from the "Chunking" section
results = vectorstore.similarity_search(
    query="what splitting strategies exist?",
    k=3,
    filter={"h2": "2. Chunking"},   # metadata filter
)
```

---

### Option 2 — Markdown: pure Python (no LangChain)

When you need full control over what goes into each chunk — including building a breadcrumb path yourself.

```python
import re
from dataclasses import dataclass, field


@dataclass
class StructuredChunk:
    text: str
    metadata: dict = field(default_factory=dict)


def parse_markdown_chunks(text: str) -> list[StructuredChunk]:
    """
    Split markdown on heading boundaries.
    Each chunk inherits the full heading breadcrumb as metadata.
    """
    lines = text.splitlines()
    heading_pattern = re.compile(r'^(#{1,6})\s+(.*)')

    chunks: list[StructuredChunk] = []
    current_lines: list[str] = []
    # Track current heading at each depth level
    breadcrumb: dict[int, str] = {}

    def flush(crumb: dict[int, str]):
        content = "\n".join(current_lines).strip()
        if content:
            # Build path string: "H1 > H2 > H3"
            path = " > ".join(
                crumb[d] for d in sorted(crumb) if crumb[d]
            )
            chunks.append(StructuredChunk(
                text=content,
                metadata={
                    **{f"h{d}": t for d, t in crumb.items()},
                    "path": path,
                    "char_count": len(content),
                }
            ))

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            flush(dict(breadcrumb))
            current_lines = []
            depth = len(match.group(1))       # number of # characters
            title = match.group(2).strip()
            breadcrumb[depth] = title
            # Clear all deeper levels when we step up
            for d in list(breadcrumb):
                if d > depth:
                    del breadcrumb[d]
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush(dict(breadcrumb))  # flush the last section
    return chunks


# --- Example ---
chunks = parse_markdown_chunks(markdown_doc)

for c in chunks:
    print(f"[{c.metadata.get('path', '—')}]")
    print(c.text[:120], "...")
    print()
```

---

### Option 3 — HTML: BeautifulSoup

HTML documents have rich semantic structure — `<article>`, `<section>`, `<h1>`–`<h6>`, `<p>`. BeautifulSoup makes it easy to walk that tree and chunk by section.

```python
from bs4 import BeautifulSoup
from dataclasses import dataclass, field


@dataclass
class HtmlChunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_html_by_sections(html: str) -> list[HtmlChunk]:
    """
    Split HTML into chunks at every heading boundary (h1–h4).
    Each chunk gets the heading text, tag, and CSS path as metadata.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading_tags = {"h1", "h2", "h3", "h4"}
    chunks: list[HtmlChunk] = []

    # Find every heading and collect sibling content until the next heading
    for heading in soup.find_all(heading_tags):
        title = heading.get_text(strip=True)
        level = heading.name          # "h1", "h2", etc.

        # Collect text from siblings that follow this heading
        content_parts = [title]
        for sibling in heading.next_siblings:
            if sibling.name in heading_tags:
                break
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                content_parts.append(text)

        full_text = "\n\n".join(content_parts)

        chunks.append(HtmlChunk(
            text=full_text,
            metadata={
                "tag":        level,
                "heading":    title,
                "section_id": heading.get("id", ""),
                "char_count": len(full_text),
            }
        ))

    return chunks


# --- Example ---
html_doc = """
<article id="rag-guide">
  <h1>RAG Pipeline Guide</h1>
  <p>A practical overview of building production RAG systems.</p>

  <section id="ingestion">
    <h2>Document Ingestion</h2>
    <p>Load and clean raw documents before chunking.</p>
    <p>Handle PDF, HTML, and markdown with format-specific loaders.</p>
  </section>

  <section id="chunking">
    <h2>Chunking</h2>
    <p>Structure-based chunking splits on headings and sections.</p>
    <h3>Markdown</h3>
    <p>Use MarkdownHeaderTextSplitter for clean heading-based splits.</p>
    <h3>HTML</h3>
    <p>Walk the DOM tree and split at heading boundaries.</p>
  </section>
</article>
"""

chunks = chunk_html_by_sections(html_doc)
for c in chunks:
    print(f"[{c.metadata['tag'].upper()} — {c.metadata['heading']}]")
    print(c.text[:150])
    print()
```

---

### Option 4 — Full pipeline with metadata filtering

Putting it together end-to-end. The metadata each chunk carries enables a powerful pattern: **pre-filtering before vector search**, which is faster and more precise than relying on semantic similarity alone.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. Parse markdown into structured chunks
headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
header_chunks = md_splitter.split_text(markdown_doc)

# 2. Optionally sub-split large sections (a section can still be huge)
from langchain.text_splitter import RecursiveCharacterTextSplitter

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
)
final_chunks = char_splitter.split_documents(header_chunks)
# Metadata (h1, h2, h3 breadcrumbs) is preserved through the split

# 3. Embed and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(final_chunks, embedding=embeddings)

# 4a. Semantic search across entire corpus
results = vectorstore.similarity_search("what is chunking?", k=3)

# 4b. Filtered search — only look in the "Chunking" section
results_filtered = vectorstore.similarity_search(
    query="what splitting strategies exist?",
    k=3,
    filter={"h2": "2. Chunking"},
)

for r in results_filtered:
    print(r.metadata)
    print(r.page_content[:200])
    print()
```

The two-stage split is an important pattern: `MarkdownHeaderTextSplitter` preserves semantic boundaries; `RecursiveCharacterTextSplitter` enforces a size ceiling. Neither alone is sufficient — the first can produce enormous chunks for long sections, while the second ignores structure entirely.

---

## The key engineering advantage: metadata

Structure-based chunking produces something the other strategies don't: **queryable metadata**. Every chunk knows where it lives in the document hierarchy. This unlocks several retrieval patterns:

**Section filtering** constrains the search space before ANN lookup — faster and more precise when users ask about a specific section. **Path-based ranking** lets you boost chunks whose breadcrumb matches the user's apparent intent. **Section-aware reranking** can prefer a chunk from `H2: Deployment` over `H2: Introduction` when the query is about production concerns.

The practical implication: when your documents have structure, always extract and store it as metadata. Even if you don't use it immediately, it costs nothing at index time and opens up retrieval strategies that purely text-based chunks can't support.