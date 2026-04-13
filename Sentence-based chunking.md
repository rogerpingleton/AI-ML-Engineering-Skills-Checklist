## Sentence-based chunking in RAG

Sentence-based chunking respects the natural linguistic boundaries of your text. Instead of counting characters and cutting blindly, you split on sentence boundaries — periods, question marks, exclamation points — and then group sentences together until you hit a size limit. The result is chunks that always contain complete thoughts, never a half-finished sentence.

This matters for retrieval because embedding models are trained on coherent text. A chunk ending mid-sentence produces a degraded embedding — the model saw something incomplete and produces a vector that doesn't cleanly represent any real concept.

---

## How it works visually



Sentences with a dashed border appear in two chunks — that's your overlap working. Click any chunk in the legend or the text to isolate it.

---

## Python implementation

### Option 1 — Pure Python (no dependencies)

Useful for understanding the mechanics and for simple corpora where NLTK is overkill.

```python
import re

def split_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter using regex.
    Handles common abbreviations poorly — use NLTK for production.
    """
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def sentence_chunks(
    text: str,
    sentences_per_chunk: int = 3,
    overlap: int = 1,
) -> list[str]:
    """
    Group sentences into chunks with optional sentence-level overlap.

    Args:
        text:                 Input document.
        sentences_per_chunk:  How many sentences per chunk.
        overlap:              How many sentences to repeat in the next chunk.

    Returns:
        List of chunk strings.
    """
    if overlap >= sentences_per_chunk:
        raise ValueError("overlap must be less than sentences_per_chunk")

    sentences = split_sentences(text)
    step = sentences_per_chunk - overlap
    chunks = []

    i = 0
    while i < len(sentences):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append(" ".join(group))
        i += step

    return chunks


# --- Example ---
doc = """
RAG stands for Retrieval-Augmented Generation. It combines language models
with external knowledge retrieval. The system first retrieves relevant documents,
then passes them as context to the model. This reduces hallucinations significantly.
Dense vector search is the most common retrieval mechanism. Chunks are embedded
into a high-dimensional space and indexed for fast nearest-neighbour lookup.
"""

chunks = sentence_chunks(doc, sentences_per_chunk=3, overlap=1)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
```

---

### Option 2 — NLTK (recommended for production)

The regex approach above breaks on "Dr. Smith said..." or "i.e., the result...". NLTK's `sent_tokenize` handles these edge cases correctly using a pre-trained Punkt tokenizer.

```python
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # required in newer NLTK versions

from nltk.tokenize import sent_tokenize


def sentence_chunks_nltk(
    text: str,
    sentences_per_chunk: int = 3,
    overlap: int = 1,
) -> list[dict]:
    """
    Returns chunks as dicts with text + metadata (sentence indices).
    The metadata is useful later for debugging retrieval.
    """
    if overlap >= sentences_per_chunk:
        raise ValueError("overlap must be less than sentences_per_chunk")

    sentences = sent_tokenize(text)
    step = sentences_per_chunk - overlap
    chunks = []

    i = 0
    while i < len(sentences):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append({
            "text": " ".join(group),
            "sentence_start": i,
            "sentence_end": i + len(group) - 1,
            "num_sentences": len(group),
        })
        i += step

    return chunks


# --- Example ---
doc = """
Dr. Smith published the findings in Jan. 2024. The study covered 1,200 patients
across three U.S. hospitals. Results showed a 34% improvement vs. the control group.
Statistical significance was confirmed at p < 0.01. The team concluded that early
intervention is key. Follow-up studies are planned for Q3 2025.
"""

chunks = sentence_chunks_nltk(doc, sentences_per_chunk=3, overlap=1)

for c in chunks:
    print(f"[sentences {c['sentence_start']}–{c['sentence_end']}]")
    print(c["text"])
    print()
```

Notice that "Dr. Smith" and "Jan. 2024" don't cause spurious splits — that's NLTK doing its job.

---

### Option 3 — spaCy (best linguistic quality)

spaCy's sentence segmentation uses a dependency parser, which gives the most accurate results on complex text — nested clauses, quoted speech, scientific notation.

```python
import spacy

# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000  # raise for large documents


def sentence_chunks_spacy(
    text: str,
    sentences_per_chunk: int = 3,
    overlap: int = 1,
) -> list[dict]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    step = sentences_per_chunk - overlap
    chunks = []
    i = 0

    while i < len(sentences):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append({
            "text": " ".join(group),
            "sentence_start": i,
            "sentence_end": i + len(group) - 1,
        })
        i += step

    return chunks
```

---

### Option 4 — LangChain `NLTKTextSplitter`

LangChain wraps NLTK with its standard splitter interface, which integrates cleanly into a full RAG pipeline.

```python
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from langchain.text_splitter import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. Load your document
with open("my_document.txt") as f:
    raw_text = f.read()

# 2. Chunk by sentences
splitter = NLTKTextSplitter(
    chunk_size=500,    # max characters per chunk (groups sentences until limit)
    chunk_overlap=100, # character overlap between chunks
)
chunks = splitter.split_text(raw_text)
print(f"{len(chunks)} chunks produced")

# Wrap in Documents to attach metadata
docs = [
    Document(page_content=chunk, metadata={"chunk_index": i})
    for i, chunk in enumerate(chunks)
]

# 3. Embed and index
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# 4. Retrieve
query = "How does RAG reduce hallucinations?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} (chunk {doc.metadata['chunk_index']}) ---")
    print(doc.page_content)
```

---

## Choosing your sentence splitter

|Tool|Accuracy|Speed|Best for|
|---|---|---|---|
|`re.split`|Low|Very fast|Quick prototypes, clean text|
|`nltk.sent_tokenize`|Good|Fast|Most production use cases|
|`spacy`|Best|Moderate|Legal, medical, scientific text|
|`LangChain NLTKTextSplitter`|Good|Fast|LangChain pipelines|

---

## Tuning tips

**Sentences per chunk** controls the semantic density of each chunk. Too few (1–2) and each chunk is too narrow — a single sentence often lacks enough context for the embedding to be discriminative. Too many (6+) and you're approaching fixed-size behavior, defeating the purpose. Three to four sentences is a reliable starting point for most corpora.

**Sentence overlap** at 1 means the last sentence of chunk N becomes the first sentence of chunk N+1. This is usually enough to handle cross-boundary references. Unlike character overlap, sentence overlap is cheap — you're repeating one complete thought, not an arbitrary string fragment.

**When to move on from sentence chunking:** if your documents have clear section structure (headings, numbered lists), you'll get better retrieval from structure-aware or recursive splitting, because sentence boundaries don't capture that higher-level organization. Sentence chunking shines on flat prose — articles, transcripts, reports — where structure is purely linguistic.