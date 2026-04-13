## Recursive chunking in RAG

Recursive chunking is the most practical general-purpose strategy and the one most experienced RAG engineers reach for first. The core idea: instead of one fixed separator, you give the splitter a **priority-ordered list of separators**. It tries the first one (double newline — paragraph breaks). If a resulting piece is still too large, it tries the next (single newline). Then a sentence-ending period. Then a space. Then individual characters as a last resort.

This means the splitter always cuts at the _most natural boundary available_ — it only gets more aggressive when it has to.

---Drag the chunk size down to force the splitter deeper into the separator list. Notice how it only falls back to word or character splits when it absolutely must — that's the recursive logic at work.

---

## Python implementations

### The core splitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # max characters per chunk
    chunk_overlap=200,     # characters repeated across adjacent chunks
    separators=[           # tried in this order, left to right
        "\n\n",            # paragraph break  — most preferred
        "\n",              # line break
        ". ",              # sentence boundary
        " ",               # word boundary
        "",                # hard character cut — last resort
    ],
    length_function=len,   # swap for token counter (see Option 3)
)

with open("my_document.txt") as f:
    text = f.read()

chunks = splitter.split_text(text)

print(f"{len(chunks)} chunks produced")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
```

### Option 2 — Splitting `Document` objects with metadata preservation

In production you almost always have metadata (filename, source URL, page number) that should travel with every chunk. Use `split_documents` instead of `split_text`.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Create documents with source metadata
documents = [
    Document(
        page_content=open("rag_guide.md").read(),
        metadata={"source": "rag_guide.md", "doc_type": "markdown"},
    ),
    Document(
        page_content=open("faq.txt").read(),
        metadata={"source": "faq.txt", "doc_type": "text"},
    ),
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

# split_documents preserves and propagates metadata to every chunk
chunks = splitter.split_documents(documents)

for chunk in chunks[:4]:
    print(chunk.metadata)        # source + doc_type intact on every chunk
    print(chunk.page_content[:120])
    print()
```

### Option 3 — Token-aware splitting with tiktoken

Character counts don't map predictably to LLM context windows. Code, non-Latin scripts, and technical jargon tokenise very differently. Use a token counter as the `length_function` to set precise limits.

```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

enc = tiktoken.encoding_for_model("gpt-4o")

def token_len(text: str) -> int:
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # tokens, not characters
    chunk_overlap=64,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

with open("my_document.txt") as f:
    text = f.read()

chunks = splitter.split_text(text)

# Verify token counts
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}: {token_len(chunk)} tokens, {len(chunk)} chars")
    print(chunk[:100], "...\n")
```

### Option 4 — Language-specific separators

For code files, `RecursiveCharacterTextSplitter` ships with pre-built separator lists tuned for each programming language — it splits on class/function boundaries first, falling back to lines, then statements.

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# Python: splits on class → function → decorator → block → line
py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)

python_code = open("my_module.py").read()
code_chunks = py_splitter.split_text(python_code)

# Inspect what separators it uses for Python
print(py_splitter._separators)
# ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

# Works the same for other languages
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=1000, chunk_overlap=100
)
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=100
)
# Other supported: RUST, GO, CPP, RUBY, TS, HTML, LATEX, SOL, ...
```

### Option 5 — Full RAG pipeline

End-to-end: load files, chunk recursively, embed, index, and retrieve.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import tiktoken

# --- 1. Load documents ---
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# --- 2. Chunk recursively with token counting ---
enc = tiktoken.encoding_for_model("gpt-4o")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    length_function=lambda t: len(enc.encode(t)),
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(documents)
print(f"Produced {len(chunks)} chunks")

# --- 3. Enrich metadata ---
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = i
    chunk.metadata["token_count"] = len(enc.encode(chunk.page_content))

# --- 4. Embed and store ---
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# --- 5. Retrieve ---
query = "How does recursive chunking decide where to split?"
results = vectorstore.similarity_search(query, k=4)

for i, result in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {result.metadata['source']}  |  "
          f"Tokens: {result.metadata['token_count']}")
    print(result.page_content)
```

---

## Why this is the default starting point

The key insight is what each separator level handles:

`"\n\n"` catches paragraph breaks — the most common semantic boundary in any prose document. Most of the time, this is all that gets used.

`"\n"` catches line breaks within dense text — bullet lists, code comments, wrapped lines.

`". "` catches sentence endings only when a paragraph was still too large — so you get sentence-level precision without paying for it on short paragraphs.

`" "` and `""` are genuine last resorts. If you're hitting these regularly in production, your chunk size is too small for your content, and you should tune upward.

---

## Tuning guide

|Content type|`chunk_size`|`chunk_overlap`|Notes|
|---|---|---|---|
|General prose|512–1024 tokens|64–128|Default; paragraph splits dominate|
|Technical docs|256–512 tokens|32–64|Smaller for precise Q&A|
|Long-form articles|1024–2048 tokens|128–256|Larger to preserve narrative|
|Source code|400–800 tokens|50–100|Use `from_language()`|
|FAQ / short answers|128–256 tokens|0–32|Answers are already atomic|

The overlap rule of thumb holds across all of these: target 10–20% of `chunk_size`. Going higher wastes embedding budget; going lower risks missing cross-boundary content.