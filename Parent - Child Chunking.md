## Parent-child (hierarchical) chunking in RAG

Every other chunking strategy forces a single tradeoff between retrieval precision and context richness. Small chunks embed well and retrieve precisely but lack context. Large chunks provide rich context but embed poorly — the vector averages over too many topics and matches nothing specifically.

Parent-child chunking resolves this by maintaining **two separate representations of the same content**: small child chunks for retrieval, large parent chunks for generation. The vector index only contains children. When a child is retrieved, the system fetches its parent and sends _that_ to the LLM instead.

You get the precision of small-chunk retrieval and the context richness of large-chunk generation — from the same document, at the same time.

---Click any child chunk (purple) to see its parent (teal) and the connector linking them. Adjust the size sliders to see how the parent-to-child ratio changes.

---

## How the retrieval flow works

![[parent_child_retrieval_flow.svg|692]]

The vector index and the docstore are separate systems serving separate purposes. The index is optimised for similarity search over many small vectors. The docstore is optimised for fast key-value lookup by `parent_id`. Neither needs to do what the other does.

---

## Python implementations

### Option 1 — LangChain `ParentDocumentRetriever`

LangChain's built-in implementation handles the wiring between vector store and docstore automatically.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

# --- Splitters ---
# Parent: large chunks the LLM reads for context
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
)

# Child: small chunks the retriever matches queries against
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)

# --- Stores ---
embeddings  = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=embeddings,
)
docstore = InMemoryStore()   # swap for RedisStore / SQLStore in production

# --- Retriever ---
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# --- Index ---
loader = TextLoader("my_document.txt")
docs   = loader.load()

retriever.add_documents(docs, ids=None)
# Internally: splits to parents, splits each parent to children,
# stores parents in docstore, embeds and indexes children in vectorstore

# --- Retrieve ---
# Returns parent chunks, not the child that matched
query   = "How does vector search work in RAG?"
results = retriever.get_relevant_documents(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ({len(doc.page_content)} chars) ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
```

### Option 2 — Pure Python (shows the mechanics)

Build it from scratch so you fully understand the data structures involved.

```python
import uuid
from dataclasses import dataclass, field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


@dataclass
class ParentChunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    id: str
    parent_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class ParentChildRetriever:
    """
    Maintains two stores:
      docstore:    parent_id → ParentChunk  (full context for the LLM)
      vectorstore: child embeddings         (precision for retrieval)
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size:  int = 300,
        parent_overlap:    int = 100,
        child_overlap:     int = 30,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )
        self.docstore:    dict[str, ParentChunk] = {}   # parent_id → ParentChunk
        self.child_index: list[ChildChunk]       = []
        self.vectorstore = None
        self.embeddings  = OpenAIEmbeddings()

    def add_documents(self, documents: list[Document]) -> None:
        all_child_docs: list[Document] = []

        for doc in documents:
            # 1. Split into parents
            parent_texts = self.parent_splitter.split_text(doc.page_content)

            for parent_text in parent_texts:
                parent_id = str(uuid.uuid4())

                # 2. Store parent in docstore
                self.docstore[parent_id] = ParentChunk(
                    id=parent_id,
                    text=parent_text,
                    metadata={**doc.metadata},
                )

                # 3. Split parent into children
                child_texts = self.child_splitter.split_text(parent_text)

                for child_text in child_texts:
                    child_id = str(uuid.uuid4())
                    child    = ChildChunk(
                        id=child_id,
                        parent_id=parent_id,
                        text=child_text,
                        metadata={**doc.metadata, "parent_id": parent_id},
                    )
                    self.child_index.append(child)

                    # Wrap as Document for the vectorstore
                    all_child_docs.append(Document(
                        page_content=child_text,
                        metadata={"child_id": child_id, "parent_id": parent_id},
                    ))

        # 4. Embed and index all children
        self.vectorstore = Chroma.from_documents(
            documents=all_child_docs,
            embedding=self.embeddings,
        )
        print(f"Indexed {len(self.docstore)} parents, "
              f"{len(all_child_docs)} children")

    def retrieve(self, query: str, k: int = 3) -> list[ParentChunk]:
        """
        1. Embed query and find top-k matching CHILD chunks.
        2. Look up their PARENT chunks.
        3. Deduplicate (multiple children can share a parent).
        4. Return parent chunks — these go to the LLM.
        """
        if not self.vectorstore:
            raise RuntimeError("No documents indexed yet")

        # Retrieve more children than k to account for deduplication
        child_results = self.vectorstore.similarity_search(query, k=k * 3)

        seen_parent_ids: set[str] = set()
        parents: list[ParentChunk] = []

        for child_doc in child_results:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent = self.docstore.get(parent_id)
                if parent:
                    parents.append(parent)
                if len(parents) >= k:
                    break

        return parents


# --- Usage ---
retriever = ParentChildRetriever(
    parent_chunk_size=1200,
    child_chunk_size=250,
)

docs = [Document(page_content=open("my_document.txt").read(),
                 metadata={"source": "my_document.txt"})]
retriever.add_documents(docs)

results = retriever.retrieve("What is the role of vector search in RAG?", k=3)

for i, parent in enumerate(results):
    print(f"\n--- Parent {i+1} ({len(parent.text)} chars) ---")
    print(parent.text[:400], "...")
```

### Option 3 — Three-level hierarchy

For very long documents — books, large codebases, lengthy legal contracts — you can extend the pattern to three levels: document → section → sentence. The retriever matches sentences; the LLM receives sections.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Level 1: full document sections (what the LLM reads)
section_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

# Level 2: paragraphs (intermediate — not used in retrieval or generation)
# Level 3: sentences (what gets embedded and retrieved)
sentence_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

vectorstore = Chroma(
    collection_name="sentence_index",
    embedding_function=OpenAIEmbeddings(),
)
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=sentence_splitter,
    parent_splitter=section_splitter,   # LLM receives sections
)

from langchain_community.document_loaders import TextLoader
docs = TextLoader("large_document.txt").load()
retriever.add_documents(docs)

# Retrieval: matches sentence-level, returns section-level
results = retriever.get_relevant_documents("your query here")
```

### Option 4 — Persistent docstore

`InMemoryStore` is lost on restart. For production, back the docstore with Redis or a database.

```python
from langchain.storage import RedisStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Redis-backed docstore — survives restarts
docstore = RedisStore(redis_url="redis://localhost:6379", namespace="rag_parents")

vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",    # also persistent
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100),
)
```

---

## Sizing the parent-to-child ratio

The ratio between parent and child size is the most important tuning decision. It controls how much context the LLM receives per retrieved match.

```python
def estimate_context_per_query(
    parent_size_tokens: int,
    child_size_tokens: int,
    k: int,                           # number of parents returned
    llm_context_window: int = 128000,
    system_prompt_tokens: int = 500,
    answer_tokens: int = 500,
) -> dict:
    """
    Estimates how much of the LLM's context window is used per query.
    """
    parent_to_child_ratio = parent_size_tokens / child_size_tokens
    context_used  = k * parent_size_tokens
    context_budget = llm_context_window - system_prompt_tokens - answer_tokens
    utilisation   = context_used / context_budget

    return {
        "parent_size_tokens":     parent_size_tokens,
        "child_size_tokens":      child_size_tokens,
        "parent_to_child_ratio":  round(parent_to_child_ratio, 1),
        "context_used_tokens":    context_used,
        "context_budget_tokens":  context_budget,
        "utilisation_pct":        round(utilisation * 100, 1),
        "fits_in_context":        utilisation <= 1.0,
    }


# Compare common configurations
configs = [
    (1500, 300, 3, "Standard"),
    (3000, 200, 3, "Large parent"),
    (800,  150, 5, "More results"),
    (500,  100, 8, "Many small parents"),
]

for parent, child, k, label in configs:
    est = estimate_context_per_query(parent, child, k)
    print(f"{label:20} | ratio {est['parent_to_child_ratio']:4.1f}x "
          f"| {est['context_used_tokens']:,} tokens used "
          f"| {est['utilisation_pct']}% of window")
```

Output:

```
Standard             | ratio  5.0x | 4,500 tokens used  | 3.6% of window
Large parent         | ratio 15.0x | 9,000 tokens used  | 7.1% of window
More results         | ratio  5.3x | 4,000 tokens used  | 3.2% of window
Many small parents   | ratio  5.0x | 4,000 tokens used  | 3.2% of window
```

Modern LLMs have large context windows, so in most configurations context utilisation is low. The binding constraint is usually retrieval quality (too-small children don't embed well) and answer coherence (too-large parents dilute the relevant signal with surrounding noise).

---

## When parent-child chunking earns its complexity

Use it when your documents have a natural two-level structure that maps onto a retrieval-precision vs. generation-quality tradeoff. Technical documentation (paragraph-level children, section-level parents), medical literature (sentence-level children, abstract-level parents), and legal documents (clause-level children, article-level parents) all fit this pattern well.

Skip it when your documents are already short and homogeneous — FAQ entries, product descriptions, support tickets. If each document is already the right size to both embed and read, the extra plumbing adds complexity without benefit. Skip it also when you're using contextual chunking with well-sized chunks, since prepended context already gives the LLM what it needs to understand each chunk in isolation.