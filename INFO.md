# INFO.md — Technical Architecture & Complete Documentation
## AI-Powered File Search & Chat Workspace

---

## 1. Executive Summary

* **Project Name**: AI File Search Assistant
* **Domain**: Desktop Document Management, Semantic Information Retrieval, Retrieval-Augmented Generation (RAG)
* **Main Purpose**: Provide an interactive local file desktop search workspace that bridges traditional keyword search with semantic understanding and conversational AI.
* **Core Problem Solved**: Traditional desktop search (such as Windows Search or macOS Spotlight) is heavily constrained by exact string matches, failing to retrieve files when users search using synonyms, related concepts, or natural language queries. Furthermore, once a document is located, users must manually open and search through it, rather than directly interacting with its contents.
* **Target Users**: Information workers, developers, researchers, students, and system administrators managing large local text directories, code repositories, or reference files.
* **Key Capabilities**:
  * Multi-format textual extraction (plain text, markdown, source code, HTML, PDF, Word, PowerPoint, Excel, CSV, and images).
  * Automated OCR fallback for scanned PDFs and image files via local ONNX runtimes.
  * Hybrid search blending exact keyword matching (SQLite FTS5 BM25) and semantic vector search (SentenceTransformers + CPU FAISS).
  * Local RAG pipeline that chunks documents and retrieves relevant context segments to feed Gemini API.
  * Multi-turn chat session memory allowing continuous context-aware QA on selected files.
  * Adaptive dark-mode PySide6 desktop GUI built on a thread-safe model-view architecture.
* **Project Maturity Level**: **Production Beta** — Core indexing pipelines, databases, and vector stores are fully transactional and highly optimized (parallel threads, batched database connections, vector memory reconstruction). The application utilizes state-of-the-art models locally (all-MiniLM-L6-v2) and cloud LLMs for generation (Gemini 2.5 Flash).

---

## 2. Problem Statement

### The Shortcomings of Traditional File Search
Local system file discovery is traditionally bound to filenames or basic grep-like pattern matching. When managing directories containing source code, PDFs, spreadsheets, and scanned documents, users struggle with:
1. **Semantic Gaps**: Searching for "compensation details" fails to match documents containing only "salary structure" or "bonus scheme."
2. **Scanned Documents**: Scanned papers or screenshots of text are stored as raw binaries (images or vector PDFs with empty text layers) and are completely invisible to indexers.
3. **Format Heterogeneity**: Text extraction across PDFs, Docx, Pptx, and spreadsheets requires varying parsers, encodings, and error handling, making unified grepping fragile.
4. **Information Overload**: Finding a document is only half the battle. Users must open and read extensive files to retrieve a single answer.

### The AI/NLP Paradigm Shift
Integrating local Sentence Transformers allows the system to map documents into a high-dimensional vector space where semantic similarity corresponds to geometric proximity. Exact vocabulary matches become secondary to conceptual alignment. Coupling this retrieval engine with a conversational Large Language Model (Gemini 2.5 Flash) enables users to ask clarifying questions about a document immediately after finding it, transforming static search results into interactive knowledge hubs.

---

## 3. System Overview

The **AI File Search Assistant** operates as a single-window desktop dashboard. 

```
                                 +-----------------------+
                                 |  User Search Request  |
                                 +-----------+-----------+
                                             |
                                             v
                              +--------------+--------------+
                              |    TitleBar Search Input    |
                              +--------------+--------------+
                                             |
                                             v
                              +--------------+--------------+
                              |        SearchEngine         |
                              +------+---------------+------+
                                     |               |
                (Semantic Query)     v               v     (Keyword Query)
                            +--------+---+       +---+--------+
                            | FAISS Index|       | SQLite FTS5|
                            +--------+---+       +---+--------+
                                     |               |
                  (Cosine Distance)  v               v     (BM25 Score)
                            +--------+---+       +---+--------+
                            | Vector Matches     | Text Matches
                            +--------+---+       +---+--------+
                                     |               |
                                     +-------+-------+
                                             | (Hybrid Blending)
                                             v
                                  +----------+----------+
                                  | Group & Score Files |
                                  +----------+----------+
                                             |
                                             v
                                 +-----------+-----------+
                                 |   PySide6 QListWidget |
                                 +-----------+-----------+
                                             | (Select File)
                                             v
                                 +-----------+-----------+
                                 | Preview & Chat Stack  |
                                 +-----------------------+
```

### End-to-End Lifecycle:
1. **Folder Registration**: The user selects a folder. The `FileIndexer` recursively scans paths, filters out files larger than 50MB, checks against supported extensions, and compares modification timestamps (`mtime`) with records in a local SQLite file to detect changes.
2. **Orphan Cleanup**: If files were deleted on disk since the last index pass, their records are pruned from the SQLite database, and their associated chunk vectors are purged from the FAISS index by ID.
3. **Parallel Text Extraction**: Files requiring updates are processed in parallel using a CPU-bound thread pool. If a PDF contains little to no embedded text, the system automatically renders pages to images and performs OCR.
4. **SQLite Transactional Batching**: Extracted plain text is mapped to a parent file record. The document is split into overlapping chunks, written to the `chunks` table, and registered in a virtual SQLite FTS5 table (`chunks_fts`) inside a single transactional block.
5. **Batch Vector Generation & FAISS Insertion**: Chunk texts are embedded in a single batch using a local SentenceTransformer model. The resulting 384-dimensional unit-normalized float32 vectors are added to a CPU-bound FAISS flat index matching the primary key chunk IDs.
6. **Query Processing**: When a search is triggered, the search term is processed by both FAISS (semantic matching) and FTS5 (BM25 keyword matching). Results are combined, grouped by parent files, sorted by similarity, and rendered as rich cards.
7. **Document Q&A (RAG)**: Clicking a file displays its contents. Asking a question prompts the `ChatEngine` to pull the most relevant context segments using a local vector cosine similarity lookup. The question and matched segments are passed to Gemini via a multi-turn chat session.

---

## 4. High-Level Architecture

The system follows a modular, decoupled architecture where components communicate via standard classes, direct injection, and PySide6 signals.

```
+---------------------------------------------------------------------------------+
|                                 USER INTERFACE                                  |
|  [main_window.py] (MainWindow, TitleBar, ResultCard)                            |
+------------------------+-------------------------------+------------------------+
                         | Signals                               |
                         v                               v
+------------------------+-------+             +---------+------------------------+
|       INDEXING ENGINE          |             |          SEARCH ENGINE           |
|  [file_indexer.py]             |             |  [search_engine.py]              |
+------------------------+-------+             +---------+------------------------+
                         |                               |
                         +---------------+---------------+
                                         |
                                         v
+----------------------------------------+----------------------------------------+
|                               CORE SERVICES                                     |
|  +------------------------+  +-------------------+  +------------------------+  |
|  |     EXTRACTOR          |  | EMBEDDING MANAGER |  |     FAISS MANAGER      |  |
|  |  [extractor.py]        |  | [embedding_man.py]|  |  [faiss_manager.py]    |  |
|  +------------------------+  +-------------------+  +------------------------+  |
|  |   DATABASE MANAGER     |                         |      CHAT ENGINE       |  |
|  |   [db_manager.py]      |                         |    [chat_engine.py]    |  |
|  +------------------------+                         +------------------------+  |
+---------------------------------------------------------------------------------+
```

### Subsystems Breakdown:

#### 1. UI Layer (`app/ui/main_window.py`)
* **Purpose**: Responsive, frameless desktop container displaying files, previews, and chat blocks.
* **Inputs**: User text queries, mouse selection events, indexing folder paths.
* **Outputs**: Thread spawns, filtered file listings, rendered markdown chat bubbles, plain text document panels.
* **Dependencies**: `PySide6.QtWidgets`, `PySide6.QtCore`, `PySide6.QtGui`, `qdarktheme`.
* **Internal Logic**: Spawns concurrent `QThread` workers for indexing and search to preserve 60 FPS UI responsiveness. Connects filter tag buttons and suggestion cards to action slots.

#### 2. File Indexing Orchestrator (`app/indexing/file_indexer.py`)
* **Purpose**: Scans directories, identifies modified/deleted files, batch-updates SQLite, generates vectors, and synchronizes FAISS.
* **Inputs**: Folder path.
* **Outputs**: Synchronized index database, indexing stats dictionary.
* **Dependencies**: `DatabaseManager`, `EmbeddingManager`, `FAISSManager`, `concurrent.futures`.
* **Internal Logic**: Runs thread pool extractions, groups chunking, commits SQLite records in a transaction block, feeds chunk text arrays to the embedding manager, and inserts generated embeddings into FAISS.

#### 3. Document Extractor Registry (`app/extraction/extractor.py`)
* **Purpose**: Extract text content and metadata from multiple formats.
* **Inputs**: Local file path.
* **Outputs**: Dataclass `ExtractionResult` containing truncated normalized content, `FileMetadata`, and character-level overlapping chunks.
* **Dependencies**: `fitz` (PyMuPDF), `docx`, `pptx`, `openpyxl`, `bs4`, `rapidocr_onnxruntime`.
* **Internal Logic**: Uses a registry pattern resolving extensions to specific class extractors (e.g. `PDFExtractor`). Truncates text at 50,000 characters to prevent huge token payloads.

#### 4. Embedding Manager (`app/embeddings/embedding_manager.py`)
* **Purpose**: Convert clean text strings into high-dimensional numerical vectors.
* **Inputs**: Plain text string or list of text strings.
* **Outputs**: Float32 NumPy arrays of shape `(n, 384)`.
* **Dependencies**: `sentence_transformers`, `torch`, `numpy`.
* **Internal Logic**: Implements a lazy-loaded thread-safe singleton wrapper for the `all-MiniLM-L6-v2` transformer model. Checks `torch.cuda.is_available()` to offload computation to GPUs when possible. Normalizes output vectors to unit length.

#### 5. SQLite Metadata & FTS Store (`app/database/db_manager.py`)
* **Purpose**: Persist document structures, path references, modification times, chunks, and FTS virtual indexes.
* **Inputs**: Upsert properties, SQL queries.
* **Outputs**: SQLite rows resolved as Python dictionaries, row IDs, file/chunk counts, keyword matching lists.
* **Dependencies**: `sqlite3`.
* **Internal Logic**: Configures foreign keys with cascade deletions, enables FTS5 porter-unicode tokenization, and supports external connection reuse for database transactions.

#### 6. FAISS Vector Database (`app/search/faiss_manager.py`)
* **Purpose**: Persist and query float32 unit-normalized vector embeddings.
* **Inputs**: Chunk IDs, search queries.
* **Outputs**: List of matching `(chunk_id, cosine_score)` tuples.
* **Dependencies**: `faiss`, `numpy`.
* **Internal Logic**: Wraps `faiss.IndexFlatIP` (flat inner product) inside `faiss.IndexIDMap` to support vector addition/removal by custom SQLite chunk primary keys. Since vectors are pre-normalized, the inner product matches cosine similarity.

#### 7. Search & Retrieval Engine (`app/search/search_engine.py`)
* **Purpose**: Perform hybrid matching by combining semantic results and FTS keyword matches.
* **Inputs**: Query string.
* **Outputs**: Blended list of parent document records with score values.
* **Dependencies**: `DatabaseManager`, `EmbeddingManager`, `FAISSManager`.
* **Internal Logic**: Queries both FAISS and SQLite FTS5, normalizes FTS5 BM25 scores, blends them using weights (`0.7` vector / `0.3` keyword), resolves chunk matches to parent file records, and returns the sorted collection.

#### 8. Conversational Chat Engine (`app/chat/chat_engine.py`)
* **Purpose**: Execute local RAG retrieval, construct prompt instructions, manage conversational history, and interface with Gemini.
* **Inputs**: Document content, user question, file ID (optional).
* **Outputs**: Markdown-formatted response containing user-AI history.
* **Dependencies**: `google-genai`, `numpy`, `hashlib`, `dotenv`.
* **Internal Logic**: Computes prompt-to-chunk cosine similarities, retrieves top 5 relevant document segments as context, maps conversation sessions using MD5 document hashes, and queries `gemini-2.5-flash` with system instructions restricting responses to the context.

---

## 5. Complete Folder Structure Analysis

```text
ai-file-search-assistant/
├── main.py                     # Application entry point & QApp loop.
├── requirements.txt            # System dependencies manifest.
├── performance_opt_plan.md     # Architectural optimization proposal.
├── INFO.md                     # Comprehensive technical documentation (this file).
├── neww.png                    # Mockup design reference image.
├── rough ui.png                # Layout wireframe reference image.
├── app/
│   ├── __init__.py             # Exposes submodules package namespaces.
│   ├── chat/
│   │   ├── __init__.py         # Package initialization.
│   │   └── chat_engine.py      # LLM chat session and local RAG search.
│   ├── database/
│   │   ├── __init__.py         # Package initialization.
│   │   └── db_manager.py       # SQLite database connectivity & CRUD.
│   ├── embeddings/
│   │   ├── __init__.py         # Package initialization.
│   │   └── embedding_manager.py# SentenceTransformers model lazy loader.
│   ├── extraction/
│   │   ├── __init__.py         # Package initialization.
│   │   └── extractor.py        # Text & metadata extraction classes.
│   ├── indexing/
│   │   ├── __init__.py         # Package initialization.
│   │   └── file_indexer.py     # Parallel directory scanner.
│   ├── search/
│   │   ├── __init__.py         # Package initialization.
│   │   ├── faiss_manager.py    # FAISS binary vector wrapper.
│   │   └── search_engine.py    # Hybrid search scoring & rank blending.
│   ├── ui/
│   │   ├── __init__.py         # Package initialization.
│   │   ├── ai_assistant.png    # Empty state speech bubble illustration asset.
│   │   ├── main_window.py      # PySide6 main window GUI layout.
│   │   └── no_file_selected.png# Empty state magnifying glass illustration asset.
│   └── utils/
│       └── __init__.py         # Reserved for shared utility functions.
├── data/
│   ├── faiss.index             # Persisted FAISS binary vector file.
│   └── metadata.db             # Persisted SQLite relational database file.
├── models/
│   └── all-MiniLM-L6-v2/       # Local SentenceTransformer directory.
├── tests/
│   └── test_pipeline.py        # End-to-end command line test script.
└── Sample_files/               # Test documents directory.
```

### Critical Files Deep Dive:

#### `main.py`
* **Purpose**: Entry point. Sets up the environment, initializes the PySide6 `QApplication`, configures window settings, instantiates `MainWindow`, and starts the main event loop.
* **Dependent Files**: None.
* **Critical Classes/Functions**:
  * `main()`: Resolves sys arguments, applies the application name, creates the layout frame, and executes `app.exec()`.

#### `app/ui/main_window.py`
* **Purpose**: UI coordinator. Implements custom title bar dragging, dynamic theme switching, result card rendering, list filter mapping, stacked widget controls for empty states, and handles asynchronous worker thread creation.
* **Dependent Files**: `main.py`.
* **Critical Classes/Functions**:
  * `MainWindow`: Main GUI controller.
  * `TitleBar`: Custom frameless window header bar incorporating window actions and search widgets.
  * `ResultCard`: Custom widget for items in the search list.
  * `IndexWorker` / `SearchWorker`: QThread classes managing backend processes.

#### `app/indexing/file_indexer.py`
* **Purpose**: Manages file status transitions. Matches physical files on disk with DB records. Runs thread pool processes for extraction and commits chunk changes.
* **Dependent Files**: `app/ui/main_window.py`, `tests/test_pipeline.py`.
* **Critical Classes/Functions**:
  * `FileIndexer`: Coordination engine class.
  * `index_folder()`: Cleans deleted items, identifies files needing updates, extracts text, writes SQLite records, and saves FAISS indices.

#### `app/database/db_manager.py`
* **Purpose**: Data layer. Provides SQL connection pooling, table schemas, FTS5 matches, chunk reads, and cascade deletions.
* **Dependent Files**: `app/indexing/file_indexer.py`, `app/search/search_engine.py`, `app/chat/chat_engine.py`, `app/ui/main_window.py`.
* **Critical Classes/Functions**:
  * `DatabaseManager`: SQLite operations coordinator.
  * `upsert_file()`: Inserts/updates metadata.
  * `insert_chunk()`: Adds a chunk to both `chunks` and `chunks_fts` virtual tables.
  * `search_fts()`: Performs keyword query operations using `MATCH` and BM25 scoring.

#### `app/search/faiss_manager.py`
* **Purpose**: High-speed vector index. Adds embeddings, deletes by ID list, reconstructs floats, and executes inner product searches.
* **Dependent Files**: `app/indexing/file_indexer.py`, `app/search/search_engine.py`, `app/chat/chat_engine.py`, `app/ui/main_window.py`.
* **Critical Classes/Functions**:
  * `FAISSManager`: FAISS interface wrapper.
  * `add_embeddings()`: Registers vectors with matching chunk IDs.
  * `search()`: Executes float matrix multiplication to locate matches.

---

## 6. Tech Stack Analysis

| Technology | Sub-component | Purpose | Rationale | Alternatives | Tradeoffs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Python** | Runtime | Core programming language | Vast ecosystem for AI, ML, NLP libraries, and GUI toolkits. | Go, C++ | slower runtime execution than C++, but highly acceptable due to binding C/C++ libraries. |
| **PySide6** | GUI Framework | Qt6 GUI Wrapper | Qt is the industry standard for cross-platform desktop GUIs. PySide6 provides native bindings. | PyQt6, Tkinter | Strict licensing compared to Tkinter, larger binary footprint. |
| **PyQtDarkTheme**| Styling | Styling layout skin | Provides clean CSS theme styles and Light/Dark options on startup. | Custom QSS | Simple setup, but has styling limits on custom widgets. |
| **Sentence-Transformers** | NLP Embeddings | Local vector generation | Runs completely offline. `all-MiniLM-L6-v2` is small, fast, and generates high-quality semantic representations. | OpenAI Embeddings | Requires local RAM/CPU, does not scale to long sequences. |
| **FAISS (CPU)** | Vector Search | Local vector index | Fast similarity search on multi-dimensional matrices. | ChromaDB, Qdrant | In-memory index, requires rebuilding if data changes. |
| **SQLite (FTS5)** | Database | Metadata & Keyword search | Native database requiring zero configuration. FTS5 provides fast keyword matching. | PostgreSQL, MySQL | Lacks support for concurrent write connections, but ideal for desktop apps. |
| **PyMuPDF** | Extractor | PDF text extraction | Fast PDF parser with support for page rendering. | PyPDF2, pdfplumber | C-bound (fitz), can be difficult to build on some systems. |
| **RapidOCR** | Extractor | Optical Character Recognition | Lightweight CPU-optimized OCR using ONNX. | Tesseract OCR | Relies on external model files, slower on large scanned docs. |
| **Google GenAI SDK** | RAG LLM | Generative Q&A | Interface for Gemini API models like `gemini-2.5-flash`. | OpenAI, Anthropic | Requires an internet connection and API key. |

---

## 7. File Processing Pipeline

The file processing pipeline executes in the background when indexing is triggered:

```
[Local File]
     |
     v
(Filters: Size > 50MB?, Unsupported extension?)
     |
     v
(Compare DB mtime to check if update is needed)
     |
     v
[ThreadPoolExecutor Text Extraction]
     | (PDF, Word, Code, Image...)
     v
[Truncation & Normalization]
     | (Limit to 50k characters, collapse spaces)
     v
[Character-Level Chunking]
     | (Size: 800, Overlap: 150)
     v
[Transactional DB Writing]
     | (Write to files, chunks, & chunks_fts)
     v
[Batch Semantic Encoding]
     | (EmbeddingManager SentenceTransformer)
     v
[FAISS Sync & Persistence]
     | (add_embeddings, save faiss.index)
     v
[Status Message Output]
```

### Format Extraction Strategies:
* **Plain Text / Source Code / Markdown**: Direct read via UTF-8, ignoring decoding errors.
* **HTML**: Uses BeautifulSoup to strip `<script>`, `<style>`, and `<noscript>` blocks, extracting visible text.
* **Word (`.docx`)**: Iterates through document paragraphs and merges text.
* **PowerPoint (`.pptx`)**: Collects shape text boxes, slide titles, tables, and speaker notes.
* **Excel (`.xlsx`)**: Loads workbook in read-only mode, extracts cell text row-by-row with tab separation.
* **CSV**: Parses rows and compiles tab-separated lines.
* **PDF (`.pdf`)**: Parses pages via PyMuPDF. If page text length is below 100 characters, renders page to PNG and runs RapidOCR.
* **Images**: Reads raw bytes directly into RapidOCR. Falls back to a descriptive filename string if no text is found.

---

## 8. Database Design

The local relational storage is managed by SQLite via `data/metadata.db`.

### Entity Relationship Diagram (ERD):

```
+------------------+          +---------------------+          +------------------+
|      files       |          |       chunks        |          |    chunks_fts    |
+------------------+          +---------------------+          +------------------+
| id (PK, AutoInc) |<--------| file_id (FK, Cascade)|          | rowid (PK, FK)   |
| path (Unique)    |          | id (PK, AutoInc)    |<---------| content (Text)   |
| filename         |          | chunk_index (Int)   |          +------------------+
| extension        |          | content (Text)      |
| size             |          +---------------------+
| modified_time    |
| content          |
| indexed_at       |
+------------------+
```

### Schema DDL Statements:

```sql
-- Files Table
CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT    UNIQUE NOT NULL,
    filename      TEXT    NOT NULL,
    extension     TEXT,
    size          INTEGER,
    modified_time TEXT,
    content       TEXT,
    indexed_at    TEXT    DEFAULT CURRENT_TIMESTAMP
);

-- Chunks Table
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content       TEXT    NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

-- Full-Text-Search Virtual Table
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    tokenize="porter unicode61"
);
```

### Indexing and Constraints:
* `files.path` is defined as `UNIQUE`. Conflicts during indexing trigger an `ON CONFLICT(path) DO UPDATE SET` upsert behavior.
* `chunks.file_id` is a Foreign Key referencing `files.id` with `ON DELETE CASCADE`. If a file is deleted from the database, all its chunks are deleted automatically.
* `chunks_fts` uses the Porter stemmer tokenizer (`tokenize="porter unicode61"`), allowing keyword searches to match root words (e.g. "indexing" matches "index").

---

## 9. Data Flow Diagrams (DFD)

### Level 0: Context Diagram

```
+--------+                 Search & Chat Queries                 +-----------------+
|        | ----------------------------------------------------> |                 |
|  User  |                                                       | AI File Search  |
|        | <---------------------------------------------------- |    Assistant    |
+--------+                 Files & AI Answers                    +--------+--------+
                                                                          |
                                                      Gemini API Requests | Responses
                                                                          v
                                                                 +-----------------+
                                                                 |   External LLM  |
                                                                 |   (Gemini API)  |
                                                                 +-----------------+
```

### Level 1: Subsystem Interaction

```
                       +-------------------------------------------------+
                       |                  UI Layer                       |
                       +---------+----------------------------+----------+
                                 |                            |
                         Search  |                            | Chat
                         Query   v                            v Q&A
                 +---------------+---------------+   +--------+------------------+
                 |         SearchEngine          |   |       ChatEngine          |
                 +---+-----------------------+---+   +--------+------------------+
                     |                       |                |
                     | Semantic              | Keyword        | Cosine Similarity
                     v                       v                v Context Lookup
              +------+------+         +------+------+  +------+------+
              | FAISS Index |         | SQLite FTS5 |  | SQLite Chunks|
              +-------------+         +-------------+  +-------------+
```

---

## 10. Search Engine Internals

The system uses a custom hybrid retrieval model, blending keyword matching and vector matching:

```
[Query Input] 
     |
     v
(Tokenize & Clean) 
     |
     +-----------------------------------+
     |                                   |
     v (Semantic Branch)                 v (Keyword Branch)
[EmbeddingManager Vector Gen]       [Format terms for FTS5 (term1 OR term2)]
     |                                   |
     v                                   v
[FAISS FlatIP Cosine Search]        [SQLite chunks_fts Query]
     | (Yields chunk_ids & scores)       | (Yields chunk_ids & BM25 scores)
     v                                   v
[Clamp Cosine Scores to [0,1]]      [Normalize BM25 scores to [0,1]]
     |                                   |
     +-----------------+-----------------+
                       |
                       v
            [Score Blending Logic]
        If Semantic and Keyword match: 
            score = 0.7 * semantic + 0.3 * keyword
        If Semantic only match: 
            score = semantic
        If Keyword only match: 
            score = 0.5 * keyword
                       |
                       v
            [File-Level Grouping]
        Map chunk matches to parent files.
        Take MAX(score) and accumulate chunk texts.
                       |
                       v
         [Sort & Return Top K Files]
```

### Blending Logic and Weight Rationale:
* **Vector Cosine Similarity** (`semantic`) is given a high weight of `0.7` because it matches search intent and related concepts, even if vocabulary differs.
* **FTS5 BM25** (`keyword`) is given a weight of `0.3` to act as a precision booster for exact matches like filenames, variable names, or acronyms.
* Keyword-only matches are slightly discounted (`0.5 * keyword`) to prevent minor matching keywords from ranking above strong semantic matches.

---

## 11. AI Chat Pipeline

The conversational pipeline implements a local RAG model before calling the Gemini API:

```
                          [User Question + Selected File]
                                         |
                                         v
                      [Check Context Length of File Content]
                         Is length <= 3000 characters?
                          /                         \
                    Yes  /                           \ No
                        v                             v
           [Use Full Text as Context]        [Local RAG Context Retrieval]
                                              1. Retrieve file chunks from SQLite
                                              2. Retrieve FAISS embeddings by ID
                                              3. Calculate cosine similarity
                                              4. Select top 5 relevant chunks
                                              5. Merge chunks as context
                                                       |
                                                       v
                                            [Context Construction]
                                                       |
                                                       v
                                            [Prompt Construction]
                                                       |
                                                       v
                                            [Gemini API Request]
                                                       |
                                                       v
                                            [Response Formatting]
```

### Prompt Construction Template:
The `ChatEngine` formats context payloads with system instructions to prevent hallucinations:
```text
SYSTEM INSTRUCTION:
You are an AI assistant helping a user understand a file.
Answer questions based ONLY on the provided context.
If the answer cannot be determined from the context, state that clearly.
Be concise and clear in your responses.

CONTEXT:
{retrieved_context_segments}

QUESTION:
{user_question}
```

### Conversational Memory:
The application caches chat sessions by hashing the selected file content using MD5 (`hashlib.md5`). If a file is modified, a new hash is generated, starting a fresh session. History logs are formatted in markdown blocks:
```text
### 💬 User
{user_question}

---

### 💬 AI
{ai_response}
```

---

## 12. Gemini API Integration

The application integrates with Gemini using the new `google-genai` Python SDK.

### Configuration & Pricing Details:
* **Target SDK Version**: `google-genai>=2.0`
* **Authentication**: Environment variable `GEMINI_API_KEY` (loaded from `.env` at the root).
* **AI Engine**: `gemini-2.5-flash` for document Q&A, and `gemini-1.5-flash` for general queries.
* **Token/Cost Optimization**: Text contents are truncated to a maximum of 50,000 characters before chunking. Implementing RAG means the application only sends the top 5 matching context chunks (around 3,000–4,000 characters) instead of full 50,000-character files, reducing token usage.

---

## 13. Runtime Execution Flow

```
[Main Application Start] -> [sys.argv setup & QApplication init]
     |
     v
[MainWindow instantiation]
     |
     +-> [Database Connection & Table Verification]
     |   - Opens data/metadata.db
     |   - Creates files, chunks, & chunks_fts tables
     |
     +-> [FAISS Vector Index Load]
     |   - Loads data/faiss.index from disk if present
     |
     +-> [Asynchronous Thread Spawning]
     |   - Pre-loads SentenceTransformer model in background thread
     |
     +-> [load_all_files()]
     |   - Pulls indexed files from DB, populates UI QListWidget
     |
     +-> [GUI Displayed (Frameless Window)]
     |
[User Search Action] -> [SearchWorker QThread Spawned]
     |
     +-> [Query processing via SearchEngine]
     |   - Run Hybrid vector + keyword retrieval
     |   - Blends scores and groups by file
     |
     +-> [UI Updates via Finished Signal]
     |   - Refreshes QListWidget result cards
     |
[User Selects File] -> [Preview Stack Updated]
     |
     +-> [Toggles stacked widgets, displays text content]
     |
[User Chat Action] -> [Gemini API Query]
     |
     +-> [Local RAG chunk similarity calculations]
     |   - Matches question embedding with document chunk embeddings
     |   - Prepares prompt payload
     |
     +-> [Calls chat.send_message()]
     v
[Application Shutdown] -> [Closes thread pools, saves files]
```

---

## 14. Module-by-Module Breakdown

### 1. `app/database/db_manager.py:DatabaseManager`
* **Responsibilities**: Connection manager for the SQLite database. Handles all CRUD operations for document metadata and text chunks.
* **Methods**:
  * `connect()`: Returns a connection with `foreign_keys = ON` and `row_factory = sqlite3.Row`.
  * `upsert_file()`: Inserts or updates a file record by path.
  * `insert_chunk()`: Inserts a chunk text record and registers it in the `chunks_fts` virtual table.
  * `search_fts()`: Performs full-text search queries using Porter stemmer mapping.

### 2. `app/embeddings/embedding_manager.py:EmbeddingManager`
* **Responsibilities**: Thread-safe manager for generating vector embeddings locally.
* **Methods**:
  * `_ensure_model_loaded()`: Lazily loads the `all-MiniLM-L6-v2` transformer model to CPU or GPU.
  * `generate_embedding()`: Encodes a single query string.
  * `generate_embeddings()`: Encodes a batch of chunks.

### 3. `app/search/faiss_manager.py:FAISSManager`
* **Responsibilities**: Vector database manager. Interfaces with the FAISS index to add, search, and delete embeddings.
* **Methods**:
  * `add_embeddings()`: Inserts float32 vectors mapped to chunk IDs.
  * `remove_ids()`: Deletes vectors from the index.
  * `get_embeddings()`: Reconstructs vector dimensions by ID.
  * `search()`: Executes vector similarity lookups.

### 4. `app/indexing/file_indexer.py:FileIndexer`
* **Responsibilities**: Indexing manager. Scans local folders, detects changes, manages parallel extraction, and syncs indexes.
* **Methods**:
  * `index_folder()`: High-level directory indexing pipeline.
  * `_cleanup_deleted_files()`: Prunes missing files from SQLite and FAISS.

### 5. `app/chat/chat_engine.py:ChatEngine`
* **Responsibilities**: Conversational AI engine. Handles context selection, prompt formatting, and chat history.
* **Methods**:
  * `ask_about_file()`: Runs local RAG similarity math, manages session cache, and queries Gemini.
  * `_get_context()`: Selects top matching chunks for context prompts.

---

## 15. Algorithms & Core Logic

### 1. Hybrid Ranking Score Blending
The search engine blends semantic vector search scores and FTS5 BM25 search scores:
$$\text{Score} = \begin{cases} 
0.7 \cdot S + 0.3 \cdot K & \text{if } S > 0 \text{ and } K > 0 \\ 
S & \text{if } S > 0 \text{ and } K = 0 \\ 
0.5 \cdot K & \text{if } S = 0 \text{ and } K > 0 
\end{cases}$$
Where $S$ is the clamped vector Cosine Similarity score, and $K$ is the normalized keyword BM25 score.

### 2. Text Chunking Algorithm
The text chunker splits files into overlapping segments while preserving sentences:
```python
# Pseudocode representation of text chunking
def chunk_text(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if end < len(text):
            boundary = chunk.rfind(". ")  # Try sentence end
            if boundary == -1 or boundary < size // 2:
                boundary = chunk.rfind(" ")  # Fall back to word boundary
            if boundary != -1:
                chunk = chunk[:boundary + 1]
        chunks.append(chunk.strip())
        start += max(1, len(chunk) - overlap)
    return chunks
```
**Complexity**: $O(N)$ where $N$ is the number of characters in the file.

---

## 16. Configuration & Environment

The application configuration is managed via a root `.env` file and local folder hierarchies.

### Keys & Environment Variables:
* `GEMINI_API_KEY`: Required string to authenticate GenAI requests. If missing, `ChatEngine` raises a `ValueError`.
* Database File: Defaults to `data/metadata.db`.
* FAISS Index File: Defaults to `data/faiss.index`.
* Local Model Store: Defaults to `models/all-MiniLM-L6-v2`. If this folder is missing, the application downloads the model from Hugging Face on first run.

---

## 17. Dependency Analysis

| Library | Type | Purpose | Impact if Missing |
| :--- | :--- | :--- | :--- |
| **PySide6** | Mandatory | Application GUI library | **System Crash**: App cannot start. |
| **sentence-transformers** | Mandatory | Local vector generation | **System Crash**: Vector generation fails. |
| **faiss-cpu** | Mandatory | High-speed vector index | **System Crash**: Vector database fails. |
| **google-genai** | Mandatory | Gemini API client | **System Crash**: Chat engine cannot query models. |
| **PyMuPDF (fitz)** | Mandatory | PDF text extraction | **System Crash**: Cannot parse PDF files. |
| **python-docx** | Optional | Word document parser | Word documents are skipped during indexing. |
| **python-pptx** | Optional | PowerPoint document parser | PowerPoint slides are skipped during indexing. |
| **openpyxl** | Optional | Excel spreadsheet parser | Excel spreadsheets are skipped during indexing. |
| **beautifulsoup4** | Optional | HTML text parser | Returns raw HTML strings instead of clean text. |
| **rapidocr-onnxruntime**| Optional | Optical Character Recognition | Scanned PDFs and image text cannot be indexed. |

---

## 18. Error Handling

* **Missing Environment Variables**: Handled during initialization. If `GEMINI_API_KEY` is missing, `ChatEngine` raises a warning and falls back to a warning message in the UI instead of crashing.
* **Corrupted Files**: If an office document is corrupted, extractors catch the exception, log it, and return an empty string. The indexing engine registers this as a skip or failure count instead of crashing the process.
* **Model Download Failures**: If the local model is missing and there is no internet connection, `EmbeddingManager` raises an exception that is caught by `MainWindow`, which shows a warning box.
* **Database Locks**: SQLite database connections use `with db.connect() as conn` to ensure locks are released automatically. The application uses transactional blocks (`BEGIN TRANSACTION` and `COMMIT`) to prevent database corruption during crashes.

---

## 19. Security Analysis

* **API Keys**: Saved in a local `.env` file. Users must ensure this file is not committed to public repositories.
* **File Access Risks**: The application runs with user privileges, meaning it can only access folders and files the user has permission to read.
* **Local Processing Privacy**: Document text extraction, chunking, database storage, and vector generation happen entirely offline on the user's local machine, protecting data privacy.
* **External LLM Exposure**: Asking a question sends the top 5 matching context chunks to the Gemini API. Users should avoid querying files containing highly confidential or regulated data if they do not want segments processed by cloud models.

---

## 20. Performance Analysis

* **CPU-Intensive Tasks**: Text extraction, OCR, and vector generation are CPU-bound. The application resolves this by running text extraction in parallel thread pools and pre-loading models in the background to prevent UI freezes.
* **Memory Footprint**: Loading `all-MiniLM-L6-v2` consumes ~450MB of RAM. The FAISS flat index operates entirely in RAM. For 10,000 document chunks, the FAISS memory usage is under 15MB, making it highly efficient for standard desktop systems.
* **Disk I/O**: Previously, connections were opened and closed for every file. Implementing transactional batching reduced indexing I/O overhead significantly.

---

## 21. Current Limitations

* **Single-user Database**: SQLite does not support concurrent write connections, meaning only one instance of the app can index files at a time.
* **Context Window Boundaries**: While local RAG reduces token payload, files with complex tables or long-range context dependencies might lose meaning when chunked into isolated 800-character segments.
* **Frameless Snap Layouts**: The custom title bar does not support OS native snap layout actions (such as Windows Aero Snap) out of the box.

---

## 22. Improvement Recommendations

### Short-Term:
1. **Dynamic Chunk Sizing**: Automatically adjust chunk sizes based on file types (e.g. larger chunks for source code files to preserve function boundaries).
2. **Metadata Filtering**: Expose filters in the UI to allow searching only within specific folders or modified date ranges.

### Long-Term:
1. **Multi-Model Support**: Allow users to run local LLMs (e.g. LLaMA via Ollama) for a completely offline, private RAG pipeline.
2. **Graph-based RAG**: Build a semantic relationship graph between files to support complex queries across multiple documents.

---

## 23. Setup & Usage Guide

### Prerequisites:
* Python 3.9 to 3.11
* pip (Python package manager)

### Installation:
1. Clone the project repository and navigate to the root directory.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your Gemini API key:
   ```text
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application:
Launch the desktop application using the virtual environment:
```powershell
.venv\Scripts\python.exe main.py
```

### Running Tests:
Run the end-to-end command-line test script to verify indexing and retrieval:
```powershell
.venv\Scripts\python.exe tests/test_pipeline.py
```

---

## 24. Glossary

* **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors.
* **FTS5 (Full-Text Search 5)**: An SQLite virtual table module that allows users to perform full-text searches on database content.
* **RAG (Retrieval-Augmented Generation)**: A technique that optimizes LLM output by querying an external authoritative knowledge base before generating a response.
* **BM25**: A ranking function used by search engines to estimate the relevance of documents to a given search query.
* **Cosine Similarity**: A metric used to measure how similar two vectors are, calculated by the cosine of the angle between them.
* **OCR (Optical Character Recognition)**: The electronic conversion of images of typed, handwritten, or printed text into machine-encoded text.

---

## 25. Final Assessment

The **AI File Search Assistant** is a well-engineered local RAG application. It features a clean separation of concerns, robust multi-threaded UI safety, and transactional database batching. 

Implementing hybrid search and local RAG makes the system highly efficient, minimizing API cost while maintaining semantic retrieval precision. The code quality is excellent, with thread-safe singleton patterns, lazy loading, and error fallbacks. The application is highly maintainable, stable, and ready for production deployment.
