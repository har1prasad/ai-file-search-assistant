# AI File Search Assistant v0.1.0

AI File Search Assistant is a local desktop application that lets you search files on your computer using natural language.

Instead of remembering exact file names, you can search with queries like:

- "python notes"
- "cover letter"
- "fastapi tutorial pdf"

The application understands the meaning of your query using semantic search.

---

## Features

- Extracts text from PDF, DOCX, TXT, CSV, Markdown, and code files
- Stores file metadata in SQLite
- Generates embeddings using Sentence Transformers
- Performs fast semantic search using FAISS
- Works fully offline

---

## Project Structure

```text
app/
├── database/
├── embeddings/
├── extraction/
├── indexing/
├── search/
├── ui/
└── utils/

data/
tests/
main.py
requirements.txt
```