"""
test_pipeline.py — End-to-end pipeline test for AI File Search Assistant.

Indexes all files in Sample_files/ into:
    - SQLite  (metadata + extracted text)
    - FAISS   (semantic embeddings)

Run from the project root:
    python tests/test_pipeline.py
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction import extract_text
from app.database import DatabaseManager
from app.embeddings import EmbeddingManager
from app.search import FAISSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "Sample_files"

db = DatabaseManager()
em = EmbeddingManager()
fm = FAISSManager()

# ---------------------------------------------------------------------------
# Index all files
# ---------------------------------------------------------------------------

files = list(SAMPLE_DIR.iterdir())
print(f"\nFound {len(files)} files in {SAMPLE_DIR}\n")
print("=" * 60)

indexed = 0
skipped = 0

for file_path in sorted(files):
    if not file_path.is_file():
        continue

    print(f"\nProcessing : {file_path.name}")

    # Step 1 — Extract text
    content = extract_text(str(file_path))

    if not content:
        print(f"  ⚠  Skipped  — no text extracted (unsupported or empty)")
        skipped += 1
        continue

    print(f"  ✓  Extracted  — {len(content)} characters")

    # Step 2 — Store metadata in SQLite
    file_id = db.upsert_file(
        path=str(file_path),
        filename=file_path.name,
        extension=file_path.suffix.lower(),
        size=file_path.stat().st_size,
        modified_time=str(file_path.stat().st_mtime),
        content=content,
    )
    print(f"  ✓  DB upsert  — file_id={file_id}")

    # Step 3 — Generate embedding and store in FAISS
    embedding = em.generate_embedding(content)
    fm.add_embedding(file_id=file_id, embedding=embedding)
    print(f"  ✓  Embedded   — shape={embedding.shape}")

    indexed += 1

# ---------------------------------------------------------------------------
# Save FAISS index
# ---------------------------------------------------------------------------

fm.save()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"  Indexed : {indexed} files")
print(f"  Skipped : {skipped} files")
print(f"  DB total: {db.count_files()} records")
print(f"  FAISS   : {fm.get_total_vectors()} vectors")
print("=" * 60)

# ---------------------------------------------------------------------------
# Quick verification — fetch and preview each DB record
# ---------------------------------------------------------------------------

print("\nDB Records Preview:")
print("-" * 60)
for record in db.get_all_files():
    print(f"  [{record['id']}] {record['filename']} "
          f"({record['extension']}) — {record['size']} bytes")

# ---------------------------------------------------------------------------
# Uncomment to reset everything and start fresh
# ---------------------------------------------------------------------------
# db.clear_all()
# fm.reset()
# fm.save()