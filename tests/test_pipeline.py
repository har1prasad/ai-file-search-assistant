"""
test_pipeline.py — End-to-end pipeline test for AI File Search Assistant.

Indexes all files in Sample_files/ using the production FileIndexer class.
Verify chunks database extraction and FAISS vector matching.

Run from the project root:
    python tests/test_pipeline.py
"""

import sys
import logging

if sys.platform == "win32":
    import io
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.db_manager import DatabaseManager
from app.embeddings.embedding_manager import EmbeddingManager
from app.search.faiss_manager import FAISSManager
from app.indexing.file_indexer import FileIndexer
from app.search.search_engine import SearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "Sample_files"

db = DatabaseManager()
em = EmbeddingManager()
fm = FAISSManager()

indexer = FileIndexer(
    db_manager=db,
    embedding_manager=em,
    faiss_manager=fm,
)

search_engine = SearchEngine(
    embedding_manager=em,
    faiss_manager=fm,
    db_manager=db,
)

# Reset databases for a clean test run
print("Resetting databases for a clean run...")
db.clear_all()
fm.reset()
fm.save()

# ---------------------------------------------------------------------------
# Index all files
# ---------------------------------------------------------------------------

print(f"\nIndexing folder: {SAMPLE_DIR}")
print("=" * 60)

summary = indexer.index_folder(SAMPLE_DIR)

print("\n" + "=" * 60)
print("INDEXING PIPELINE SUMMARY:")
for key, value in summary.items():
    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

print("\nSQLite Database Files Preview:")
print("-" * 60)
for record in db.get_all_files():
    chunks = db.get_chunks_by_file_id(record['id'])
    print(f"  [{record['id']}] {record['filename']} ({record['extension']}) "
          f"— {record['size']} bytes, {len(chunks)} chunk(s)")

print("\nFAISS Index Statistics:")
print("-" * 60)
print(f"  Total Vectors: {fm.get_total_vectors()}")

# ---------------------------------------------------------------------------
# Run a test search query
# ---------------------------------------------------------------------------

test_query = "python programming resume or cover letter"
print(f"\nRunning test semantic search for: '{test_query}'")
print("-" * 60)
search_results = search_engine.search(test_query, top_k=3)

for idx, result in enumerate(search_results, start=1):
    print(f"  {idx}. {result['filename']} — Score: {result['similarity_score']:.4f}")
    if "matching_chunks" in result and result["matching_chunks"]:
        print(f"     Match: {result['matching_chunks'][0][:150]}...")