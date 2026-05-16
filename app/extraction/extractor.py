"""
extractor.py — Text extraction module for AI File Search Assistant.

Provides a single public entry point `extract_text(file_path)` that routes
each supported file type to a dedicated private extractor, normalises the
result, and enforces a configurable length ceiling.

Supported types:
    Plain text  : .txt
    PDF         : .pdf  (via PyMuPDF / fitz)
    Word        : .docx (via python-docx)
    Spreadsheet : .csv  (built-in csv module)
    Code        : .py .js .ts .java .c .cpp .h .hpp .cs .go .rs .php .rb .swift .kt
    Markup      : .md .json .xml .html
    Image       : .jpg .jpeg .png .bmp .gif .webp  (metadata only)

Usage:
    from app.extraction.extractor import extract_text
    text = extract_text("/path/to/file.pdf")
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path


# Constants

MAX_TEXT_LENGTH: int = 50_000

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".json", ".xml", ".html"}
)

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".js", ".ts", ".java", ".c", ".cpp",
        ".h", ".hpp", ".cs", ".go", ".rs", ".php",
        ".rb", ".swift", ".kt",
    }
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
)


# Logging

logger = logging.getLogger(__name__)


# Public API

def extract_text(file_path: str) -> str:
    """Extract plain text from a file and return it as a clean string.

    Routes the file to the appropriate extractor based on its extension,
    normalises whitespace, and truncates to MAX_TEXT_LENGTH.

    Args:
        file_path: Absolute or relative path to the target file.

    Returns:
        Extracted and normalised text, or an empty string if extraction fails
        or the file type is unsupported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if not path.exists():
        logger.warning("File not found: %s", path)
        return ""

    if ext == ".pdf":
        raw = extract_pdf(path)
    elif ext == ".docx":
        raw = extract_docx(path)
    elif ext == ".csv":
        raw = extract_csv(path)
    elif ext in IMAGE_EXTENSIONS:
        raw = extract_image_metadata(path)
    elif ext in TEXT_EXTENSIONS or ext in CODE_EXTENSIONS:
        raw = extract_txt(path)
    else:
        logger.warning("Unsupported file type '%s': %s", ext, path.name)
        return ""

    return truncate_text(normalize_text(raw))


# Helper: supported extensions

def get_supported_extensions() -> set[str]:
    """Return the full set of file extensions this module can process.

    Returns:
        A set of lowercase extension strings (e.g. {'.txt', '.pdf', ...}).
    """
    return (
        {".pdf", ".docx", ".csv"}
        | set(TEXT_EXTENSIONS)
        | set(CODE_EXTENSIONS)
        | set(IMAGE_EXTENSIONS)
    )


# Extractor functions

def extract_txt(file_path: Path) -> str:
    """Read any UTF-8 compatible text or code file and return its content.

    Covers plain text (.txt), markup (.md, .html, .json, .xml), and all
    code file extensions defined in CODE_EXTENSIONS.

    Args:
        file_path: Path object pointing to the file.

    Returns:
        File content as a string, or an empty string on error.
    """
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Failed to read text file '%s': %s", file_path.name, exc)
        return ""


def extract_pdf(file_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz).

    Handles encrypted PDFs gracefully — logs a warning and returns "".

    Args:
        file_path: Path object pointing to a .pdf file.

    Returns:
        Concatenated text from all pages, or an empty string on error.
    """
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415

        doc = fitz.open(str(file_path))

        if doc.is_encrypted:
            logger.warning("Encrypted PDF, cannot extract text: %s", file_path.name)
            doc.close()
            return ""

        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())

        doc.close()
        return "\n".join(pages)

    except Exception as exc:
        logger.error("PDF extraction failed for '%s': %s", file_path.name, exc)
        return ""


def extract_docx(file_path: Path) -> str:
    """Extract text from a Word document (.docx) using python-docx.

    Reads every paragraph in document order.

    Args:
        file_path: Path object pointing to a .docx file.

    Returns:
        Paragraph text joined by newlines, or an empty string on error.
    """
    try:
        from docx import Document  # noqa: PLC0415

        doc = Document(str(file_path))
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)

    except Exception as exc:
        logger.error("DOCX extraction failed for '%s': %s", file_path.name, exc)
        return ""


def extract_csv(file_path: Path) -> str:
    """Extract content from a CSV file using Python's built-in csv module.

    Each row is converted to a tab-separated line so column boundaries are
    preserved in the output text.

    Args:
        file_path: Path object pointing to a .csv file.

    Returns:
        All rows as a single string, or an empty string on error.
    """
    try:
        lines: list[str] = []
        with file_path.open(encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                lines.append("\t".join(row))
        return "\n".join(lines)

    except Exception as exc:
        logger.error("CSV extraction failed for '%s': %s", file_path.name, exc)
        return ""


def extract_image_metadata(file_path: Path) -> str:
    """Return a short metadata description for an image file.

    No image decoding is performed. This keeps the module dependency-light
    and is sufficient for keyword-based or semantic indexing of file names.

    Args:
        file_path: Path object pointing to an image file.

    Returns:
        A human-readable metadata string describing the image.
    """
    return (
        f"Image file named {file_path.name} "
        f"with extension {file_path.suffix.lower()}"
    )


# Text utilities

def normalize_text(text: str) -> str:
    """Clean and normalise extracted text.

    Steps applied (in order):
        1. Collapse runs of spaces / tabs into a single space.
        2. Collapse three or more consecutive newlines into two.
        3. Strip leading and trailing whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        Normalised text string.
    """
    # Collapse horizontal whitespace (spaces & tabs) within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to a maximum character length.

    Truncation happens at a word boundary when possible to avoid cutting
    mid-word, and a notice is appended so consumers know the text is partial.

    Args:
        text:       The text to truncate.
        max_length: Maximum allowed character count (default: MAX_TEXT_LENGTH).

    Returns:
        Original text if within limit, otherwise a safely truncated version.
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(" ", 1)[0]
    return truncated + "\n\n[... text truncated for indexing ...]"



# Checking
if __name__ == "__main__":
    sample = "Sample_files/tstpython.py"
    result = extract_text(sample)
    print(result if result else "No text extracted.")
    print(f"\n--- Total characters extracted: {len(result)} ---")

# Not checked for an encrypted 