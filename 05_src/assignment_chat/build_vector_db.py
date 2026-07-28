from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
DATA_FILE = BASE_DIR / "data" / "clinical_skills.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "nora_clinical_skills"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

load_dotenv(SRC_DIR / ".env")
load_dotenv(SRC_DIR / ".secrets", override=True)

from utils.clients import get_client  # noqa: E402


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def read_rows() -> list[dict[str, str]]:
    """Read, validate, and deduplicate the source dataset."""
    with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required_columns = {
        "id",
        "title",
        "topic",
        "difficulty",
        "source_type",
        "content",
    }
    if not rows:
        raise ValueError("The clinical-skills dataset is empty.")

    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    cleaned_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        cleaned = {
            key: (row.get(key) or "").strip()
            for key in required_columns
        }

        if any(not cleaned[key] for key in required_columns):
            raise ValueError(f"Row {row_number} contains an empty required value.")

        normalized_content = " ".join(cleaned["content"].lower().split())

        if cleaned["id"] in seen_ids:
            raise ValueError(f"Duplicate document ID: {cleaned['id']}")

        if normalized_content in seen_content:
            continue

        seen_ids.add(cleaned["id"])
        seen_content.add(normalized_content)
        cleaned_rows.append(cleaned)

    return cleaned_rows


def embed_documents(documents: list[str]) -> list[list[float]]:
    """Create document embeddings with the course OpenAI client."""
    cleaned_documents = [
        " ".join(document.replace("\n", " ").split())
        for document in documents
    ]
    client = get_client()
    response = client.embeddings.create(
        input=cleaned_documents,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def build_database() -> None:
    """Rebuild the file-persistent Chroma collection."""
    rows = read_rows()
    documents = [row["content"] for row in rows]
    embeddings = embed_documents(documents)

    ids = [row["id"] for row in rows]
    metadatas = [
        {
            "title": row["title"],
            "topic": row["topic"],
            "difficulty": row["difficulty"],
            "source_type": row["source_type"],
        }
        for row in rows
    ]

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Original nursing clinical-skills study summaries"},
    )
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Created collection: {COLLECTION_NAME}")
    print(f"Stored documents: {collection.count()}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Database folder: {CHROMA_DIR}")


if __name__ == "__main__":
    build_database()
