import os
import sys
import warnings

# Suppress DeprecationWarnings from LangChain community modules
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure 'project' is in the Python path regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__)) # services/
project_dir = os.path.dirname(current_dir) # project/
root_dir = os.path.dirname(project_dir) # Multi_agent_researcher/
sys.path.insert(0, root_dir)

# FIX: langchain_community.vectorstores.Chroma is deprecated. Use langchain_chroma instead.
from langchain_chroma import Chroma
from project.config.setting import config
from project.services.extractor import load_documents
from project.services.chunker import split_into_chunks
from project.services.embedder import get_embeddings
from langchain_core.documents import Document

def _get_indexed_filenames(vectorstore: Chroma) -> set:
    """Helper to get the set of filenames already stored in ChromaDB."""
    try:
        results = vectorstore._collection.get(include=["metadatas"])
        filenames = set()
        for meta in results.get("metadatas") or []:
            if meta and "filename" in meta:
                filenames.add(meta["filename"])
        return filenames
    except Exception:
        return set()


def _load_or_create_chroma(embeddings) -> Chroma:
    """Load existing ChromaDB or create a new empty one."""
    return Chroma(
        persist_directory=str(config.CHROMA_DB_DIR),
        embedding_function=embeddings
    )


def sync_corpus():
    """
    Smart sync: reads corpus, compares against what's already in ChromaDB,
    and only ingests NEW files. If you add a new PDF/TXT/DOCX to the corpus
    folder and re-run, it gets auto-ingested without rebuilding the whole DB.
    """
    print("\n🔄 Syncing corpus with vector database...")
    embeddings = get_embeddings()
    vectorstore = _load_or_create_chroma(embeddings)

    # 1. Find what is already indexed
    existing_files = _get_indexed_filenames(vectorstore)
    if existing_files:
        print(f"   Already indexed: {existing_files}. Skipping these.")

    # 2. Load and filter to only new documents
    all_docs = load_documents()
    new_docs = [d for d in all_docs if d["filename"] not in existing_files]

    if not new_docs:
        print("✅ Corpus is already up to date. No new documents to ingest.")
        return vectorstore

    print(f"   Found {len(new_docs)} new document(s). Ingesting...")

    # 3. Chunk new documents only
    chunks = split_into_chunks(new_docs)
    if not chunks:
        print("   No chunks generated from new documents.")
        return vectorstore

    # 4. Convert to LangChain Documents
    # Each chunk dict has {"text": str, "metadata": dict} from our Chunker
    lc_documents = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in chunks
    ]

    # 5. Add to the existing Chroma collection (does NOT wipe existing data)
    print(f"   Adding {len(lc_documents)} chunks to ChromaDB...")
    # New safe batched code
    batch_size = 5000
    for i in range(0, len(lc_documents), batch_size):
        batch = lc_documents[i:i + batch_size]
        vectorstore.add_documents(batch)
    return vectorstore


# Module-level cached retriever — initialized once per process.
_retriever_cache = None

def get_retriever():
    """
    Returns a retriever. Runs sync_corpus on first call to detect new files.
    Subsequent calls within the same process return the cached retriever instantly.
    """
    global _retriever_cache
    if _retriever_cache is None:
        # You pass a raw text string (e.g., "climate targets").
        # LangChain vectorizes it for you automatically via the retriever.
        vectorstore = sync_corpus()
        if vectorstore:
            _retriever_cache = vectorstore.as_retriever(
                search_kwargs={"k": config.DOCUMENT_SEARCH_TOP_K}
            )
    return _retriever_cache


if __name__ == "__main__":
    # If run directly, sync corpus and build/update the vector store
    sync_corpus()