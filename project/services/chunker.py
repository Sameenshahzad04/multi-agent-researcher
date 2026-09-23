from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from project.config.setting import config

class Chunker:
    """Text and Markdown Table Chunker for RAG pipelines."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split regular text into chunks."""
        if not text or not text.strip():
            return []
        return self.text_splitter.split_text(text)

    def chunk_table(self, table_markdown: str, table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk markdown table by row groups while preserving the table header."""
        if not table_markdown:
            return []

        max_chars = config.TABLE_MAX_CHARS
        total_rows = table_info.get("num_rows", 0)

        # Return whole table if small enough
        if len(table_markdown) <= max_chars:
            return [{
                "chunk": table_markdown,
                "chunk_index": 0,
                "row_start": 0,
                "row_end": total_rows,
                "is_full_table": True
            }]

        lines = table_markdown.split("\n")
        if len(lines) < 3:  # Needs header (1), separator (1), and data rows (1+)
            return [{
                "chunk": table_markdown,
                "chunk_index": 0,
                "row_start": 0,
                "row_end": total_rows,
                "is_full_table": True
            }]

        headers = lines[:2]
        data_rows = lines[2:]
        header_len = sum(len(line) + 1 for line in headers)

        chunks = []
        i = 0
        overlap = config.TABLE_CHUNK_OVERLAP

        while i < len(data_rows):
            current_chunk = headers.copy()
            current_len = header_len
            row_start = i

            # Accumulate rows up to max character limit
            while i < len(data_rows):
                row_len = len(data_rows[i]) + 1
                if current_len + row_len > max_chars and len(current_chunk) > 2:
                    break
                current_chunk.append(data_rows[i])
                current_len += row_len
                i += 1

            row_end = i

            chunks.append({
                "chunk": "\n".join(current_chunk),
                "chunk_index": len(chunks),
                "row_start": row_start,
                "row_end": row_end,
                "is_full_table": False
            })

            # Apply row overlap for next iteration
            if i < len(data_rows):
                i = max(row_start + 1, i - overlap)

        return chunks

    def create_text_chunks_with_metadata(
        self, text: str, doc_id: int, filename: str, file_type: str
    ) -> List[Dict[str, Any]]:
        """Chunk text and attach document metadata."""
        return [
            {
                "text": chunk,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_type": config.CHUNK_TYPE_TEXT,
                    "chunk_index": idx
                }
            }
            for idx, chunk in enumerate(self.chunk_text(text))
        ]

    def create_table_chunks_with_metadata(
        self, tables: List[Dict[str, Any]], doc_id: int, filename: str, file_type: str
    ) -> List[Dict[str, Any]]:
        """Chunk tables and attach document metadata."""
        result = []
        chunk_counter = 0

        for table in tables:
            table_chunks = self.chunk_table(table.get("markdown", ""), table)
            for chunk_info in table_chunks:
                result.append({
                    "text": chunk_info["chunk"],
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": filename,
                        "file_type": file_type,
                        "chunk_type": config.CHUNK_TYPE_TABLE,
                        "chunk_index": chunk_counter,
                        "table_page": table.get("page", 0),
                        "table_index": table.get("table_index", 0),
                        "row_start": chunk_info["row_start"],
                        "row_end": chunk_info["row_end"],
                        "is_full_table": chunk_info["is_full_table"]
                    }
                })
                chunk_counter += 1

        return result

def split_into_chunks(documents_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Splits structured loaded documents into chunks with metadata."""
    if not documents_data:
        return []

    print("Splitting documents into chunks with advanced chunker...")
    chunker = Chunker()
    all_chunks = []

    for doc in documents_data:
        # 1. Process Text
        if doc.get("text"):
            text_chunks = chunker.create_text_chunks_with_metadata(
                text=doc["text"],
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                file_type=doc["file_type"]
            )
            all_chunks.extend(text_chunks)
            
        # 2. Process Tables
        if doc.get("tables"):
            table_chunks = chunker.create_table_chunks_with_metadata(
                tables=doc["tables"],
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                file_type=doc["file_type"]
            )
            all_chunks.extend(table_chunks)

    print(f"Generated {len(all_chunks)} advanced chunks.")
    return all_chunks
