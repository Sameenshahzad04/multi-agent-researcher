from pathlib import Path
from pypdf import PdfReader
import pdfplumber
from docx import Document
from tabulate import tabulate
from typing import Tuple, List, Dict, Any
import re
from project.config.setting import config

class TableExtractor:
    """
    Table Extraction and Markdown Conversion
    Extract tables separately → Convert to Markdown → Chunk intelligently
    """
    def __init__(self):
        """Initialize table extractor"""
        pass
    
    def extract_tables_from_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        tables = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table_idx, table_data in enumerate(page_tables):
                    if table_data and len(table_data) > 1:
                        # Clean table data (remove None values)
                        cleaned_data = self._clean_table_data(table_data)
                        # Convert to Markdown
                        markdown = self._convert_to_markdown(cleaned_data)
                        tables.append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "data": cleaned_data,
                            "markdown": markdown,
                            "num_rows": len(cleaned_data),
                            "num_cols": len(cleaned_data[0]) if cleaned_data else 0
                        })
        return tables
    
    def extract_tables_from_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        tables = []
        try:
            doc = Document(str(file_path))
            for table_idx, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                
                if table_data and len(table_data) > 1:
                    markdown = self._convert_to_markdown(table_data)
                    tables.append({
                        "page": 0,  # DOCX doesn't have pages
                        "table_index": table_idx,
                        "data": table_data,
                        "markdown": markdown,
                        "num_rows": len(table_data),
                        "num_cols": len(table_data[0]) if table_data else 0
                    })
        except Exception as e:
            print(f"⚠️  Failed to extract tables from DOCX: {str(e)}")
        
        return tables
    
    def _clean_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        cleaned = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    cleaned_row.append(str(cell).strip())
            cleaned.append(cleaned_row)
        return cleaned
    
    def _convert_to_markdown(self, table_data: List[List[str]]) -> str:
        if not table_data or len(table_data) < 2:
            return ""
        try:
            headers = table_data[0]
            data_rows = table_data[1:]
            markdown = tabulate(
                data_rows,
                headers=headers,
                tablefmt="github"
            )
            return markdown
        except Exception as e:
            return self._simple_markdown_fallback(table_data)
    
    def _simple_markdown_fallback(self, table_data: List[List[str]]) -> str:
        if not table_data:
            return ""
        lines = []
        header = "| " + " | ".join(str(cell) for cell in table_data[0]) + " |"
        lines.append(header)
        separator = "| " + " | ".join("---" for _ in table_data[0]) + " |"
        lines.append(separator)
        for row in table_data[1:]:
            data_row = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(data_row)
        return "\n".join(lines)
    
    def extract_all_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        file_ext = file_path.suffix.lower()
        if file_ext == ".pdf":
            return self.extract_tables_from_pdf(file_path)
        elif file_ext in [".docx", ".doc"]:
            return self.extract_tables_from_docx(file_path)
        else:
            return []


def extract_from_pdf(file_path: Path) -> str:
    """Extract text from PDF using pypdf (FREE)"""
    try:
        reader = PdfReader(str(file_path))
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i + 1} ---\n"
                text += page_text
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF {file_path}: {str(e)}")

def extract_from_docx(file_path: Path) -> str:
    """Extract text from DOCX using python-docx (FREE)"""
    try:
        doc = Document(str(file_path))
        text = ""
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX {file_path}: {str(e)}")

def extract_from_txt(file_path: Path) -> str:
    """Extract text from TXT file (built-in Python)"""
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1").strip()

def extract_text(file_path: Path) -> str:
    """Extract text from uploaded file based on its extension."""
    file_ext = file_path.suffix.lower()
    if file_ext == ".pdf":
        return extract_from_pdf(file_path)
    elif file_ext in [".docx", ".doc"]:
        return extract_from_docx(file_path)
    elif file_ext == ".txt":
        return extract_from_txt(file_path)
    else:
        print(f"Skipping unsupported file type: {file_ext}")
        return ""

def load_documents() -> List[Dict[str, Any]]:
    """Iterates through the corpus, extracting both text and tables."""
    print("Loading documents from corpus using advanced extraction...")
    
    if not config.CORPUS_DIR.exists():
        print(f"Corpus directory {config.CORPUS_DIR} not found.")
        return []

    documents_data = []
    table_extractor = TableExtractor()
    doc_id_counter = 1

    for file_path in config.CORPUS_DIR.glob("**/*"):
        if file_path.is_file():
            print(f"Processing {file_path.name}...")
            
            # Extract main text
            text_content = extract_text(file_path)
            
            # Extract tables
            tables_data = table_extractor.extract_all_tables(file_path)
            
            if text_content or tables_data:
                documents_data.append({
                    "doc_id": doc_id_counter,
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "text": text_content,
                    "tables": tables_data
                })
                doc_id_counter += 1

    print(f"Loaded {len(documents_data)} documents.")
    return documents_data
