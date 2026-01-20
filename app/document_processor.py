"""
Document Processor Module.
Handles PDF loading, text extraction, table extraction, and chunking.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    content: str
    page_number: int
    section: str = ""
    chunk_type: str = "text"  # "text" or "table"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "page_number": self.page_number,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata
        }


class DocumentProcessor:
    """
    Processes PDF documents for RAG applications.
    
    Features:
    - Text extraction with layout preservation
    - Table extraction as structured data
    - Section-aware chunking
    - Configurable chunk size and overlap
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.current_section = "Introduction"
    
    def load_pdf(self, pdf_path: Path) -> Dict[int, str]:
        """
        Load PDF and extract text page by page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to text content
        """
        pages = {}
        
        with fitz.open(pdf_path) as doc:
            for page_num in tqdm(range(len(doc)), desc="Extracting text"):
                page = doc[page_num]
                text = page.get_text("text")
                # Clean the text
                text = self._clean_text(text)
                pages[page_num + 1] = text  # 1-indexed page numbers
        
        return pages
    
    def extract_tables(self, pdf_path: Path) -> List[Chunk]:
        """
        Extract tables from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Chunk objects containing table data
        """
        table_chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting tables"), 1):
                tables = page.extract_tables()
                
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:  # Has header and at least one row
                        table_text = self._format_table(table)
                        if table_text.strip():
                            chunk = Chunk(
                                id=f"table_p{page_num}_t{table_idx}",
                                content=table_text,
                                page_number=page_num,
                                section=self._detect_section_from_table(table),
                                chunk_type="table",
                                metadata={"rows": len(table), "cols": len(table[0]) if table else 0}
                            )
                            table_chunks.append(chunk)
        
        return table_chunks
    
    def chunk_pages(self, pages: Dict[int, str]) -> List[Chunk]:
        """
        Split pages into chunks with overlap.
        
        Uses recursive splitting to maintain semantic boundaries.
        
        Args:
            pages: Dictionary of page number to text
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_idx = 0
        
        for page_num, text in pages.items():
            # Detect sections in the text
            self._update_current_section(text)
            
            # Split into paragraphs first
            paragraphs = self._split_into_paragraphs(text)
            
            # Build chunks from paragraphs
            current_chunk = ""
            
            for para in paragraphs:
                # Check if adding this paragraph exceeds chunk size
                if len(current_chunk) + len(para) > self.chunk_size:
                    if current_chunk.strip():
                        chunk = Chunk(
                            id=f"chunk_{chunk_idx}",
                            content=current_chunk.strip(),
                            page_number=page_num,
                            section=self.current_section,
                            chunk_type="text"
                        )
                        chunks.append(chunk)
                        chunk_idx += 1
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + para
                else:
                    current_chunk += para
            
            # Don't forget the last chunk from this page
            if current_chunk.strip():
                chunk = Chunk(
                    id=f"chunk_{chunk_idx}",
                    content=current_chunk.strip(),
                    page_number=page_num,
                    section=self.current_section,
                    chunk_type="text"
                )
                chunks.append(chunk)
                chunk_idx += 1
        
        return chunks
    
    def process_document(self, pdf_path: Path) -> List[Chunk]:
        """
        Full document processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of all chunks (text + tables)
        """
        print(f"Processing document: {pdf_path}")
        
        # Extract text
        pages = self.load_pdf(pdf_path)
        print(f"Extracted text from {len(pages)} pages")
        
        # Chunk text
        text_chunks = self.chunk_pages(pages)
        print(f"Created {len(text_chunks)} text chunks")
        
        # Extract tables
        table_chunks = self.extract_tables(pdf_path)
        print(f"Extracted {len(table_chunks)} tables")
        
        # Combine and sort by page number
        all_chunks = text_chunks + table_chunks
        all_chunks.sort(key=lambda c: (c.page_number, c.id))
        
        # Re-assign chunk IDs
        for idx, chunk in enumerate(all_chunks):
            chunk.id = f"chunk_{idx}"
        
        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Fix encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'Annual Report FY 2023-24', '', text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        return text.strip()
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on multiple newlines or semantic markers."""
        # Split on double newlines or bullet points
        paragraphs = re.split(r'\n\n+|\n(?=•|\-\s|\d+\.)', text)
        return [p.strip() + "\n\n" for p in paragraphs if p.strip()]
    
    def _update_current_section(self, text: str) -> None:
        """Detect and update current section based on text content."""
        section_patterns = [
            (r"(?i)board['']?s?\s*report", "Board's Report"),
            (r"(?i)management\s*discussion", "Management Discussion & Analysis"),
            (r"(?i)corporate\s*governance", "Corporate Governance"),
            (r"(?i)financial\s*statements?", "Financial Statements"),
            (r"(?i)director['']?s?\s*report", "Director's Report"),
            (r"(?i)auditor['']?s?\s*report", "Auditor's Report"),
            (r"(?i)notes\s*to\s*(?:the\s*)?financial", "Notes to Financial Statements"),
            (r"(?i)business\s*overview", "Business Overview"),
            (r"(?i)risk\s*(?:factors|management)", "Risk Management"),
            (r"(?i)sustainability", "Sustainability"),
        ]
        
        for pattern, section_name in section_patterns:
            if re.search(pattern, text[:500]):  # Check first 500 chars
                self.current_section = section_name
                break
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format a table as readable text."""
        if not table:
            return ""
        
        # Use first row as headers
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        
        formatted_rows = []
        formatted_rows.append(" | ".join(headers))
        formatted_rows.append("-" * 40)
        
        for row in table[1:]:
            cells = [str(cell).strip() if cell else "" for cell in row]
            formatted_rows.append(" | ".join(cells))
        
        return "\n".join(formatted_rows)
    
    def _detect_section_from_table(self, table: List[List[str]]) -> str:
        """Try to detect section from table headers."""
        if not table:
            return self.current_section
        
        # Check first row for keywords
        header_text = " ".join(str(cell) for cell in table[0] if cell).lower()
        
        if any(kw in header_text for kw in ["revenue", "income", "expense", "profit", "loss"]):
            return "Financial Statements"
        elif any(kw in header_text for kw in ["director", "board", "committee"]):
            return "Corporate Governance"
        elif any(kw in header_text for kw in ["risk", "mitigation"]):
            return "Risk Management"
        
        return self.current_section


# CLI for testing
if __name__ == "__main__":
    from app.config import PDF_PATH
    
    processor = DocumentProcessor()
    chunks = processor.process_document(PDF_PATH)
    
    print("\n--- Sample Chunks ---")
    for chunk in chunks[:3]:
        print(f"\n[{chunk.id}] Page {chunk.page_number} | Section: {chunk.section}")
        print(f"Type: {chunk.chunk_type}")
        print(f"Content preview: {chunk.content[:200]}...")
