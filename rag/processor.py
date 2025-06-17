import os
import datetime
from typing import List, Dict
from langchain.schema import Document


class AdvanceDocumentProcessor:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path

    @staticmethod
    def _extract_document_metadata(self, file_path: str, doc: List[Document]) -> Dict:
        file_stats = os.stat(file_path)
        metadata = {
            "filename": os.path.basename(file_path),
            "file_size": file_stats.st_size,
            "created_date": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_date": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "total_page": len(doc),
            "estimated_read_time": len(doc) * 2
        }

        if doc:
            first_page = doc[0].page_content[:500]
            lines = first_page.split('\n')
            potential_title = next((line.strip() for line in lines if len(line.strip()) > 10),
                                   "Unknown")
            metadata["extracted_title"] = potential_title

        return metadata

    @staticmethod
    def _categorize_content(content: str) -> str:
        content_lower = content.lower()

        if any(word in content_lower for word in ['chapter', 'introduction', 'conclusion']):
            return "chapter"
        elif any(word in content_lower for word in ['table', 'figure', 'chart', 'graph']):
            return "data_visualization"
        elif any(word in content_lower for word in ['references', 'bibliography', 'citations']):
            return "references"
        elif len(content.split()) < 100:
            return "short_content"
        else:
            return "main_content"

    @staticmethod
    def _extract_key_terms(content: str, max_terms: int = 5) -> List[str]:
        words = content.lower().split()
        meaningful_words = [word for word in words
                            if len(word) > 3 and word.isalpha()]
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [term[0] for term in sorted_terms[:max_terms]]


    def _enhance_page_metadata(self, doc: Document, page_num: int, doc_metadata: Dict,
                               file_path: str) -> Dict:
        content = doc.page_content
        enhanced = {
            "source": file_path,
            "page": page_num + 1,
            "file_type": "pdf",
            "character_count": len(content),
            "word_count": len(content.split()),
            "has_numbers": any(char.isdigit() for char in content),
            "has_bullet_points": any(marker in content for marker in ['•', '○', '-', '1.', '2.']),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
        }
        enhanced.update(doc_metadata)
        enhanced["content_type"] = self._categorize_content(content)
        enhanced["key_terms"] = self._extract_key_terms(content)

        return enhanced