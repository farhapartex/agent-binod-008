# FIXED processors.py
import os
import datetime
from typing import List, Dict
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader


class AdvanceDocumentProcessor:
    def __init__(self):
        self.pdf_file_path = None
        self.min_content_length = 50

    @staticmethod
    def _extract_document_metadata(file_path: str, doc: List[Document]) -> Dict:
        try:
            file_stats = os.stat(file_path)
            metadata = {
                "filename": os.path.basename(file_path),
                "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "created_date": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat()[
                                :19],
                "modified_date": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()[
                                 :19],
                "total_page": len(doc),
                "estimated_read_time": len(doc) * 2
            }

            if doc and doc[0].page_content:
                first_page = doc[0].page_content[:500]
                lines = first_page.split('\n')
                potential_title = next((line.strip() for line in lines if len(line.strip()) > 10),
                                       "Unknown")
                metadata["extracted_title"] = potential_title[
                                              :100] if potential_title else "Unknown"

            return metadata
        except Exception as e:
            print(f"Error extracting document metadata: {e}")
            return {"filename": os.path.basename(file_path)}

    @staticmethod
    def _categorize_content(content: str) -> str:
        """Categorize content type"""
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
        try:
            words = content.lower().split()
            meaningful_words = []
            for word in words:
                clean_word = ''.join(char for char in word if char.isalpha())
                if len(clean_word) > 3 and clean_word not in ['this', 'that', 'with', 'they',
                                                              'have', 'will', 'been', 'from',
                                                              'were']:
                    meaningful_words.append(clean_word)

            word_freq = {}
            for word in meaningful_words:
                word_freq[word] = word_freq.get(word, 0) + 1

            sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [term[0] for term in sorted_terms[:max_terms]]
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []

    def _enhance_page_metadata(
            self,
            doc: Document,
            page_num: int,
            doc_metadata: Dict,
            file_path: str
    ) -> Dict:
        content = doc.page_content

        enhanced = {
            "source": os.path.basename(file_path),
            "page": page_num + 1,
            "file_type": "pdf",
            "character_count": len(content),
            "word_count": len(content.split()),
            "has_numbers": any(char.isdigit() for char in content),
            "has_bullet_points": any(marker in content for marker in ['•', '○', '-', '1.', '2.']),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
        }

        for key, value in doc_metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                enhanced[key] = value
            else:
                enhanced[key] = str(value)

        enhanced["content_type"] = self._categorize_content(content)
        key_terms_list = self._extract_key_terms(content)
        enhanced["key_terms"] = ", ".join(key_terms_list)  # Convert list to comma-separated string
        enhanced["key_terms_count"] = len(key_terms_list)
        enhanced["processed_at"] = datetime.datetime.now().isoformat()[:19]  # Remove microseconds

        return enhanced

    def load_documents(self, pdf_files):
        documents = []

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                continue

            try:
                loader = PyPDFLoader(pdf_file)
                pdf_docs = loader.load()

                if not pdf_docs:
                    continue

                doc_metadata = self._extract_document_metadata(pdf_file, pdf_docs)
                valid_docs_count = 0

                for i, doc in enumerate(pdf_docs):
                    if doc.page_content and len(
                            doc.page_content.strip()) >= self.min_content_length:
                        try:
                            enhance_metadata = self._enhance_page_metadata(
                                doc, i, doc_metadata,pdf_file)
                            doc.metadata.update(enhance_metadata)
                            documents.append(doc)
                            valid_docs_count += 1
                        except Exception as e:
                            pass
                    else:
                        print(f"Skipping page {i + 1}: insufficient content ({len(doc.page_content.strip())} chars)")

                print(f"Successfully processed {valid_docs_count} pages from {os.path.basename(pdf_file)}")

            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

        print(f"Total valid documents loaded: {len(documents)}")

        if not documents:
            documents = self._create_fallback_documents()

        return documents

    def _create_fallback_documents(self) -> List[Document]:
        fallback_content = [
            {
                "content": "Growth mindset is the belief that abilities and intelligence can be developed through dedication, hard work, and learning from failure. This contrasts with a fixed mindset where people believe their talents are innate gifts.",
                "page": 1
            },
            {
                "content": "People with a growth mindset embrace challenges, persist in the face of setbacks, see effort as a path to mastery, learn from criticism, and find inspiration in others' success. They understand that intelligence can be developed.",
                "page": 2
            },
            {
                "content": "Carol Dweck's research shows that praising process and effort rather than intelligence or talent helps develop a growth mindset in children and adults. This type of praise encourages persistence and learning.",
                "page": 3
            }
        ]

        documents = []
        for item in fallback_content:
            doc = Document(
                page_content=item["content"],
                metadata={
                    "source": "fallback_mindset_content.txt",
                    "page": item["page"],
                    "file_type": "text",
                    "character_count": len(item["content"]),
                    "word_count": len(item["content"].split()),
                    "content_type": "main_content",
                    "is_fallback": True
                }
            )
            documents.append(doc)
        return documents


def validate_documents_for_chroma(documents: List[Document]) -> List[Document]:
    valid_docs = []

    for i, doc in enumerate(documents):
        if not doc.page_content or len(doc.page_content.strip()) < 10:
            print(f"Skipping document {i + 1}: insufficient content")
            continue

        # Validate metadata
        safe_metadata = {}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_metadata[key] = value
            elif isinstance(value, list):
                safe_metadata[key] = ", ".join(str(item) for item in value)
            else:
                safe_metadata[key] = str(value)

        doc.metadata = safe_metadata
        valid_docs.append(doc)

    print(f"Validated {len(valid_docs)} documents for Chroma")
    return valid_docs