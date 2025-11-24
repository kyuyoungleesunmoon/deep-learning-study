"""
Chapter 04: Document Loaders & Text Splitters 실습 코드
=======================================================

이 파일은 문서 로딩과 텍스트 분할을 실습합니다:
1. 다양한 문서 포맷 처리
2. Text Splitter 구현
3. Chunk Overlap 개념

실행 방법:
    pip install langchain langchain-text-splitters
    python chapter_04_practice.py
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import re


# ============================================================
# Part 1: Document 클래스
# ============================================================

@dataclass
class Document:
    """LangChain Document와 유사한 클래스"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Part 2: Document Loader 구현
# ============================================================

class TextLoader:
    """텍스트 파일 로더 시뮬레이션"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """파일 로드 시뮬레이션"""
        # 실제로는 파일을 읽지만, 여기서는 시뮬레이션
        content = f"[{self.file_path}의 내용]\n" + "샘플 텍스트 " * 50
        
        return [Document(
            page_content=content,
            metadata={"source": self.file_path}
        )]


class PDFLoader:
    """PDF 로더 시뮬레이션"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """PDF 페이지별 로드 시뮬레이션"""
        # 3개 페이지 시뮬레이션
        pages = []
        for i in range(3):
            content = f"PDF 페이지 {i+1}의 내용입니다. " + "텍스트 " * 30
            pages.append(Document(
                page_content=content,
                metadata={"source": self.file_path, "page": i}
            ))
        
        return pages


class CSVLoader:
    """CSV 로더 시뮬레이션"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """CSV 행별 로드 시뮬레이션"""
        rows = [
            {"name": "Alice", "age": 30, "city": "Seoul"},
            {"name": "Bob", "age": 25, "city": "Busan"},
            {"name": "Charlie", "age": 35, "city": "Incheon"}
        ]
        
        docs = []
        for i, row in enumerate(rows):
            content = ", ".join([f"{k}: {v}" for k, v in row.items()])
            docs.append(Document(
                page_content=content,
                metadata={"source": self.file_path, "row": i}
            ))
        
        return docs


def demo_loaders():
    """Document Loader 데모"""
    print("\n" + "="*60)
    print("📄 Document Loader 데모")
    print("="*60)
    
    # 텍스트 로더
    text_loader = TextLoader("sample.txt")
    text_docs = text_loader.load()
    print(f"\n[TextLoader]")
    print(f"  문서 수: {len(text_docs)}")
    print(f"  내용 미리보기: {text_docs[0].page_content[:50]}...")
    
    # PDF 로더
    pdf_loader = PDFLoader("sample.pdf")
    pdf_docs = pdf_loader.load()
    print(f"\n[PDFLoader]")
    print(f"  페이지 수: {len(pdf_docs)}")
    for doc in pdf_docs:
        print(f"  - 페이지 {doc.metadata['page']}: {doc.page_content[:30]}...")
    
    # CSV 로더
    csv_loader = CSVLoader("sample.csv")
    csv_docs = csv_loader.load()
    print(f"\n[CSVLoader]")
    print(f"  행 수: {len(csv_docs)}")
    for doc in csv_docs:
        print(f"  - 행 {doc.metadata['row']}: {doc.page_content}")


# ============================================================
# Part 3: Text Splitter 구현
# ============================================================

class CharacterTextSplitter:
    """문자 기반 텍스트 분할기"""
    
    def __init__(self, separator: str = "\n", chunk_size: int = 100,
                 chunk_overlap: int = 20, length_function=len):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    def split_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        # 구분자로 분할
        splits = text.split(self.separator)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            if current_length + split_length > self.chunk_size and current_chunk:
                # 현재 청크 저장
                chunk_text = self.separator.join(current_chunk)
                chunks.append(chunk_text)
                
                # Overlap 처리
                overlap_text = chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            
            current_chunk.append(split)
            current_length += split_length + len(self.separator)
        
        # 마지막 청크
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Document 리스트 분할"""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i}
                ))
        return result


class RecursiveCharacterTextSplitter:
    """재귀적 문자 텍스트 분할기"""
    
    def __init__(self, separators: List[str] = None, chunk_size: int = 100,
                 chunk_overlap: int = 20):
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """재귀적으로 텍스트 분할"""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        
        # 현재 구분자
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []
        
        # 구분자로 분할
        if separator:
            splits = text.split(separator)
        else:
            # 마지막 수단: 문자 단위 분할
            splits = list(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    
                    # 청크가 너무 크면 재귀적으로 더 분할
                    if len(chunk_text) > self.chunk_size and remaining_separators:
                        sub_chunks = self._split_text_recursive(chunk_text, remaining_separators)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk_text)
                    
                    # Overlap 처리 (간소화)
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(split)
            current_length += split_length + len(separator)
        
        # 마지막 청크
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.chunk_size and remaining_separators:
                sub_chunks = self._split_text_recursive(chunk_text, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Document 리스트 분할"""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                result.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i}
                ))
        return result


def demo_text_splitter():
    """Text Splitter 데모"""
    print("\n" + "="*60)
    print("✂️ Text Splitter 데모")
    print("="*60)
    
    # 샘플 텍스트
    sample_text = """인공지능(AI)은 컴퓨터가 인간의 지능을 모방하는 기술입니다. 
    
머신러닝은 AI의 한 분야입니다. 데이터에서 패턴을 학습합니다.
딥러닝은 머신러닝의 한 방법입니다. 신경망을 사용합니다.

자연어 처리(NLP)는 컴퓨터가 텍스트를 이해하는 기술입니다.
LLM은 대규모 언어 모델의 약자입니다. GPT가 대표적인 예시입니다."""
    
    print(f"\n[원본 텍스트]")
    print(f"길이: {len(sample_text)}자")
    print(sample_text[:100] + "...")
    
    # CharacterTextSplitter
    char_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=150,
        chunk_overlap=30
    )
    char_chunks = char_splitter.split_text(sample_text)
    
    print(f"\n[CharacterTextSplitter] (separator='\\n\\n', chunk_size=150)")
    print(f"청크 수: {len(char_chunks)}")
    for i, chunk in enumerate(char_chunks):
        print(f"  청크 {i+1} ({len(chunk)}자): {chunk[:50]}...")
    
    # RecursiveCharacterTextSplitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=100,
        chunk_overlap=20
    )
    recursive_chunks = recursive_splitter.split_text(sample_text)
    
    print(f"\n[RecursiveCharacterTextSplitter] (chunk_size=100)")
    print(f"청크 수: {len(recursive_chunks)}")
    for i, chunk in enumerate(recursive_chunks):
        print(f"  청크 {i+1} ({len(chunk)}자): {chunk[:50]}...")


# ============================================================
# Part 4: Chunk Overlap 시각화
# ============================================================

def visualize_overlap():
    """Chunk Overlap 시각화"""
    print("\n" + "="*60)
    print("🔗 Chunk Overlap 시각화")
    print("="*60)
    
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    chunk_size = 12
    overlaps = [0, 3, 6]
    
    for overlap in overlaps:
        print(f"\n[chunk_overlap={overlap}]")
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((start, end, text[start:end]))
            start = end - overlap
            if start >= len(text) - overlap:
                break
        
        # 시각화
        print(f"원본: {text}")
        print("-" * (len(text) + 10))
        
        for i, (s, e, chunk) in enumerate(chunks):
            padding = " " * s
            print(f"청크{i+1}: {padding}{chunk}")
        
        print(f"청크 수: {len(chunks)}")


# ============================================================
# Part 5: Document 분할 워크플로우
# ============================================================

def demo_workflow():
    """전체 워크플로우 데모"""
    print("\n" + "="*60)
    print("🔄 전체 워크플로우 데모")
    print("="*60)
    
    # 1. 문서 로드
    loader = PDFLoader("sample.pdf")
    docs = loader.load()
    print(f"\n1. 문서 로드: {len(docs)}개 페이지")
    
    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )
    chunks = splitter.split_documents(docs)
    print(f"2. 텍스트 분할: {len(chunks)}개 청크")
    
    # 3. 결과 확인
    print("\n3. 분할 결과:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"  청크 {i+1}:")
        print(f"    내용: {chunk.page_content[:40]}...")
        print(f"    메타데이터: {chunk.metadata}")


# ============================================================
# Part 6: LangChain 실제 사용 (선택적)
# ============================================================

def demo_langchain_splitter():
    """LangChain Text Splitter 사용"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        print("\n" + "="*60)
        print("🚀 LangChain Text Splitter 데모")
        print("="*60)
        
        text = "긴 문서 내용입니다. " * 50
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        print(f"원본 길이: {len(text)}")
        print(f"청크 수: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  청크 {i+1} ({len(chunk)}자): {chunk[:30]}...")
        
    except ImportError:
        print("\n⚠️ langchain-text-splitters가 설치되지 않았습니다.")
        print("설치: pip install langchain-text-splitters")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 04: Document Loaders & Text Splitters 실습")
    print("="*60)
    
    demo_loaders()
    demo_text_splitter()
    visualize_overlap()
    demo_workflow()
    demo_langchain_splitter()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
