# 📖 Chapter 04: Document Loaders & Text Splitters

## 📋 개요

이 챕터에서는 문서를 로드하고 분할하는 방법을 학습합니다.
- PDF, CSV 등 다양한 문서 로더
- 텍스트 분할 전략
- Chunk Overlap의 중요성

## 🔬 핵심 개념

### 1. Document Loader

**목적**: 다양한 형식의 문서를 LangChain Document 객체로 변환

**주요 로더**:
| 로더 | 파일 형식 | 특징 |
|------|----------|------|
| `PyPDFLoader` | PDF | 페이지별 분리 |
| `PyPDFium2Loader` | PDF | 빠른 속도 |
| `TextLoader` | TXT | 기본 텍스트 |
| `CSVLoader` | CSV | 행별 문서 |
| `UnstructuredLoader` | 다양함 | 범용 |

### 2. Text Splitter

**목적**: 긴 문서를 작은 청크로 분할

**왜 분할이 필요한가?**
- LLM의 컨텍스트 길이 제한
- 벡터 검색의 효율성
- 관련 정보만 추출

**핵심 파라미터**:
- `chunk_size`: 청크 최대 크기 (문자 수)
- `chunk_overlap`: 청크 간 중첩 크기
- `separator`: 분할 기준 문자

### 3. Chunk Overlap

**왜 필요한가?**
```
청크1: "인공지능은 데이터에서 패턴을 학습하는"
청크2: "학습하는 기술입니다. 머신러닝은"
청크3: "머신러닝은 인공지능의 한 분야입니다."
```

Overlap으로 문맥이 끊기는 것을 방지!

### 4. Splitter 종류

**CharacterTextSplitter**:
- 단일 구분자로 분할
- 간단하지만 청크 크기가 불균일할 수 있음

**RecursiveCharacterTextSplitter**:
- 여러 구분자를 순차적으로 적용
- `["\n\n", "\n", " ", ""]` 순서로 시도
- 의미 단위를 더 잘 유지

**TokenTextSplitter**:
- 토큰 기준으로 분할
- LLM 토큰 제한에 정확히 맞춤

## 📊 실습 예제

### 예제 1: PDF 로딩

```python
from langchain.document_loaders import PyPDFLoader

# PDF 로드
loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()

# 결과 확인
for i, page in enumerate(pages[:3]):
    print(f"--- 페이지 {i+1} ---")
    print(f"내용: {page.page_content[:200]}...")
    print(f"메타데이터: {page.metadata}")
```

### 예제 2: CharacterTextSplitter

```python
from langchain_text_splitters import CharacterTextSplitter

text = """
인공지능(AI)은 컴퓨터가 인간의 지능을 모방하는 기술입니다.

머신러닝은 AI의 한 분야로, 데이터에서 패턴을 학습합니다.
딥러닝은 머신러닝의 한 방법으로, 신경망을 사용합니다.

자연어 처리(NLP)는 컴퓨터가 텍스트를 이해하는 기술입니다.
"""

splitter = CharacterTextSplitter(
    separator="\n\n",  # 빈 줄로 분할
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"청크 {i+1} ({len(chunk)}자): {chunk[:50]}...")
```

### 예제 3: RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"청크 {i+1}: {chunk}")
```

### 예제 4: Document 분할

```python
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 로드
loader = PyPDFLoader("document.pdf")
pages = loader.load()

# 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = splitter.split_documents(pages)

print(f"원본 페이지 수: {len(pages)}")
print(f"분할 후 청크 수: {len(docs)}")
```

### 예제 5: 청크 크기별 비교

```python
text = "매우 긴 문서 내용..." * 100

for chunk_size in [100, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1)
    )
    chunks = splitter.split_text(text)
    
    print(f"chunk_size={chunk_size}:")
    print(f"  청크 수: {len(chunks)}")
    print(f"  평균 길이: {sum(len(c) for c in chunks) / len(chunks):.0f}")
```

## 🎯 핵심 포인트

1. **RecursiveCharacterTextSplitter 권장**: 의미 단위 보존에 효과적
2. **적절한 chunk_size**: 너무 작으면 문맥 손실, 너무 크면 검색 정확도 저하
3. **chunk_overlap 필수**: 10~20% 정도 권장
4. **메타데이터 활용**: 원본 페이지, 소스 파일 정보 유지

## ⚠️ 주의사항

- PDF OCR: 이미지 내 텍스트는 추가 설정 필요
- 한글 문서: 토큰 분할 시 영어와 다른 특성
- 테이블: 일반 로더로는 구조 손실 가능

## 📚 참고 자료

- 원본 코드: https://github.com/Kane0002/Langchain-RAG/tree/main/4장
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
