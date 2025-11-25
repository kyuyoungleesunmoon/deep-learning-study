# 🤖 LLM 실습 가이드

이 폴더는 LLM(Large Language Model) 관련 두 개의 GitHub 저장소를 분석하고, 단계별로 실습할 수 있는 코드를 제공합니다.

## 📚 분석 대상 저장소

### 1. [onlybooks/llm](https://github.com/onlybooks/llm)
LLM 관련 서적의 예제 코드로, 10장부터 16장까지 분석합니다.

| 챕터 | 주제 | 설명 |
|------|------|------|
| 10장 | 검색 증강 생성 (RAG) | 벡터 검색, BM25, 하이브리드 검색 |
| 11장 | 문장 임베딩 | Sentence-Transformers, KLUE 데이터셋 |
| 12장 | 벡터 데이터베이스 | FAISS, Pinecone, 인덱싱 기법 |
| 14장 | 멀티모달 LLM | CLIP, 이미지-텍스트 처리 |
| 15장 | LLM 에이전트 | AutoGen, 멀티에이전트 시스템 |
| 16장 | Mamba 아키텍처 | 상태 공간 모델(SSM) |

### 2. [Kane0002/Langchain-RAG](https://github.com/Kane0002/Langchain-RAG)
Langchain을 활용한 RAG 구현 가이드로, 3장부터 6장까지 분석합니다.

| 챕터 | 주제 | 설명 |
|------|------|------|
| 3장 | Models & Prompts | LLM API 활용, 프롬프트 템플릿, Output Parser |
| 4장 | Document Loaders | PDF 로더, Text Splitters |
| 5장 | RAG 기초 | 벡터 저장소, Retriever, LCEL |
| 6장 | Tools & Agents | Tavily 검색, Agent 구축 |

## 📁 폴더 구조

```
llm_practice/
├── README.md                    # 이 파일
├── requirements.txt             # 필요한 패키지 목록
├── onlybooks_llm/              # onlybooks/llm 저장소 분석
│   ├── chapter_10_rag_search.md        # 10장: RAG와 하이브리드 검색
│   ├── chapter_10_practice.py          # 10장: 실습 코드
│   ├── chapter_11_embeddings.md        # 11장: 문장 임베딩
│   ├── chapter_11_practice.py          # 11장: 실습 코드
│   ├── chapter_12_vectordb.md          # 12장: 벡터 데이터베이스
│   ├── chapter_12_practice.py          # 12장: 실습 코드
│   ├── chapter_14_multimodal.md        # 14장: 멀티모달 LLM
│   ├── chapter_14_practice.py          # 14장: 실습 코드
│   ├── chapter_15_agents.md            # 15장: LLM 에이전트
│   ├── chapter_15_practice.py          # 15장: 실습 코드
│   ├── chapter_16_mamba.md             # 16장: Mamba 아키텍처
│   └── chapter_16_practice.py          # 16장: 실습 코드
└── langchain_rag/              # Langchain-RAG 저장소 분석
    ├── chapter_03_models.md            # 3장: Models & Prompts
    ├── chapter_03_practice.py          # 3장: 실습 코드
    ├── chapter_04_loaders.md           # 4장: Document Loaders
    ├── chapter_04_practice.py          # 4장: 실습 코드
    ├── chapter_05_rag.md               # 5장: RAG 기초
    ├── chapter_05_practice.py          # 5장: 실습 코드
    ├── chapter_06_agents.md            # 6장: Tools & Agents
    └── chapter_06_practice.py          # 6장: 실습 코드
```

## 🚀 시작하기

### 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# API 키 설정 (환경 변수)
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"  # 6장 Agent 실습용
```

### 학습 순서

1. **Langchain 기초** (langchain_rag/chapter_03) - LLM API와 프롬프트 이해
2. **문서 처리** (langchain_rag/chapter_04) - PDF 로딩과 텍스트 분할
3. **RAG 구축** (langchain_rag/chapter_05) - 기본 RAG 파이프라인
4. **에이전트** (langchain_rag/chapter_06) - Tool 사용과 Agent 구축
5. **고급 검색** (onlybooks_llm/chapter_10) - 하이브리드 검색
6. **임베딩** (onlybooks_llm/chapter_11) - 문장 임베딩 학습
7. **벡터DB** (onlybooks_llm/chapter_12) - FAISS 최적화
8. **멀티모달** (onlybooks_llm/chapter_14) - CLIP 활용
9. **멀티에이전트** (onlybooks_llm/chapter_15) - AutoGen
10. **최신 아키텍처** (onlybooks_llm/chapter_16) - Mamba SSM

## 📖 각 챕터 설명

### onlybooks/llm 분석

#### 10장: 검색 증강 생성 (RAG)
- **핵심 알고리즘**: BM25, Dense Vector Search, Reciprocal Rank Fusion
- **주요 내용**: 
  - 키워드 기반 BM25 검색과 벡터 유사도 검색의 장단점
  - 하이브리드 검색으로 두 방식의 장점 결합
  - RRF(Reciprocal Rank Fusion)로 랭킹 통합

#### 11장: 문장 임베딩
- **핵심 알고리즘**: Sentence-BERT, Contrastive Learning
- **주요 내용**:
  - 사전 학습된 언어 모델로 문장 임베딩 생성
  - KLUE STS 데이터셋으로 유사도 측정
  - Mean Pooling 기법

#### 12장: 벡터 데이터베이스
- **핵심 알고리즘**: IVF, PQ, HNSW
- **주요 내용**:
  - FAISS 인덱스 최적화
  - 메모리 효율적인 양자화
  - 대규모 데이터 처리

#### 14장: 멀티모달 LLM
- **핵심 알고리즘**: CLIP (Contrastive Language-Image Pretraining)
- **주요 내용**:
  - 이미지-텍스트 임베딩
  - Zero-shot 이미지 분류
  - 멀티모달 검색

#### 15장: LLM 에이전트
- **핵심 알고리즘**: AutoGen Multi-Agent Framework
- **주요 내용**:
  - UserProxyAgent와 AssistantAgent
  - 코드 실행 에이전트
  - 멀티에이전트 협업

#### 16장: Mamba 아키텍처
- **핵심 알고리즘**: Selective State Space Model (S6)
- **주요 내용**:
  - Transformer 대안 아키텍처
  - 선형 시간 복잡도
  - 긴 시퀀스 처리

### Langchain-RAG 분석

#### 3장: Models & Prompts
- **핵심 개념**: LLM API 호출, 프롬프트 엔지니어링
- **주요 내용**:
  - OpenAI, Anthropic API 통합
  - ChatPromptTemplate
  - Temperature, Streaming, Caching

#### 4장: Document Loaders
- **핵심 개념**: 문서 로딩 및 분할
- **주요 내용**:
  - PyPDF, PyPDFium2 로더
  - CharacterTextSplitter
  - RecursiveCharacterTextSplitter

#### 5장: RAG 기초
- **핵심 개념**: Retrieval-Augmented Generation
- **주요 내용**:
  - ChromaDB 벡터 저장소
  - LCEL (LangChain Expression Language)
  - 대화 히스토리와 메모리

#### 6장: Tools & Agents
- **핵심 개념**: 외부 도구 활용 에이전트
- **주요 내용**:
  - Tavily AI 검색 도구
  - OpenAI Tools Agent
  - 벡터 DB + 웹 검색 통합

## ⚡ 최신 라이브러리 버전 (2024년 11월 기준)

| 패키지 | 권장 버전 | 설명 |
|--------|----------|------|
| langchain | >=0.3.0 | 핵심 프레임워크 |
| langchain-openai | >=0.2.0 | OpenAI 통합 |
| langchain-community | >=0.3.0 | 커뮤니티 통합 |
| langchain-chroma | >=0.1.0 | ChromaDB 통합 |
| openai | >=1.50.0 | OpenAI API |
| sentence-transformers | >=3.0.0 | 문장 임베딩 |
| faiss-cpu | >=1.8.0 | 벡터 검색 |
| transformers | >=4.45.0 | Hugging Face |
| torch | >=2.4.0 | PyTorch |

## 📝 라이선스

이 자료는 학습 목적으로 제작되었습니다.
원본 코드의 라이선스는 각 원본 저장소를 참조하세요.
