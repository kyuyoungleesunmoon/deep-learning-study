# 📖 Chapter 05: RAG 기초 (Retrieval-Augmented Generation)

## 📋 개요

이 챕터에서는 RAG 시스템의 핵심 구성요소를 학습합니다.
- 벡터 저장소 (Vector Store)
- Retriever
- LCEL (LangChain Expression Language)

## 🔬 핵심 개념

### 1. RAG 아키텍처

```
질문 → Retriever → 관련 문서 → LLM → 답변
         ↑
    Vector Store
    (임베딩 + 인덱스)
```

**핵심 단계**:
1. **Indexing**: 문서 → 청크 → 임베딩 → 벡터 저장소
2. **Retrieval**: 질문 → 임베딩 → 유사 문서 검색
3. **Generation**: 질문 + 문서 → LLM → 답변

### 2. Vector Store

**주요 벡터 저장소**:
| 저장소 | 특징 | 용도 |
|--------|------|------|
| Chroma | 경량, 로컬 | 개발/테스트 |
| FAISS | 빠른 검색 | 대규모 데이터 |
| Pinecone | 클라우드 | 프로덕션 |
| Milvus | 분산 처리 | 엔터프라이즈 |

### 3. Embedding 모델

**OpenAI Embeddings**:
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

**Hugging Face Embeddings**:
```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 4. LCEL (LangChain Expression Language)

**특징**: `|` 연산자로 체인 구성

```python
chain = prompt | model | parser
result = chain.invoke({"question": "..."})
```

**병렬 처리**:
```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
) | prompt | llm
```

## 📊 실습 예제

### 예제 1: 기본 RAG 파이프라인

```python
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# 1. 문서 로드 및 분할
loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# 2. 벡터 저장소 생성
vectorstore = Chroma.from_documents(
    docs, 
    OpenAIEmbeddings(model='text-embedding-3-small')
)
retriever = vectorstore.as_retriever()

# 3. LLM 및 프롬프트
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("rlm/rag-prompt")

# 4. 체인 구성
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 질문
answer = rag_chain.invoke("문서의 주요 내용은?")
print(answer)
```

### 예제 2: 대화형 RAG (Memory)

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# 히스토리 인식 retriever
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "대화 기록을 고려하여 검색 쿼리를 재작성하세요."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 메모리
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 대화형 체인
conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
```

### 예제 3: 다양한 Retriever 설정

```python
# MMR (Maximal Marginal Relevance)
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# 유사도 점수 임계값
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8}
)

# 검색 결과 수 조절
retriever_k = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)
```

### 예제 4: 커스텀 프롬프트

```python
from langchain_core.prompts import ChatPromptTemplate

custom_prompt = ChatPromptTemplate.from_template("""
당신은 전문 어시스턴트입니다. 주어진 컨텍스트만을 사용하여 질문에 답변하세요.
컨텍스트에 없는 정보는 "알 수 없습니다"라고 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)
```

### 예제 5: 로컬 임베딩 모델

```python
from langchain_huggingface import HuggingFaceEmbeddings

# 한국어 모델
embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 벡터 저장소 생성
vectorstore = Chroma.from_documents(docs, embeddings)
```

## 🎯 핵심 포인트

1. **청크 크기 조절**: 너무 작으면 문맥 손실, 너무 크면 노이즈 증가
2. **MMR 활용**: 다양한 정보 검색 시 유용
3. **프롬프트 엔지니어링**: 컨텍스트 활용 방법 명시
4. **한국어 임베딩**: 전용 모델 사용 권장

## ⚠️ 주의사항

- 할루시네이션 방지: "모르면 모른다고 답해" 지시
- 비용 관리: 임베딩 API 호출 횟수 확인
- 청크 오버랩: 문맥 연결을 위해 필수

## 📚 참고 자료

- 원본 코드: https://github.com/Kane0002/Langchain-RAG/tree/main/5장
- LangChain RAG: https://python.langchain.com/docs/use_cases/question_answering/
- ChromaDB: https://www.trychroma.com/
