"""
Chapter 05: RAG 기초 실습 코드
==============================

이 파일은 RAG (Retrieval-Augmented Generation)의 핵심 개념을 실습합니다:
1. 벡터 저장소 시뮬레이션
2. Retriever 구현
3. RAG 체인 구성

실행 방법:
    pip install numpy
    python chapter_05_practice.py

    # LangChain 사용 시:
    pip install langchain langchain-openai langchain-chroma
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field


# ============================================================
# Part 1: Document 클래스
# ============================================================

@dataclass
class Document:
    """문서 클래스"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Part 2: 간단한 임베딩 모델
# ============================================================

class SimpleEmbedding:
    """간단한 임베딩 모델 (데모용)"""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """문서들을 임베딩으로 변환"""
        return [self._embed(text) for text in texts]
    
    def embed_query(self, text: str) -> np.ndarray:
        """쿼리를 임베딩으로 변환"""
        return self._embed(text)
    
    def _embed(self, text: str) -> np.ndarray:
        """텍스트를 임베딩으로 변환 (해시 기반)"""
        np.random.seed(hash(text.lower()[:100]) % 2**31)
        embedding = np.random.randn(self.dim)
        return embedding / np.linalg.norm(embedding)


# ============================================================
# Part 3: 벡터 저장소
# ============================================================

class SimpleVectorStore:
    """간단한 벡터 저장소"""
    
    def __init__(self, embedding_model: SimpleEmbedding):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
    
    def add_documents(self, documents: List[Document]):
        """문서 추가"""
        texts = [doc.page_content for doc in documents]
        new_embeddings = self.embedding_model.embed_documents(texts)
        
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """유사도 검색"""
        query_embedding = self.embedding_model.embed_query(query)
        
        # 코사인 유사도 계산
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb)
            similarities.append(sim)
        
        # 상위 k개 인덱스
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """유사도 점수와 함께 검색"""
        query_embedding = self.embedding_model.embed_query(query)
        
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb)
            similarities.append(sim)
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [(self.documents[i], similarities[i]) for i in top_indices]
    
    def as_retriever(self, search_kwargs: Dict = None) -> 'SimpleRetriever':
        """Retriever 반환"""
        return SimpleRetriever(self, search_kwargs or {"k": 4})


# ============================================================
# Part 4: Retriever
# ============================================================

class SimpleRetriever:
    """간단한 Retriever"""
    
    def __init__(self, vectorstore: SimpleVectorStore, search_kwargs: Dict):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs
    
    def invoke(self, query: str) -> List[Document]:
        """쿼리에 대한 관련 문서 검색"""
        k = self.search_kwargs.get("k", 4)
        return self.vectorstore.similarity_search(query, k=k)


# ============================================================
# Part 5: RAG 체인
# ============================================================

class SimpleLLM:
    """간단한 LLM 시뮬레이터"""
    
    def invoke(self, prompt: str) -> str:
        """LLM 호출 시뮬레이션"""
        # 프롬프트에서 컨텍스트와 질문 추출 (간소화)
        if "컨텍스트" in prompt or "context" in prompt.lower():
            return "주어진 컨텍스트를 기반으로 답변합니다: [답변 내용]"
        return f"'{prompt[:50]}...'에 대한 응답입니다."


class SimpleRAGChain:
    """간단한 RAG 체인"""
    
    def __init__(self, retriever: SimpleRetriever, llm: SimpleLLM, 
                 prompt_template: str = None):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or """
다음 컨텍스트를 사용하여 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
    
    def _format_docs(self, docs: List[Document]) -> str:
        """문서들을 문자열로 포맷팅"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """RAG 체인 실행"""
        # 1. 검색
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)
        
        # 2. 프롬프트 생성
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # 3. LLM 호출
        answer = self.llm.invoke(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "context": context
        }


# ============================================================
# 데모 함수들
# ============================================================

def demo_vector_store():
    """벡터 저장소 데모"""
    print("\n" + "="*60)
    print("🗄️ 벡터 저장소 데모")
    print("="*60)
    
    # 임베딩 모델
    embedding_model = SimpleEmbedding(dim=64)
    
    # 벡터 저장소
    vectorstore = SimpleVectorStore(embedding_model)
    
    # 샘플 문서
    documents = [
        Document(page_content="인공지능은 컴퓨터가 인간의 지능을 모방하는 기술입니다."),
        Document(page_content="머신러닝은 데이터에서 패턴을 학습하는 AI의 한 분야입니다."),
        Document(page_content="딥러닝은 신경망을 사용하는 머신러닝 방법입니다."),
        Document(page_content="자연어 처리는 컴퓨터가 텍스트를 이해하는 기술입니다."),
        Document(page_content="컴퓨터 비전은 이미지를 분석하는 AI 기술입니다.")
    ]
    
    # 문서 추가
    vectorstore.add_documents(documents)
    print(f"\n문서 {len(documents)}개 추가됨")
    
    # 검색 테스트
    query = "AI 기술"
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    print(f"\n쿼리: '{query}'")
    print("\n검색 결과:")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc.page_content}")


def demo_retriever():
    """Retriever 데모"""
    print("\n" + "="*60)
    print("🔍 Retriever 데모")
    print("="*60)
    
    # 설정
    embedding_model = SimpleEmbedding(dim=64)
    vectorstore = SimpleVectorStore(embedding_model)
    
    documents = [
        Document(page_content="대한민국의 수도는 서울입니다.", metadata={"source": "지리"}),
        Document(page_content="서울의 인구는 약 천만 명입니다.", metadata={"source": "통계"}),
        Document(page_content="한강은 서울을 관통하는 강입니다.", metadata={"source": "지리"}),
        Document(page_content="경복궁은 서울에 있는 조선시대 궁궐입니다.", metadata={"source": "역사"}),
        Document(page_content="파리는 프랑스의 수도입니다.", metadata={"source": "지리"})
    ]
    
    vectorstore.add_documents(documents)
    
    # Retriever 생성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 검색
    queries = ["서울에 대해 알려줘", "유럽의 도시"]
    
    for query in queries:
        print(f"\n쿼리: '{query}'")
        docs = retriever.invoke(query)
        print("검색된 문서:")
        for doc in docs:
            print(f"  - {doc.page_content[:50]}...")


def demo_rag_chain():
    """RAG 체인 데모"""
    print("\n" + "="*60)
    print("⛓️ RAG 체인 데모")
    print("="*60)
    
    # 구성 요소 생성
    embedding_model = SimpleEmbedding(dim=64)
    vectorstore = SimpleVectorStore(embedding_model)
    
    documents = [
        Document(page_content="GPT-4는 OpenAI가 개발한 대규모 언어 모델입니다."),
        Document(page_content="Claude는 Anthropic이 개발한 AI 어시스턴트입니다."),
        Document(page_content="LLaMA는 Meta가 개발한 오픈소스 언어 모델입니다."),
        Document(page_content="Gemini는 Google이 개발한 멀티모달 AI 모델입니다.")
    ]
    
    vectorstore.add_documents(documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = SimpleLLM()
    
    # RAG 체인
    rag_chain = SimpleRAGChain(retriever, llm)
    
    # 질문
    question = "OpenAI의 모델은 무엇인가요?"
    result = rag_chain.invoke(question)
    
    print(f"\n질문: {result['question']}")
    print(f"\n컨텍스트:")
    for doc in result['source_documents']:
        print(f"  - {doc.page_content}")
    print(f"\n답변: {result['answer']}")


def demo_lcel_simulation():
    """LCEL 스타일 체인 데모"""
    print("\n" + "="*60)
    print("🔗 LCEL 스타일 체인 시뮬레이션")
    print("="*60)
    
    class Runnable:
        """LCEL Runnable 시뮬레이션"""
        
        def __init__(self, func):
            self.func = func
        
        def __or__(self, other):
            """| 연산자 오버로딩"""
            return ChainedRunnable([self, other])
        
        def invoke(self, input_data):
            return self.func(input_data)
    
    class ChainedRunnable(Runnable):
        """체인된 Runnable"""
        
        def __init__(self, runnables):
            self.runnables = runnables
        
        def __or__(self, other):
            return ChainedRunnable(self.runnables + [other])
        
        def invoke(self, input_data):
            result = input_data
            for runnable in self.runnables:
                result = runnable.invoke(result)
            return result
    
    # 체인 구성 요소
    def retrieve(query):
        return {"query": query, "docs": ["문서1", "문서2"]}
    
    def format_context(data):
        return {**data, "context": " | ".join(data["docs"])}
    
    def generate(data):
        return f"쿼리 '{data['query']}'에 대한 답변 (컨텍스트: {data['context']})"
    
    # LCEL 스타일 체인
    chain = Runnable(retrieve) | Runnable(format_context) | Runnable(generate)
    
    result = chain.invoke("테스트 질문")
    print(f"\n체인 결과: {result}")


def demo_langchain_rag():
    """LangChain RAG 실제 사용 (선택적)"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_chroma import Chroma
        import os
        
        print("\n" + "="*60)
        print("🚀 LangChain RAG 데모")
        print("="*60)
        
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            return
        
        # 실제 RAG 구현 예시 코드
        print("""
예제 코드:

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

# 문서 준비 및 벡터 저장소 생성
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG 체인
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = rag_chain.invoke("질문...")
        """)
        
    except ImportError:
        print("\n⚠️ langchain 패키지가 설치되지 않았습니다.")
        print("설치: pip install langchain langchain-openai langchain-chroma")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 05: RAG 기초 실습")
    print("="*60)
    
    demo_vector_store()
    demo_retriever()
    demo_rag_chain()
    demo_lcel_simulation()
    demo_langchain_rag()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
