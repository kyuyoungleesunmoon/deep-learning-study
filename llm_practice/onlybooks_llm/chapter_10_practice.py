"""
Chapter 10: RAG와 하이브리드 검색 실습 코드
=============================================

이 파일은 검색 증강 생성(RAG)의 핵심 검색 기법들을 실습합니다:
1. Dense Vector Search (벡터 검색)
2. BM25 (키워드 검색)
3. Hybrid Search (하이브리드 검색)

실행 방법:
    pip install sentence-transformers faiss-cpu transformers torch
    python chapter_10_practice.py
"""

import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass


# ============================================================
# Part 1: 간단한 벡터 검색 (FAISS 없이도 동작하는 버전)
# ============================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간의 코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def simple_dense_search(query_embedding: np.ndarray, 
                        doc_embeddings: np.ndarray, 
                        k: int = 3) -> List[Tuple[int, float]]:
    """
    간단한 벡터 검색 구현
    
    Args:
        query_embedding: 쿼리 벡터
        doc_embeddings: 문서 벡터들
        k: 반환할 상위 k개 결과
    
    Returns:
        (문서 인덱스, 유사도 점수) 튜플 리스트
    """
    similarities = []
    for idx, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((idx, sim))
    
    # 유사도 내림차순 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


# ============================================================
# Part 2: BM25 검색 구현
# ============================================================

class BM25:
    """
    BM25 검색 알고리즘 구현
    
    BM25는 TF-IDF의 개선 버전으로, 다음 요소를 고려합니다:
    - 단어 빈도 (Term Frequency)
    - 역문서 빈도 (Inverse Document Frequency)
    - 문서 길이 정규화
    """
    
    def __init__(self, corpus: List[str], k1: float = 1.2, b: float = 0.75):
        """
        Args:
            corpus: 문서 리스트
            k1: 포화 파라미터 (1.2 ~ 2.0 권장)
            b: 문서 길이 정규화 파라미터 (0.75 권장)
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        
        # 간단한 토크나이저 (공백 기반)
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.n_docs = len(self.tokenized_corpus)
        self.avg_doc_len = sum(len(doc) for doc in self.tokenized_corpus) / self.n_docs
        
        # IDF와 TF 계산
        self.idf = self._calculate_idf()
        self.term_freqs = self._calculate_term_freqs()
    
    def _calculate_idf(self) -> Dict[str, float]:
        """IDF (역문서 빈도) 계산"""
        idf = defaultdict(float)
        
        # 각 단어의 문서 빈도 계산
        for doc in self.tokenized_corpus:
            for token in set(doc):
                idf[token] += 1
        
        # IDF 계산: log((N - df + 0.5) / (df + 0.5) + 1)
        for token, doc_freq in idf.items():
            idf[token] = math.log(
                ((self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1
            )
        
        return dict(idf)
    
    def _calculate_term_freqs(self) -> List[Dict[str, int]]:
        """각 문서별 단어 빈도 계산"""
        term_freqs = []
        for doc in self.tokenized_corpus:
            tf = defaultdict(int)
            for token in doc:
                tf[token] += 1
            term_freqs.append(dict(tf))
        return term_freqs
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        쿼리에 대한 모든 문서의 BM25 점수 계산
        
        BM25 공식:
        score(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
        """
        query_tokens = query.lower().split()
        scores = np.zeros(self.n_docs)
        
        for i, (doc_tf, doc_tokens) in enumerate(zip(self.term_freqs, self.tokenized_corpus)):
            doc_len = len(doc_tokens)
            score = 0.0
            
            for q_token in query_tokens:
                if q_token not in self.idf:
                    continue
                
                idf = self.idf[q_token]
                freq = doc_tf.get(q_token, 0)
                
                # BM25 점수 계산
                numerator = idf * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += numerator / denominator
            
            scores[i] = score
        
        return scores
    
    def get_top_k(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """상위 k개 문서 반환"""
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [(idx, scores[idx]) for idx in top_k_indices]


# ============================================================
# Part 3: Reciprocal Rank Fusion (RRF)
# ============================================================

def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion으로 여러 검색 결과 통합
    
    RRF 공식: RRF(d) = Σ 1 / (k + rank(d))
    
    Args:
        rankings: 각 검색 방식의 문서 인덱스 순위 리스트
                  예: [[2, 0, 1], [0, 2, 1]] - 두 검색 결과
        k: 상수 (기본값 60)
    
    Returns:
        통합 점수로 정렬된 (문서인덱스, 점수) 리스트
    """
    rrf_scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):  # 1부터 시작
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    # 점수 내림차순 정렬
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# Part 4: 하이브리드 검색
# ============================================================

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    doc_index: int
    document: str
    score: float
    method: str


class HybridSearcher:
    """
    Dense Search와 BM25를 결합한 하이브리드 검색기
    """
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.bm25 = BM25(documents)
        
        # 간단한 임베딩 (실제로는 Sentence-Transformers 사용)
        # 여기서는 데모용으로 랜덤 벡터 사용
        np.random.seed(42)
        self.doc_embeddings = np.random.randn(len(documents), 128)
        # 정규화
        self.doc_embeddings = self.doc_embeddings / np.linalg.norm(
            self.doc_embeddings, axis=1, keepdims=True
        )
    
    def dense_search(self, query: str, k: int = 10) -> List[int]:
        """벡터 검색 (데모용 랜덤 쿼리 벡터)"""
        np.random.seed(hash(query) % 2**31)
        query_embedding = np.random.randn(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = simple_dense_search(query_embedding, self.doc_embeddings, k)
        return [idx for idx, _ in results]
    
    def sparse_search(self, query: str, k: int = 10) -> List[int]:
        """BM25 검색"""
        results = self.bm25.get_top_k(query, k)
        return [idx for idx, _ in results]
    
    def hybrid_search(self, query: str, k: int = 10, rrf_k: int = 60) -> List[SearchResult]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            rrf_k: RRF 상수
        
        Returns:
            SearchResult 리스트
        """
        # 두 검색 방식 수행
        dense_ranking = self.dense_search(query, k=k)
        sparse_ranking = self.sparse_search(query, k=k)
        
        # RRF로 통합
        fused_results = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=rrf_k)
        
        # SearchResult 생성
        results = []
        for doc_idx, score in fused_results[:k]:
            results.append(SearchResult(
                doc_index=doc_idx,
                document=self.documents[doc_idx],
                score=score,
                method="hybrid"
            ))
        
        return results


# ============================================================
# Part 5: 데모 실행
# ============================================================

def demo_bm25():
    """BM25 검색 데모"""
    print("\n" + "="*60)
    print("📊 BM25 검색 데모")
    print("="*60)
    
    documents = [
        "올해 여름 장마가 17일 제주도에서 시작됐다 서울 중부지방은 예년보다 늦다",
        "갤럭시 S5 언제 발매한다는 건지 27일 판매한다고 했다가 26일 판매한다",
        "로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 연구했다",
        "프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스터로 갔다",
        "인구 비율당 노벨상을 가장 많이 받은 나라"
    ]
    
    bm25 = BM25(documents)
    
    queries = [
        "비가 언제 올까",
        "로버트 헨리 딕 연구"
    ]
    
    for query in queries:
        print(f"\n🔍 쿼리: '{query}'")
        results = bm25.get_top_k(query, k=3)
        for rank, (idx, score) in enumerate(results, 1):
            print(f"  {rank}. [점수: {score:.4f}] {documents[idx][:50]}...")


def demo_rrf():
    """Reciprocal Rank Fusion 데모"""
    print("\n" + "="*60)
    print("🔗 Reciprocal Rank Fusion 데모")
    print("="*60)
    
    # 두 검색 방식의 결과 (문서 인덱스 순위)
    dense_ranking = [1, 4, 3, 5, 6]  # 벡터 검색: 1번 문서가 1위
    sparse_ranking = [2, 1, 3, 6, 4]  # BM25 검색: 2번 문서가 1위
    
    print(f"Dense 검색 순위: {dense_ranking}")
    print(f"Sparse 검색 순위: {sparse_ranking}")
    
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=5)
    
    print("\n통합 결과 (RRF):")
    for doc_id, score in fused:
        print(f"  문서 {doc_id}: RRF 점수 = {score:.6f}")


def demo_hybrid():
    """하이브리드 검색 데모"""
    print("\n" + "="*60)
    print("🔀 하이브리드 검색 데모")
    print("="*60)
    
    documents = [
        "올해 여름 장마가 시작됐다. 비가 많이 올 예정이다.",
        "로버트 헨리 딕이 1946년에 매사추세츠에서 연구했다.",
        "갤럭시 S5가 27일에 발매된다.",
        "프린스턴 대학교에서 학위를 받았다.",
        "인공지능 기술이 빠르게 발전하고 있다."
    ]
    
    searcher = HybridSearcher(documents)
    
    query = "비가 언제 올까"
    print(f"\n🔍 쿼리: '{query}'")
    
    # 각 검색 방식 결과 출력
    dense_results = searcher.dense_search(query, k=3)
    sparse_results = searcher.sparse_search(query, k=3)
    hybrid_results = searcher.hybrid_search(query, k=3)
    
    print("\n[Dense 검색 결과]")
    for rank, idx in enumerate(dense_results, 1):
        print(f"  {rank}. {documents[idx][:40]}...")
    
    print("\n[BM25 검색 결과]")
    for rank, idx in enumerate(sparse_results, 1):
        print(f"  {rank}. {documents[idx][:40]}...")
    
    print("\n[하이브리드 검색 결과]")
    for rank, result in enumerate(hybrid_results, 1):
        print(f"  {rank}. [점수: {result.score:.6f}] {result.document[:40]}...")


def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 10: RAG와 하이브리드 검색 실습")
    print("="*60)
    
    demo_bm25()
    demo_rrf()
    demo_hybrid()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
