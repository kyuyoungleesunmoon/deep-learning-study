"""
Chapter 11: 문장 임베딩 실습 코드
=================================

이 파일은 문장 임베딩(Sentence Embeddings)을 실습합니다:
1. 직접 구현한 Mean Pooling
2. 문장 유사도 계산
3. 유사 문장 검색
4. (선택) Sentence-Transformers 활용

실행 방법:
    pip install numpy torch transformers
    python chapter_11_practice.py

    # Sentence-Transformers 사용 시:
    pip install sentence-transformers
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


# ============================================================
# Part 1: 기본 벡터 연산
# ============================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간의 코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간의 유클리드 거리 계산"""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """벡터 정규화 (L2 norm = 1)"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # 0으로 나누기 방지
    return vectors / norms


# ============================================================
# Part 2: Mean Pooling 구현
# ============================================================

def mean_pooling_simple(token_embeddings: np.ndarray, 
                        attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean Pooling 구현 (NumPy 버전)
    
    Args:
        token_embeddings: (batch_size, seq_len, hidden_dim) 토큰 임베딩
        attention_mask: (batch_size, seq_len) 어텐션 마스크 (1=유효, 0=패딩)
    
    Returns:
        (batch_size, hidden_dim) 문장 임베딩
    """
    # Attention mask 확장: (batch, seq_len) -> (batch, seq_len, hidden)
    input_mask_expanded = attention_mask[:, :, np.newaxis]
    
    # 마스킹된 토큰 제외하고 합계
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    
    # 유효 토큰 수로 나누기
    sum_mask = np.sum(input_mask_expanded, axis=1)
    sum_mask = np.maximum(sum_mask, 1e-9)  # 0으로 나누기 방지
    
    return sum_embeddings / sum_mask


def demo_mean_pooling():
    """Mean Pooling 데모"""
    print("\n" + "="*60)
    print("📊 Mean Pooling 데모")
    print("="*60)
    
    # 가상의 토큰 임베딩 (2개 문장, 5개 토큰, 4차원)
    # 문장 1: "안녕하세요" → 실제 토큰 3개 + 패딩 2개
    # 문장 2: "반갑습니다 오늘" → 실제 토큰 4개 + 패딩 1개
    
    np.random.seed(42)
    token_embeddings = np.random.randn(2, 5, 4)
    
    attention_mask = np.array([
        [1, 1, 1, 0, 0],  # 문장 1: 3개 토큰 유효
        [1, 1, 1, 1, 0]   # 문장 2: 4개 토큰 유효
    ])
    
    print("토큰 임베딩 shape:", token_embeddings.shape)
    print("어텐션 마스크:\n", attention_mask)
    
    # Mean Pooling 적용
    sentence_embeddings = mean_pooling_simple(token_embeddings, attention_mask)
    print("\n문장 임베딩 shape:", sentence_embeddings.shape)
    print("문장 임베딩:\n", sentence_embeddings)
    
    # 수동 계산으로 검증 (문장 1)
    manual_mean = np.mean(token_embeddings[0, :3, :], axis=0)
    print("\n수동 계산 (문장 1):", manual_mean)
    print("Mean Pooling 결과:", sentence_embeddings[0])
    print("일치 여부:", np.allclose(manual_mean, sentence_embeddings[0]))


# ============================================================
# Part 3: 문장 유사도 계산
# ============================================================

@dataclass
class SentencePair:
    """문장 쌍 데이터"""
    sentence1: str
    sentence2: str
    similarity: float


class SimpleSentenceEncoder:
    """
    간단한 문장 인코더 (데모용)
    
    실제로는 Sentence-Transformers를 사용해야 합니다.
    여기서는 단어 임베딩의 평균을 사용합니다.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.word_embeddings = {}
        np.random.seed(42)
    
    def _get_word_embedding(self, word: str) -> np.ndarray:
        """단어 임베딩 반환 (해시 기반 랜덤 생성)"""
        if word not in self.word_embeddings:
            # 단어별 일관된 임베딩 생성
            np.random.seed(hash(word) % 2**31)
            embedding = np.random.randn(self.embedding_dim)
            self.word_embeddings[word] = embedding / np.linalg.norm(embedding)
        return self.word_embeddings[word]
    
    def encode(self, sentences: List[str]) -> np.ndarray:
        """문장들을 벡터로 인코딩"""
        embeddings = []
        
        for sentence in sentences:
            words = sentence.lower().split()
            if not words:
                embeddings.append(np.zeros(self.embedding_dim))
                continue
            
            word_embs = [self._get_word_embedding(w) for w in words]
            sentence_emb = np.mean(word_embs, axis=0)
            embeddings.append(sentence_emb)
        
        return np.array(embeddings)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """임베딩들 간의 유사도 행렬 계산"""
    # 정규화
    normalized = normalize_vectors(embeddings)
    # 코사인 유사도 = 정규화된 벡터의 내적
    return np.dot(normalized, normalized.T)


def demo_similarity():
    """문장 유사도 데모"""
    print("\n" + "="*60)
    print("🔍 문장 유사도 계산 데모")
    print("="*60)
    
    encoder = SimpleSentenceEncoder()
    
    sentences = [
        "오늘 날씨가 좋습니다",
        "오늘 하늘이 맑아요",
        "프로그래밍을 공부합니다",
        "코딩을 배우고 있습니다"
    ]
    
    print("문장 목록:")
    for i, s in enumerate(sentences):
        print(f"  {i}: {s}")
    
    # 임베딩 생성
    embeddings = encoder.encode(sentences)
    print(f"\n임베딩 shape: {embeddings.shape}")
    
    # 유사도 행렬 계산
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    print("\n유사도 행렬:")
    print("     ", end="")
    for i in range(len(sentences)):
        print(f"  [{i}]  ", end="")
    print()
    
    for i, row in enumerate(similarity_matrix):
        print(f"[{i}]", end="")
        for val in row:
            print(f"  {val:5.3f}", end="")
        print()
    
    # 가장 유사한 쌍 찾기
    print("\n가장 유사한 문장 쌍:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = similarity_matrix[i][j]
            if sim > 0.5:
                print(f"  '{sentences[i]}' ↔ '{sentences[j]}': {sim:.3f}")


# ============================================================
# Part 4: 유사 문장 검색
# ============================================================

class SentenceSearchEngine:
    """문장 검색 엔진"""
    
    def __init__(self, encoder: SimpleSentenceEncoder):
        self.encoder = encoder
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: List[str]):
        """문서 추가"""
        self.documents = documents
        self.embeddings = self.encoder.encode(documents)
        # 정규화
        self.embeddings = normalize_vectors(self.embeddings)
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """쿼리와 유사한 문서 검색"""
        query_embedding = self.encoder.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 코사인 유사도 계산
        similarities = np.dot(self.embeddings, query_embedding)
        
        # 상위 k개 인덱스
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], similarities[idx]))
        
        return results


def demo_search():
    """유사 문장 검색 데모"""
    print("\n" + "="*60)
    print("🔎 유사 문장 검색 데모")
    print("="*60)
    
    documents = [
        "인공지능은 미래 기술의 핵심입니다",
        "머신러닝은 데이터에서 패턴을 학습합니다",
        "딥러닝은 심층 신경망을 사용합니다",
        "오늘 점심 메뉴는 김치찌개입니다",
        "파이썬은 인기있는 프로그래밍 언어입니다",
        "자연어 처리는 텍스트를 분석합니다",
        "컴퓨터 비전은 이미지를 이해합니다"
    ]
    
    encoder = SimpleSentenceEncoder()
    search_engine = SentenceSearchEngine(encoder)
    search_engine.add_documents(documents)
    
    queries = [
        "AI 기술",
        "음식 메뉴",
        "이미지 분석"
    ]
    
    for query in queries:
        print(f"\n🔍 쿼리: '{query}'")
        results = search_engine.search(query, top_k=3)
        
        for rank, (doc, score) in enumerate(results, 1):
            print(f"  {rank}. [{score:.3f}] {doc}")


# ============================================================
# Part 5: Spearman 상관계수 계산
# ============================================================

def spearman_correlation(predictions: List[float], labels: List[float]) -> float:
    """
    Spearman 상관계수 계산
    
    문장 유사도 모델 평가에 사용됩니다.
    """
    n = len(predictions)
    
    # 순위 계산
    pred_ranks = np.argsort(np.argsort(predictions)) + 1
    label_ranks = np.argsort(np.argsort(labels)) + 1
    
    # 순위 차이
    d = pred_ranks - label_ranks
    d_squared = np.sum(d ** 2)
    
    # Spearman 상관계수
    correlation = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
    return correlation


def demo_evaluation():
    """모델 평가 데모"""
    print("\n" + "="*60)
    print("📈 모델 평가 (Spearman 상관계수) 데모")
    print("="*60)
    
    # 가상의 예측값과 실제값
    predictions = [0.9, 0.7, 0.3, 0.8, 0.2, 0.6]
    labels = [0.85, 0.75, 0.25, 0.9, 0.15, 0.65]
    
    print("예측값:", [f"{p:.2f}" for p in predictions])
    print("실제값:", [f"{l:.2f}" for l in labels])
    
    correlation = spearman_correlation(predictions, labels)
    print(f"\nSpearman 상관계수: {correlation:.4f}")
    
    if correlation > 0.8:
        print("평가: 매우 좋음 ✅")
    elif correlation > 0.6:
        print("평가: 좋음 👍")
    elif correlation > 0.4:
        print("평가: 보통 ⚠️")
    else:
        print("평가: 개선 필요 ❌")


# ============================================================
# Part 6: Sentence-Transformers 사용 (선택적)
# ============================================================

def demo_sentence_transformers():
    """Sentence-Transformers 사용 데모 (설치된 경우)"""
    try:
        from sentence_transformers import SentenceTransformer
        
        print("\n" + "="*60)
        print("🚀 Sentence-Transformers 데모")
        print("="*60)
        
        # 다국어 모델 로드 (한국어 지원)
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        sentences = [
            "오늘 날씨가 좋습니다",
            "오늘 하늘이 맑아요",
            "프로그래밍을 공부합니다"
        ]
        
        print("문장:")
        for s in sentences:
            print(f"  - {s}")
        
        # 임베딩 생성
        embeddings = model.encode(sentences)
        print(f"\n임베딩 shape: {embeddings.shape}")
        
        # 유사도 계산
        from sentence_transformers.util import cos_sim
        similarity = cos_sim(embeddings, embeddings)
        
        print("\n유사도 행렬:")
        print(similarity.numpy())
        
    except ImportError:
        print("\n⚠️ sentence-transformers가 설치되지 않았습니다.")
        print("설치: pip install sentence-transformers")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 11: 문장 임베딩 실습")
    print("="*60)
    
    demo_mean_pooling()
    demo_similarity()
    demo_search()
    demo_evaluation()
    demo_sentence_transformers()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
