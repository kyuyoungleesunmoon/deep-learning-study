# 📖 Chapter 10: 검색 증강 생성 (RAG)과 하이브리드 검색

## 📋 개요

이 챕터에서는 RAG(Retrieval-Augmented Generation)의 핵심 검색 기법들을 학습합니다.
- 벡터 기반 밀집 검색 (Dense Vector Search)
- BM25 키워드 기반 희소 검색 (Sparse Search)
- 두 방식을 결합한 하이브리드 검색 (Hybrid Search)

## 🔬 핵심 알고리즘

### 1. Dense Vector Search (밀집 벡터 검색)

**원리**: 문장을 고차원 벡터로 변환하고, 코사인 유사도로 의미적 유사성을 측정합니다.

```
similarity(q, d) = cos(θ) = (q · d) / (||q|| × ||d||)
```

**장점**:
- 의미적 유사성 포착 (동의어, 패러프레이즈)
- "비가 온다" ↔ "장마가 시작됐다" 연결 가능

**단점**:
- 고유명사, 숫자 등 정확한 매칭에 약함

### 2. BM25 (Best Match 25)

**원리**: TF-IDF의 개선 버전으로, 문서 내 단어 빈도와 희소성을 고려합니다.

```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

**수식 설명**:
- `IDF(qi)`: 단어 qi의 역문서 빈도 (드문 단어일수록 높음)
- `f(qi, D)`: 문서 D에서 단어 qi의 출현 빈도
- `k1`: 포화 파라미터 (보통 1.2~2.0)
- `b`: 문서 길이 정규화 파라미터 (보통 0.75)
- `avgdl`: 전체 문서의 평균 길이

**장점**:
- 정확한 키워드 매칭
- "로버트 헨리 딕" 같은 고유명사 검색에 강함

**단점**:
- 의미적 유사성 포착 불가
- "비" ↔ "장마" 연결 어려움

### 3. Reciprocal Rank Fusion (RRF)

**원리**: 여러 검색 결과의 순위를 통합하여 최종 랭킹을 생성합니다.

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

**수식 설명**:
- `d`: 문서
- `k`: 상수 (보통 60)
- `rank_i(d)`: i번째 검색에서 문서 d의 순위

**장점**:
- 다양한 검색 방식의 장점 결합
- 점수 스케일이 다른 검색 결과도 통합 가능

## 📊 실습 예제

### 예제 1: 문장 임베딩으로 벡터 검색

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 한국어 임베딩 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 예시 문서
documents = [
    "올해 여름 장마가 시작됐다",
    "프린스턴 대학교에서 학위를 받았다",
    "갤럭시 S5가 출시됐다"
]

# 문서 임베딩
doc_embeddings = model.encode(documents)

# FAISS 인덱스 생성
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))

# 쿼리 검색
query = "비가 많이 올 시기는?"
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding.astype('float32'), k=3)

print("검색 결과:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. {documents[idx]} (거리: {dist:.4f})")
```

### 예제 2: BM25 검색 구현

```python
import math
from collections import defaultdict
from transformers import AutoTokenizer

class SimpleBM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized_docs = [
            self.tokenizer.tokenize(doc) for doc in documents
        ]
        self.avg_doc_len = sum(len(d) for d in self.tokenized_docs) / len(self.tokenized_docs)
        self.idf = self._compute_idf()
    
    def _compute_idf(self):
        idf = defaultdict(float)
        N = len(self.tokenized_docs)
        
        # 각 단어의 문서 빈도 계산
        df = defaultdict(int)
        for doc in self.tokenized_docs:
            for token in set(doc):
                df[token] += 1
        
        # IDF 계산
        for token, freq in df.items():
            idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        
        return idf
    
    def score(self, query):
        query_tokens = self.tokenizer.tokenize(query)
        scores = []
        
        for doc in self.tokenized_docs:
            score = 0
            doc_len = len(doc)
            
            # 단어 빈도 계산
            tf = defaultdict(int)
            for token in doc:
                tf[token] += 1
            
            for token in query_tokens:
                if token in tf:
                    freq = tf[token]
                    numerator = self.idf[token] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    score += numerator / denominator
            
            scores.append(score)
        
        return scores

# 사용 예시
documents = ["로버트 헨리 딕이 1946년에 연구했다", "2023년 AI 기술이 발전했다", "프린스턴 대학교"]
bm25 = SimpleBM25(documents)
query = "로버트 헨리 딕 연구"
scores = bm25.score(query)
print(f"BM25 점수: {scores}")
```

### 예제 3: 하이브리드 검색

```python
def reciprocal_rank_fusion(rankings, k=60):
    """
    여러 검색 결과의 순위를 RRF로 통합
    
    Args:
        rankings: 각 검색 방식의 문서 인덱스 순위 리스트
        k: 상수 (기본값 60)
    
    Returns:
        통합 점수로 정렬된 (문서인덱스, 점수) 리스트
    """
    rrf_scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query, dense_ranking, sparse_ranking, k=60):
    """
    Dense(벡터)와 Sparse(BM25) 검색 결과 통합
    """
    results = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=k)
    return results

# 사용 예시
# 벡터 검색 결과 (문서 인덱스): [2, 0, 1] (2번 문서가 1위)
# BM25 검색 결과 (문서 인덱스): [0, 2, 1] (0번 문서가 1위)
dense_ranking = [2, 0, 1]
sparse_ranking = [0, 2, 1]

final_ranking = hybrid_search("검색 쿼리", dense_ranking, sparse_ranking)
print(f"하이브리드 검색 결과: {final_ranking}")
```

## 🎯 핵심 포인트

1. **Dense Search**: 의미적 유사성에 강함 → 질문-답변, 패러프레이즈 검색
2. **BM25**: 정확한 키워드 매칭에 강함 → 고유명사, 특정 용어 검색
3. **Hybrid Search**: 두 방식의 장점을 결합 → 실무에서 가장 효과적

## 📚 참고 자료

- 원본 코드: https://github.com/onlybooks/llm/tree/main/10장
- FAISS 문서: https://faiss.ai/
- Sentence-Transformers: https://www.sbert.net/
