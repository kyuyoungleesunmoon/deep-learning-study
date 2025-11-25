# 📖 Chapter 11: 문장 임베딩 (Sentence Embeddings)

## 📋 개요

이 챕터에서는 문장을 의미있는 벡터로 변환하는 방법을 학습합니다.
- Sentence-Transformers를 활용한 문장 임베딩
- Mean Pooling 기법
- 한국어 모델 활용 (KR-SBERT)

## 🔬 핵심 알고리즘

### 1. Sentence-BERT (SBERT)

**원리**: BERT의 출력을 Pooling하여 고정 크기 문장 벡터를 생성합니다.

```
[CLS] 문장 [SEP] → BERT → 토큰 벡터들 → Mean Pooling → 문장 벡터
```

**기존 BERT의 한계**:
- 문장 유사도 계산 시 모든 쌍을 Cross-Encoder로 비교 → O(n²) 복잡도
- 10,000개 문장 비교 시 약 65시간 소요

**SBERT의 해결책**:
- 각 문장을 독립적으로 인코딩 → O(n) 복잡도
- 코사인 유사도로 빠른 비교 가능
- 10,000개 문장 비교 시 약 5초 소요

### 2. Mean Pooling

**원리**: BERT 출력의 모든 토큰 벡터를 평균하여 문장 벡터 생성

```python
# Attention Mask를 고려한 Mean Pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    
    # Attention Mask 확장
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    
    # 마스킹된 토큰은 제외하고 평균
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask
```

**Pooling 방법 비교**:
| 방법 | 설명 | 성능 |
|------|------|------|
| [CLS] 토큰 | 첫 번째 토큰만 사용 | 보통 |
| Mean Pooling | 모든 토큰 평균 | **가장 좋음** |
| Max Pooling | 각 차원의 최댓값 | 보통 |

### 3. Contrastive Learning

**원리**: 유사한 문장은 가깝게, 다른 문장은 멀게 학습

**손실 함수 (Contrastive Loss)**:
```
L = (1-y) × ½ × D² + y × ½ × max(0, margin - D)²
```

- `y = 0`: 유사한 쌍 → 거리 D 최소화
- `y = 1`: 다른 쌍 → 거리가 margin보다 크도록

**Triplet Loss**:
```
L = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
```

- anchor와 positive는 가깝게
- anchor와 negative는 멀게

## 📊 한국어 문장 임베딩 모델

### 추천 모델

| 모델 | 설명 | 용도 |
|------|------|------|
| `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | 40K 어휘, KLUE 데이터 학습 | 범용 |
| `jhgan/ko-sbert-nli` | NLI 데이터 학습 | 문장 유사도 |
| `BM-K/KoSimCSE-roberta` | SimCSE 기법 적용 | 문장 유사도 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 다국어 지원 | 다국어 환경 |

## 📊 실습 예제

### 예제 1: Sentence-Transformers로 문장 임베딩

```python
from sentence_transformers import SentenceTransformer

# 한국어 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

sentences = [
    "오늘 날씨가 좋습니다.",
    "오늘 하늘이 맑아요.",
    "프로그래밍을 배우고 있습니다."
]

# 문장 임베딩 생성
embeddings = model.encode(sentences)
print(f"임베딩 크기: {embeddings.shape}")  # (3, 768)

# 문장 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)
print(f"유사도 행렬:\n{similarity_matrix}")
```

### 예제 2: 커스텀 Pooling Layer 추가

```python
from sentence_transformers import SentenceTransformer, models

# Transformer 모델 로드
transformer = models.Transformer('klue/roberta-base')

# Mean Pooling 레이어 추가
pooling = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,  # Mean Pooling 사용
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# 모델 조립
model = SentenceTransformer(modules=[transformer, pooling])

# 문장 임베딩 생성
sentences = ["안녕하세요", "반갑습니다"]
embeddings = model.encode(sentences)
```

### 예제 3: KLUE STS 데이터셋으로 유사도 측정

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

# 데이터셋 로드
dataset = load_dataset('klue', 'sts', split='validation')

# 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 예측 및 평가
predictions = []
labels = []

for item in dataset:
    # 두 문장 임베딩
    emb1 = model.encode(item['sentence1'])
    emb2 = model.encode(item['sentence2'])
    
    # 코사인 유사도 계산
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    predictions.append(similarity)
    labels.append(item['labels']['label'] / 5.0)  # 0-5 → 0-1 정규화

# Spearman 상관계수
correlation, _ = spearmanr(predictions, labels)
print(f"Spearman Correlation: {correlation:.4f}")
```

### 예제 4: 문장 임베딩으로 유사 문장 검색

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 문서 데이터베이스
documents = [
    "인공지능은 미래 기술의 핵심입니다.",
    "머신러닝은 데이터에서 패턴을 학습합니다.",
    "딥러닝은 심층 신경망을 사용합니다.",
    "오늘 점심 메뉴는 김치찌개입니다.",
    "파이썬은 인기있는 프로그래밍 언어입니다."
]

# 문서 임베딩 (오프라인에서 미리 계산)
doc_embeddings = model.encode(documents)

def search(query, top_k=3):
    """쿼리와 유사한 문서 검색"""
    query_embedding = model.encode([query])[0]
    
    # 코사인 유사도 계산
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # 상위 k개 반환
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx]
        })
    
    return results

# 검색 테스트
query = "AI 기술에 대해 알려주세요"
results = search(query)

print(f"쿼리: '{query}'")
for i, result in enumerate(results, 1):
    print(f"{i}. [{result['similarity']:.4f}] {result['document']}")
```

## 🎯 핵심 포인트

1. **Mean Pooling이 가장 효과적**: [CLS] 토큰보다 모든 토큰 평균이 더 좋은 표현
2. **Contrastive Learning**: 유사한 쌍은 가깝게, 다른 쌍은 멀게 학습
3. **한국어 전용 모델 사용**: 영어 모델보다 한국어 전용 모델이 성능 좋음
4. **정규화 필수**: 코사인 유사도 사용 시 벡터 정규화로 일관된 스케일 유지

## 📚 참고 자료

- 원본 코드: https://github.com/onlybooks/llm/tree/main/11장
- Sentence-Transformers: https://www.sbert.net/
- KLUE 벤치마크: https://klue-benchmark.com/
