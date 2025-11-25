# 📖 Chapter 12: 벡터 데이터베이스 (Vector Databases)

## 📋 개요

이 챕터에서는 대규모 벡터 데이터를 효율적으로 저장하고 검색하는 방법을 학습합니다.
- FAISS 인덱스 종류와 특성
- 양자화 기법 (Product Quantization)
- 근사 최근접 이웃 검색 (ANN)

## 🔬 핵심 알고리즘

### 1. 정확한 검색 vs 근사 검색

**정확한 검색 (Exact Search)**:
- 모든 벡터와 비교
- 시간 복잡도: O(n × d) (n: 문서 수, d: 차원)
- 100% 정확하지만 느림

**근사 검색 (Approximate Nearest Neighbor, ANN)**:
- 일부 벡터만 비교
- 시간 복잡도: O(log n) ~ O(√n)
- 약간의 정확도 손실, 매우 빠름

### 2. FAISS 인덱스 종류

| 인덱스 | 설명 | 메모리 | 속도 | 정확도 |
|--------|------|--------|------|--------|
| `IndexFlatL2` | Brute-force L2 | 높음 | 느림 | 100% |
| `IndexFlatIP` | Brute-force 내적 | 높음 | 느림 | 100% |
| `IndexIVFFlat` | IVF + Flat | 높음 | 빠름 | 높음 |
| `IndexIVFPQ` | IVF + PQ | **낮음** | 빠름 | 중간 |
| `IndexHNSWFlat` | HNSW 그래프 | 중간 | 매우 빠름 | 높음 |

### 3. IVF (Inverted File Index)

**원리**: 벡터 공간을 여러 클러스터로 분할하고, 검색 시 관련 클러스터만 탐색

```
1. 학습 단계: K-means로 nlist개의 클러스터 생성
2. 추가 단계: 각 벡터를 가장 가까운 클러스터에 할당
3. 검색 단계: 쿼리와 가장 가까운 nprobe개 클러스터만 탐색
```

**파라미터**:
- `nlist`: 클러스터 수 (보통 √n ~ 4√n)
- `nprobe`: 검색할 클러스터 수 (클수록 정확, 느림)

### 4. Product Quantization (PQ)

**원리**: 고차원 벡터를 여러 서브벡터로 나누고 각각을 양자화

```
768차원 벡터 → 8개의 96차원 서브벡터 → 각각 256개 클러스터로 양자화

원래 크기: 768 × 4 bytes = 3,072 bytes
PQ 후: 8 × 1 byte = 8 bytes (384배 압축!)
```

**장점**:
- 메모리 대폭 절감
- 코드북 기반 빠른 거리 계산

**단점**:
- 정확도 손실
- 학습 시간 필요

### 5. HNSW (Hierarchical Navigable Small World)

**원리**: 다층 그래프 구조로 효율적인 탐색

```
Layer 2:  [A] -------- [F]
           |            |
Layer 1:  [A] -- [C] -- [F]
           |    / \     |
Layer 0:  [A]-[B]-[C]-[D]-[E]-[F]
```

**특징**:
- 상위 레이어: 장거리 점프 (빠른 탐색)
- 하위 레이어: 지역 탐색 (정밀도)
- 검색 시간: O(log n)

## 📊 실습 예제

### 예제 1: FAISS 기본 사용

```python
import faiss
import numpy as np

# 데이터 생성
np.random.seed(42)
d = 128  # 벡터 차원
n = 10000  # 데이터 수
xb = np.random.randn(n, d).astype('float32')  # 데이터베이스
xq = np.random.randn(5, d).astype('float32')  # 쿼리

# 정확한 검색 (Brute-force)
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)

k = 4  # 상위 4개 검색
distances, indices = index_flat.search(xq, k)
print("Flat 인덱스 결과:")
print(f"  거리: {distances[0]}")
print(f"  인덱스: {indices[0]}")
```

### 예제 2: IVF 인덱스

```python
import faiss
import numpy as np

d = 128
n = 100000
xb = np.random.randn(n, d).astype('float32')

# IVF 인덱스 생성
nlist = 100  # 클러스터 수
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 학습 (클러스터링)
index.train(xb)

# 데이터 추가
index.add(xb)

# 검색 (nprobe 조절)
index.nprobe = 10  # 검색할 클러스터 수
distances, indices = index.search(xq, k=5)
```

### 예제 3: Product Quantization

```python
import faiss
import numpy as np

d = 128
n = 100000
xb = np.random.randn(n, d).astype('float32')

# IVF + PQ 인덱스
nlist = 100
m = 8  # 서브벡터 수 (d가 m으로 나누어 떨어져야 함)
nbits = 8  # 각 서브벡터당 비트 수 (2^8 = 256 클러스터)

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# 학습
index.train(xb)

# 추가
index.add(xb)

# 검색
index.nprobe = 10
distances, indices = index.search(xq, k=5)

# 메모리 비교
flat_memory = n * d * 4  # float32
pq_memory = n * m  # 각 벡터당 m bytes

print(f"Flat 메모리: {flat_memory / 1e6:.2f} MB")
print(f"PQ 메모리: {pq_memory / 1e6:.2f} MB")
print(f"압축률: {flat_memory / pq_memory:.0f}x")
```

### 예제 4: HNSW 인덱스

```python
import faiss
import numpy as np

d = 128
n = 100000
xb = np.random.randn(n, d).astype('float32')

# HNSW 인덱스
M = 32  # 각 노드의 연결 수
index = faiss.IndexHNSWFlat(d, M)

# efConstruction: 인덱스 구축 시 탐색 깊이
index.hnsw.efConstruction = 40

# 데이터 추가 (학습 불필요)
index.add(xb)

# efSearch: 검색 시 탐색 깊이
index.hnsw.efSearch = 16
distances, indices = index.search(xq, k=5)
```

### 예제 5: 성능 비교

```python
import faiss
import numpy as np
import time

def benchmark_index(index, xb, xq, name):
    """인덱스 성능 벤치마크"""
    # 추가 시간
    start = time.time()
    if hasattr(index, 'train'):
        index.train(xb)
    index.add(xb)
    add_time = time.time() - start
    
    # 검색 시간
    start = time.time()
    for _ in range(10):
        index.search(xq, k=10)
    search_time = (time.time() - start) / 10
    
    print(f"{name}:")
    print(f"  추가 시간: {add_time:.3f}s")
    print(f"  검색 시간: {search_time*1000:.2f}ms")

# 데이터 준비
d, n = 128, 100000
xb = np.random.randn(n, d).astype('float32')
xq = np.random.randn(100, d).astype('float32')

# Flat
benchmark_index(faiss.IndexFlatL2(d), xb.copy(), xq, "Flat")

# IVF
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
benchmark_index(index_ivf, xb.copy(), xq, "IVF")

# HNSW
benchmark_index(faiss.IndexHNSWFlat(d, 32), xb.copy(), xq, "HNSW")
```

## 🎯 핵심 포인트

1. **데이터 크기별 인덱스 선택**:
   - < 1만: `IndexFlatL2` (정확도 우선)
   - 1만 ~ 100만: `IndexIVFFlat` (균형)
   - > 100만: `IndexIVFPQ` (메모리 효율)

2. **nprobe 튜닝**: 정확도와 속도의 트레이드오프

3. **HNSW는 GPU 미지원**: CPU에서만 사용

4. **벡터 정규화**: 코사인 유사도 사용 시 벡터 정규화 후 `IndexFlatIP` 사용

## 📚 참고 자료

- 원본 코드: https://github.com/onlybooks/llm/tree/main/12장
- FAISS 문서: https://faiss.ai/
- FAISS 튜토리얼: https://github.com/facebookresearch/faiss/wiki
