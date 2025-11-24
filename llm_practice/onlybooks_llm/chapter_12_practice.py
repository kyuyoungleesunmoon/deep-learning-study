"""
Chapter 12: 벡터 데이터베이스 실습 코드
=======================================

이 파일은 벡터 데이터베이스의 핵심 개념을 실습합니다:
1. Brute-force 검색 구현
2. IVF (Inverted File Index) 개념
3. Product Quantization 원리
4. (선택) FAISS 사용

실행 방법:
    pip install numpy
    python chapter_12_practice.py

    # FAISS 사용 시:
    pip install faiss-cpu
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import time
from dataclasses import dataclass


# ============================================================
# Part 1: 기본 벡터 검색
# ============================================================

def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """L2 (유클리드) 거리 계산"""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def brute_force_search(query: np.ndarray, 
                       database: np.ndarray, 
                       k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute-force 최근접 이웃 검색
    
    Args:
        query: (d,) 쿼리 벡터
        database: (n, d) 데이터베이스 벡터들
        k: 반환할 이웃 수
    
    Returns:
        distances: (k,) 거리
        indices: (k,) 인덱스
    """
    # 모든 벡터와의 거리 계산
    distances = np.linalg.norm(database - query, axis=1)
    
    # 상위 k개 인덱스
    indices = np.argsort(distances)[:k]
    
    return distances[indices], indices


def demo_brute_force():
    """Brute-force 검색 데모"""
    print("\n" + "="*60)
    print("🔍 Brute-force 검색 데모")
    print("="*60)
    
    np.random.seed(42)
    n, d = 1000, 64
    database = np.random.randn(n, d).astype('float32')
    query = np.random.randn(d).astype('float32')
    
    start = time.time()
    distances, indices = brute_force_search(query, database, k=5)
    elapsed = time.time() - start
    
    print(f"데이터베이스 크기: {n} 벡터 x {d} 차원")
    print(f"검색 시간: {elapsed*1000:.2f} ms")
    print(f"상위 5개 인덱스: {indices}")
    print(f"거리: {distances}")


# ============================================================
# Part 2: IVF (Inverted File Index) 구현
# ============================================================

class SimpleIVFIndex:
    """
    간단한 IVF 인덱스 구현
    
    1. 학습: K-means로 클러스터링
    2. 추가: 각 벡터를 가장 가까운 클러스터에 할당
    3. 검색: 가장 가까운 nprobe개 클러스터만 탐색
    """
    
    def __init__(self, d: int, nlist: int = 10):
        """
        Args:
            d: 벡터 차원
            nlist: 클러스터 수
        """
        self.d = d
        self.nlist = nlist
        self.centroids = None
        self.inverted_lists: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self.is_trained = False
        self.nprobe = 1  # 검색할 클러스터 수
    
    def train(self, data: np.ndarray, n_iter: int = 20):
        """
        K-means 클러스터링으로 centroids 학습
        
        Args:
            data: (n, d) 학습 데이터
            n_iter: K-means 반복 횟수
        """
        n = len(data)
        
        # 랜덤 초기화
        indices = np.random.choice(n, self.nlist, replace=False)
        self.centroids = data[indices].copy()
        
        for _ in range(n_iter):
            # 할당: 각 점을 가장 가까운 centroid에 할당
            distances = np.linalg.norm(
                data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], 
                axis=2
            )
            assignments = np.argmin(distances, axis=1)
            
            # 업데이트: 각 클러스터의 평균으로 centroid 업데이트
            for i in range(self.nlist):
                mask = assignments == i
                if np.sum(mask) > 0:
                    self.centroids[i] = np.mean(data[mask], axis=0)
        
        self.is_trained = True
        print(f"학습 완료: {self.nlist}개 클러스터")
    
    def add(self, data: np.ndarray):
        """
        데이터를 inverted list에 추가
        
        Args:
            data: (n, d) 추가할 데이터
        """
        if not self.is_trained:
            raise ValueError("인덱스가 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        # 각 벡터를 가장 가까운 클러스터에 할당
        distances = np.linalg.norm(
            data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], 
            axis=2
        )
        assignments = np.argmin(distances, axis=1)
        
        for idx, (vec, cluster_id) in enumerate(zip(data, assignments)):
            self.inverted_lists[cluster_id].append((idx, vec))
        
        print(f"추가 완료: {len(data)}개 벡터")
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        검색 수행
        
        Args:
            query: (d,) 쿼리 벡터
            k: 반환할 결과 수
        
        Returns:
            distances, indices
        """
        # 가장 가까운 nprobe개 클러스터 찾기
        centroid_distances = np.linalg.norm(self.centroids - query, axis=1)
        probe_clusters = np.argsort(centroid_distances)[:self.nprobe]
        
        # 해당 클러스터들의 벡터만 탐색
        candidates = []
        for cluster_id in probe_clusters:
            for idx, vec in self.inverted_lists[cluster_id]:
                dist = np.linalg.norm(vec - query)
                candidates.append((dist, idx))
        
        # 상위 k개 정렬
        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:k]
        
        if not top_k:
            return np.array([]), np.array([])
        
        distances = np.array([d for d, _ in top_k])
        indices = np.array([i for _, i in top_k])
        
        return distances, indices


def demo_ivf():
    """IVF 인덱스 데모"""
    print("\n" + "="*60)
    print("📊 IVF 인덱스 데모")
    print("="*60)
    
    np.random.seed(42)
    n, d = 10000, 64
    database = np.random.randn(n, d).astype('float32')
    query = np.random.randn(d).astype('float32')
    
    # IVF 인덱스 생성
    index = SimpleIVFIndex(d=d, nlist=100)
    
    # 학습
    start = time.time()
    index.train(database)
    train_time = time.time() - start
    
    # 추가
    start = time.time()
    index.add(database)
    add_time = time.time() - start
    
    # 검색 (nprobe 비교)
    print("\nnprobe에 따른 검색 성능:")
    
    # Ground truth (brute-force)
    gt_distances, gt_indices = brute_force_search(query, database, k=10)
    
    for nprobe in [1, 5, 10, 20]:
        index.nprobe = nprobe
        
        start = time.time()
        distances, indices = index.search(query, k=10)
        search_time = time.time() - start
        
        # Recall 계산
        recall = len(set(indices) & set(gt_indices)) / len(gt_indices) * 100
        
        print(f"  nprobe={nprobe:2d}: 검색 {search_time*1000:.2f}ms, Recall@10 = {recall:.1f}%")


# ============================================================
# Part 3: Product Quantization 개념
# ============================================================

class SimpleProductQuantizer:
    """
    간단한 Product Quantization 구현
    
    원리:
    1. 벡터를 m개의 서브벡터로 분할
    2. 각 서브벡터에 대해 k-means (ksub 클러스터)
    3. 각 서브벡터를 가장 가까운 클러스터 ID로 표현
    
    메모리 절감:
    - 원래: d * 4 bytes (float32)
    - PQ 후: m * 1 byte (uint8, ksub=256인 경우)
    """
    
    def __init__(self, d: int, m: int = 8, ksub: int = 256):
        """
        Args:
            d: 벡터 차원
            m: 서브벡터 수 (d는 m으로 나누어 떨어져야 함)
            ksub: 각 서브벡터의 클러스터 수
        """
        assert d % m == 0, f"d({d})는 m({m})으로 나누어 떨어져야 합니다"
        
        self.d = d
        self.m = m
        self.ksub = ksub
        self.dsub = d // m  # 서브벡터 차원
        self.codebooks = None
        self.is_trained = False
    
    def train(self, data: np.ndarray, n_iter: int = 20):
        """
        각 서브공간에 대해 k-means 학습
        """
        n = len(data)
        self.codebooks = []
        
        for i in range(self.m):
            # i번째 서브벡터 추출
            subvectors = data[:, i * self.dsub : (i + 1) * self.dsub]
            
            # k-means 클러스터링
            centroids = self._kmeans(subvectors, self.ksub, n_iter)
            self.codebooks.append(centroids)
        
        self.codebooks = np.array(self.codebooks)  # (m, ksub, dsub)
        self.is_trained = True
        print(f"PQ 학습 완료: {self.m}개 서브공간, {self.ksub}개 클러스터/서브공간")
    
    def _kmeans(self, data: np.ndarray, k: int, n_iter: int) -> np.ndarray:
        """간단한 k-means"""
        n = len(data)
        indices = np.random.choice(n, min(k, n), replace=False)
        centroids = data[indices].copy()
        
        if len(centroids) < k:
            # 데이터가 k보다 적으면 패딩
            padding = np.zeros((k - len(centroids), data.shape[1]))
            centroids = np.vstack([centroids, padding])
        
        for _ in range(n_iter):
            distances = np.linalg.norm(
                data[:, np.newaxis, :] - centroids[np.newaxis, :, :], 
                axis=2
            )
            assignments = np.argmin(distances, axis=1)
            
            for i in range(k):
                mask = assignments == i
                if np.sum(mask) > 0:
                    centroids[i] = np.mean(data[mask], axis=0)
        
        return centroids
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        벡터를 코드로 인코딩 (압축)
        
        Args:
            data: (n, d) 원본 벡터
        
        Returns:
            codes: (n, m) 코드 (각 원소는 0~ksub-1)
        """
        n = len(data)
        codes = np.zeros((n, self.m), dtype=np.uint8)
        
        for i in range(self.m):
            subvectors = data[:, i * self.dsub : (i + 1) * self.dsub]
            distances = np.linalg.norm(
                subvectors[:, np.newaxis, :] - self.codebooks[i][np.newaxis, :, :],
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        코드를 벡터로 디코딩 (복원)
        """
        n = len(codes)
        reconstructed = np.zeros((n, self.d))
        
        for i in range(self.m):
            reconstructed[:, i * self.dsub : (i + 1) * self.dsub] = \
                self.codebooks[i][codes[:, i]]
        
        return reconstructed


def demo_pq():
    """Product Quantization 데모"""
    print("\n" + "="*60)
    print("📦 Product Quantization 데모")
    print("="*60)
    
    np.random.seed(42)
    n, d = 10000, 128
    database = np.random.randn(n, d).astype('float32')
    
    # PQ 생성 및 학습
    m = 8  # 서브벡터 수
    pq = SimpleProductQuantizer(d=d, m=m, ksub=256)
    pq.train(database[:1000])  # 일부 데이터로 학습
    
    # 인코딩
    codes = pq.encode(database)
    
    # 메모리 비교
    original_size = n * d * 4  # float32
    compressed_size = n * m * 1  # uint8
    
    print(f"\n메모리 비교:")
    print(f"  원본: {original_size / 1e6:.2f} MB")
    print(f"  압축: {compressed_size / 1e6:.2f} MB")
    print(f"  압축률: {original_size / compressed_size:.0f}x")
    
    # 복원 오차
    reconstructed = pq.decode(codes)
    mse = np.mean((database - reconstructed) ** 2)
    print(f"\n복원 오차 (MSE): {mse:.4f}")


# ============================================================
# Part 4: FAISS 사용 (선택적)
# ============================================================

def demo_faiss():
    """FAISS 사용 데모"""
    try:
        import faiss
        
        print("\n" + "="*60)
        print("🚀 FAISS 데모")
        print("="*60)
        
        np.random.seed(42)
        n, d = 100000, 128
        database = np.random.randn(n, d).astype('float32')
        queries = np.random.randn(100, d).astype('float32')
        
        indexes = {
            'Flat': faiss.IndexFlatL2(d),
            'IVF': None,
            'HNSW': faiss.IndexHNSWFlat(d, 32)
        }
        
        # IVF 인덱스 생성
        nlist = 100
        quantizer = faiss.IndexFlatL2(d)
        indexes['IVF'] = faiss.IndexIVFFlat(quantizer, d, nlist)
        
        print(f"\n데이터 크기: {n:,} 벡터 x {d} 차원")
        
        for name, index in indexes.items():
            # 학습 (필요한 경우)
            if hasattr(index, 'train') and not index.is_trained:
                index.train(database)
            
            # 추가
            start = time.time()
            index.add(database)
            add_time = time.time() - start
            
            # 검색
            if name == 'IVF':
                index.nprobe = 10
            
            start = time.time()
            distances, indices = index.search(queries, k=10)
            search_time = time.time() - start
            
            print(f"\n{name}:")
            print(f"  추가 시간: {add_time:.3f}s")
            print(f"  검색 시간: {search_time*1000:.2f}ms ({len(queries)}개 쿼리)")
        
    except ImportError:
        print("\n⚠️ faiss가 설치되지 않았습니다.")
        print("설치: pip install faiss-cpu")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 12: 벡터 데이터베이스 실습")
    print("="*60)
    
    demo_brute_force()
    demo_ivf()
    demo_pq()
    demo_faiss()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
