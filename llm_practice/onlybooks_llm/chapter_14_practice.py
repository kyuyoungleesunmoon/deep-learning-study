"""
Chapter 14: 멀티모달 LLM 실습 코드
==================================

이 파일은 CLIP 등 멀티모달 모델의 원리를 실습합니다:
1. 이미지-텍스트 임베딩 개념
2. Contrastive Learning 원리
3. Zero-shot 분류 시뮬레이션
4. (선택) Hugging Face CLIP 사용

실행 방법:
    pip install numpy pillow
    python chapter_14_practice.py

    # CLIP 사용 시:
    pip install transformers torch pillow requests
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


# ============================================================
# Part 1: Contrastive Learning 원리
# ============================================================

def contrastive_loss(image_embeddings: np.ndarray, 
                     text_embeddings: np.ndarray, 
                     temperature: float = 0.07) -> float:
    """
    Contrastive Loss 계산 (InfoNCE Loss)
    
    Args:
        image_embeddings: (N, D) 이미지 임베딩
        text_embeddings: (N, D) 텍스트 임베딩 (같은 인덱스가 매칭 쌍)
        temperature: 온도 파라미터
    
    Returns:
        loss: 평균 contrastive loss
    """
    # 정규화
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # 유사도 행렬 계산
    logits = np.dot(image_embeddings, text_embeddings.T) / temperature
    
    N = len(image_embeddings)
    labels = np.arange(N)  # 대각선이 정답
    
    # Image-to-Text Loss
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # 수치 안정성
    i2t_loss = -np.log(exp_logits[np.arange(N), labels] / exp_logits.sum(axis=1))
    
    # Text-to-Image Loss
    exp_logits_t = np.exp(logits.T - np.max(logits.T, axis=1, keepdims=True))
    t2i_loss = -np.log(exp_logits_t[np.arange(N), labels] / exp_logits_t.sum(axis=1))
    
    # 평균
    loss = (i2t_loss.mean() + t2i_loss.mean()) / 2
    return loss


def demo_contrastive_learning():
    """Contrastive Learning 데모"""
    print("\n" + "="*60)
    print("📊 Contrastive Learning 데모")
    print("="*60)
    
    np.random.seed(42)
    
    # 가상의 매칭 쌍 (유사한 임베딩)
    N, D = 4, 64
    base_embeddings = np.random.randn(N, D)
    
    # 매칭 쌍: 약간의 노이즈만 추가
    image_embeddings = base_embeddings + np.random.randn(N, D) * 0.1
    text_embeddings = base_embeddings + np.random.randn(N, D) * 0.1
    
    # 유사도 행렬
    image_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    similarity = np.dot(image_norm, text_norm.T)
    
    print("\n유사도 행렬 (대각선이 매칭 쌍):")
    print("         Text0   Text1   Text2   Text3")
    for i in range(N):
        row = " ".join([f"{similarity[i,j]:7.3f}" for j in range(N)])
        print(f"  Image{i}: {row}")
    
    # Loss 계산
    loss = contrastive_loss(image_embeddings, text_embeddings)
    print(f"\nContrastive Loss: {loss:.4f}")
    
    # 비매칭 쌍으로 비교
    random_text = np.random.randn(N, D)
    loss_random = contrastive_loss(image_embeddings, random_text)
    print(f"랜덤 쌍 Loss: {loss_random:.4f}")
    print("→ 매칭 쌍의 Loss가 더 낮음 (학습 목표)")


# ============================================================
# Part 2: 가상 CLIP 모델
# ============================================================

class SimpleCLIP:
    """
    간단한 CLIP 시뮬레이터
    
    실제 CLIP은 ViT + Transformer를 사용하지만,
    여기서는 해시 기반 임베딩으로 개념을 설명합니다.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        np.random.seed(42)
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """텍스트를 임베딩으로 변환"""
        embeddings = []
        for text in texts:
            # 해시 기반 임베딩 (데모용)
            np.random.seed(hash(text.lower()) % 2**31)
            emb = np.random.randn(self.embedding_dim)
            embeddings.append(emb / np.linalg.norm(emb))
        return np.array(embeddings)
    
    def encode_image(self, image_descriptions: List[str]) -> np.ndarray:
        """
        이미지를 임베딩으로 변환
        
        실제로는 이미지 픽셀을 처리하지만,
        여기서는 이미지 설명을 사용해 시뮬레이션합니다.
        """
        # 이미지 설명과 유사한 임베딩 생성
        embeddings = []
        for desc in image_descriptions:
            np.random.seed(hash(desc.lower()) % 2**31)
            emb = np.random.randn(self.embedding_dim)
            # 약간의 노이즈 추가 (이미지-텍스트 간 차이 시뮬레이션)
            noise = np.random.randn(self.embedding_dim) * 0.2
            emb = emb + noise
            embeddings.append(emb / np.linalg.norm(emb))
        return np.array(embeddings)
    
    def compute_similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        """이미지-텍스트 유사도 계산"""
        return np.dot(image_emb, text_emb.T)


# ============================================================
# Part 3: Zero-shot Classification
# ============================================================

def zero_shot_classification(clip_model: SimpleCLIP,
                             image_description: str,
                             class_names: List[str],
                             template: str = "a photo of a {}") -> Dict[str, float]:
    """
    Zero-shot 이미지 분류
    
    Args:
        clip_model: CLIP 모델
        image_description: 이미지 설명 (시뮬레이션용)
        class_names: 클래스 이름 리스트
        template: 프롬프트 템플릿
    
    Returns:
        각 클래스의 확률
    """
    # 클래스 프롬프트 생성
    class_prompts = [template.format(c) for c in class_names]
    
    # 임베딩
    image_emb = clip_model.encode_image([image_description])
    text_emb = clip_model.encode_text(class_prompts)
    
    # 유사도 → 확률
    similarities = clip_model.compute_similarity(image_emb, text_emb)[0]
    
    # Softmax
    exp_sim = np.exp(similarities * 10)  # temperature=0.1
    probs = exp_sim / exp_sim.sum()
    
    return {c: p for c, p in zip(class_names, probs)}


def demo_zero_shot():
    """Zero-shot Classification 데모"""
    print("\n" + "="*60)
    print("🎯 Zero-shot Classification 데모")
    print("="*60)
    
    clip = SimpleCLIP()
    
    # 테스트 이미지들 (설명으로 시뮬레이션)
    test_images = [
        "a cute cat sitting on a couch",
        "a golden retriever playing in the park",
        "a colorful bird on a tree branch"
    ]
    
    class_names = ["cat", "dog", "bird", "fish", "rabbit"]
    
    for image_desc in test_images:
        print(f"\n🖼️ 이미지: '{image_desc}'")
        
        probs = zero_shot_classification(clip, image_desc, class_names)
        
        # 정렬하여 출력
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            print(f"  {class_name:10s}: {prob:5.1%} {bar}")
        
        predicted = sorted_probs[0][0]
        print(f"  → 예측: {predicted}")


# ============================================================
# Part 4: 이미지 검색
# ============================================================

def image_search(clip_model: SimpleCLIP,
                 query_text: str,
                 image_database: List[str],
                 top_k: int = 3) -> List[Tuple[str, float]]:
    """
    텍스트 쿼리로 이미지 검색
    
    Args:
        clip_model: CLIP 모델
        query_text: 검색 쿼리
        image_database: 이미지 설명 리스트 (시뮬레이션용)
        top_k: 반환할 결과 수
    
    Returns:
        (이미지 설명, 유사도) 리스트
    """
    # 임베딩
    query_emb = clip_model.encode_text([query_text])
    image_embs = clip_model.encode_image(image_database)
    
    # 유사도
    similarities = clip_model.compute_similarity(query_emb, image_embs)[0]
    
    # 상위 k개
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(image_database[i], similarities[i]) for i in top_indices]


def demo_image_search():
    """이미지 검색 데모"""
    print("\n" + "="*60)
    print("🔍 이미지 검색 데모")
    print("="*60)
    
    clip = SimpleCLIP()
    
    # 이미지 데이터베이스 (설명으로 시뮬레이션)
    image_database = [
        "a sunset over the ocean with orange sky",
        "a busy city street with tall buildings",
        "a peaceful mountain landscape with snow",
        "a cute puppy playing with a ball",
        "a delicious pizza with cheese and tomatoes",
        "a beautiful flower garden in spring",
        "a starry night sky with milky way"
    ]
    
    queries = [
        "nature scenery",
        "urban environment",
        "food photography"
    ]
    
    for query in queries:
        print(f"\n🔎 쿼리: '{query}'")
        results = image_search(clip, query, image_database, top_k=3)
        
        for rank, (image, sim) in enumerate(results, 1):
            print(f"  {rank}. [{sim:.3f}] {image}")


# ============================================================
# Part 5: Hugging Face CLIP 사용 (선택적)
# ============================================================

def demo_huggingface_clip():
    """Hugging Face CLIP 사용 데모"""
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import requests
        import torch
        
        print("\n" + "="*60)
        print("🚀 Hugging Face CLIP 데모")
        print("="*60)
        
        # 모델 로드
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 테스트 이미지 로드
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        try:
            image = Image.open(requests.get(url, stream=True, timeout=5).raw)
            
            # 클래스 후보
            class_names = ["cat", "dog", "bird", "car", "person"]
            texts = [f"a photo of a {c}" for c in class_names]
            
            # 추론
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            
            print("\n예측 결과:")
            for name, prob in zip(class_names, probs):
                bar = "█" * int(prob.item() * 20)
                print(f"  {name:10s}: {prob.item():5.1%} {bar}")
        
        except requests.exceptions.RequestException:
            print("⚠️ 이미지를 다운로드할 수 없습니다 (네트워크 오류)")
            
    except ImportError:
        print("\n⚠️ transformers 또는 torch가 설치되지 않았습니다.")
        print("설치: pip install transformers torch pillow requests")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 14: 멀티모달 LLM 실습")
    print("="*60)
    
    demo_contrastive_learning()
    demo_zero_shot()
    demo_image_search()
    demo_huggingface_clip()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
