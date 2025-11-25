# 📖 Chapter 14: 멀티모달 LLM (Multimodal LLM)

## 📋 개요

이 챕터에서는 텍스트와 이미지를 함께 처리하는 멀티모달 모델을 학습합니다.
- CLIP (Contrastive Language-Image Pre-training)
- 이미지-텍스트 임베딩
- Zero-shot 이미지 분류

## 🔬 핵심 알고리즘

### 1. CLIP (Contrastive Language-Image Pre-training)

**원리**: 이미지와 텍스트를 같은 임베딩 공간에 매핑

```
이미지 → Image Encoder → 이미지 임베딩 (512차원)
텍스트 → Text Encoder → 텍스트 임베딩 (512차원)

유사도 = cosine(이미지 임베딩, 텍스트 임베딩)
```

**학습 방법 (Contrastive Learning)**:
```
배치 내 N개의 (이미지, 텍스트) 쌍:
- 대각선 (매칭 쌍): 유사도 최대화
- 비대각선 (비매칭 쌍): 유사도 최소화

Loss = -log(exp(sim(I_i, T_i)/τ) / Σ exp(sim(I_i, T_j)/τ))
```

**특징**:
- 4억 개의 이미지-텍스트 쌍으로 학습
- Zero-shot 분류 가능 (학습 없이 새로운 클래스 분류)
- 다양한 다운스트림 태스크에 활용

### 2. Image Encoder

**Vision Transformer (ViT) 방식**:
```
이미지 (224×224) → 패치 분할 (16×16 패치 = 196개)
→ Linear Projection → Patch Embeddings
→ Position Embeddings 추가
→ Transformer Encoder → [CLS] 토큰 → 이미지 임베딩
```

**ResNet 방식**:
```
이미지 → CNN 레이어들 → Global Average Pooling → 이미지 임베딩
```

### 3. Zero-shot Classification

**원리**: 클래스 이름을 텍스트로 인코딩하여 이미지와 비교

```python
# 클래스 프롬프트
prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# 텍스트 임베딩
text_embeddings = text_encoder(prompts)

# 이미지 임베딩
image_embedding = image_encoder(image)

# 유사도 계산
similarities = cosine_similarity(image_embedding, text_embeddings)

# 가장 유사한 클래스 선택
predicted_class = argmax(similarities)
```

## 📊 실습 예제

### 예제 1: Hugging Face CLIP 사용

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# 모델 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 이미지 로드
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 텍스트 후보
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# 전처리
inputs = processor(
    text=texts, 
    images=image, 
    return_tensors="pt", 
    padding=True
)

# 추론
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # (1, 3)
probs = logits_per_image.softmax(dim=1)

print("예측 확률:")
for text, prob in zip(texts, probs[0]):
    print(f"  {text}: {prob.item():.2%}")
```

### 예제 2: 이미지-텍스트 유사도 계산

```python
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(text):
    """텍스트 임베딩 추출"""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def get_image_embedding(image):
    """이미지 임베딩 추출"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features / image_features.norm(dim=-1, keepdim=True)

# 유사도 계산
text_emb = get_text_embedding("a happy golden retriever")
image_emb = get_image_embedding(image)

similarity = torch.matmul(text_emb, image_emb.T)
print(f"유사도: {similarity.item():.4f}")
```

### 예제 3: 이미지 검색

```python
import numpy as np
from PIL import Image

def image_search(query_text, image_embeddings, images, top_k=3):
    """텍스트 쿼리로 이미지 검색"""
    query_emb = get_text_embedding(query_text).numpy()
    
    # 코사인 유사도
    similarities = np.dot(image_embeddings, query_emb.T).flatten()
    
    # 상위 k개
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'image': images[idx],
            'similarity': similarities[idx]
        })
    
    return results

# 사용 예시
query = "a sunset over the ocean"
results = image_search(query, all_image_embeddings, all_images)
```

### 예제 4: 프롬프트 엔지니어링

```python
# 다양한 프롬프트 템플릿
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "a {} in the wild",
    "a {} in nature",
    "an image of a {}"
]

def ensemble_classification(image, class_names):
    """여러 프롬프트의 앙상블로 분류"""
    all_scores = []
    
    for template in templates:
        texts = [template.format(c) for c in class_names]
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)
        all_scores.append(scores)
    
    # 앙상블 (평균)
    ensemble_scores = torch.stack(all_scores).mean(dim=0)
    return ensemble_scores

# 사용 예시
class_names = ["cat", "dog", "bird", "fish"]
scores = ensemble_classification(image, class_names)
predicted = class_names[scores.argmax()]
```

## 🎯 핵심 포인트

1. **동일 임베딩 공간**: 이미지와 텍스트가 같은 차원의 벡터로 표현됨
2. **Zero-shot 능력**: 학습하지 않은 새로운 클래스도 분류 가능
3. **프롬프트 중요**: "a photo of a {class}" 형식이 효과적
4. **앙상블 효과**: 여러 프롬프트 템플릿 사용 시 정확도 향상

## 📚 참고 자료

- 원본 코드: https://github.com/onlybooks/llm/tree/main/14장
- CLIP 논문: https://arxiv.org/abs/2103.00020
- Hugging Face CLIP: https://huggingface.co/openai/clip-vit-base-patch32
