# PyTorch 예제 코드

이 디렉토리는 PyTorch의 핵심 개념을 단계별로 학습할 수 있는 실행 가능한 예제들을 포함합니다.

## 예제 목록

### 01_tensor_basics.py
**텐서 기초 연산**
- 텐서 생성 (empty, rand, zeros, ones, tensor, full, arange)
- 텐서 연산 (덧셈, 곱셈, 행렬 곱셈)
- 인덱싱과 슬라이싱
- 텐서 크기 변경 (view, reshape, flatten)
- NumPy 변환
- GPU 사용
- 집계 함수 (sum, mean, max, min, std)

**실행 방법:**
```bash
python 01_tensor_basics.py
```

### 02_autograd.py
**자동 미분 (Autograd)**
- 기본 자동 미분 (requires_grad, backward)
- 그래디언트 누적
- 그래디언트 흐름 제어 (no_grad, detach)
- 벡터-Jacobian 곱
- 실제 예제: 선형 회귀
- 그래디언트 체크포인팅
- requires_grad 동적 변경

**실행 방법:**
```bash
python 02_autograd.py
```

### 03_neural_network.py
**신경망 구축**
- nn.Module 기본 사용법
- Sequential 모델
- 컨볼루션 신경망 (CNN)
- 다양한 활성화 함수 (ReLU, Sigmoid, Tanh, LeakyReLU)
- 배치 정규화 (Batch Normalization)
- 잔차 연결 (Residual Connection)
- 가중치 초기화
- 모델 요약
- 학습/평가 모드 (train vs eval)

**실행 방법:**
```bash
python 03_neural_network.py
```

### 04_training_loop.py
**학습 루프**
- 데이터셋 준비
- 학습 에폭 구현
- 모델 검증
- 다양한 옵티마이저 (SGD, Adam, RMSprop, AdamW)
- 학습률 스케줄러 (StepLR, ExponentialLR, ReduceLROnPlateau)
- 그래디언트 클리핑
- Early Stopping
- 모델 저장 및 로드 (체크포인트)

**실행 방법:**
```bash
python 04_training_loop.py
```

### 05_complete_example.py
**완전한 예제: 이미지 분류**
- 커스텀 Dataset 구현
- CNN 모델 정의
- 완전한 학습 파이프라인
- 모델 평가
- 추론 예제
- 확률 출력

**실행 방법:**
```bash
python 05_complete_example.py
```

## 요구사항

```bash
pip install torch torchvision numpy
```

## 학습 순서

1. **01_tensor_basics.py** - PyTorch의 기본 데이터 구조인 텐서를 이해합니다.
2. **02_autograd.py** - 자동 미분의 원리와 사용법을 배웁니다.
3. **03_neural_network.py** - 신경망을 구축하는 방법을 학습합니다.
4. **04_training_loop.py** - 모델을 학습시키는 전체 과정을 이해합니다.
5. **05_complete_example.py** - 실제 프로젝트처럼 모든 것을 통합합니다.

## 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch Examples](https://github.com/pytorch/examples)

## 팁

- 각 예제는 독립적으로 실행 가능합니다.
- 코드를 실행하면서 출력을 관찰하고 이해하세요.
- 예제 코드를 수정해보면서 실험해보세요.
- GPU가 있다면 CUDA를 활성화하여 성능을 비교해보세요.
