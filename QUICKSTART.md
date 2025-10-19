# 빠른 시작 가이드

이 가이드는 PyTorch 학습 자료를 빠르게 시작하는 방법을 안내합니다.

## 🎯 5분 안에 시작하기

### 1단계: 환경 설정 (1분)

```bash
# PyTorch 및 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 2단계: 첫 번째 예제 실행 (1분)

```bash
# 텐서 기초 예제 실행
python examples/01_tensor_basics.py
```

**예상 출력:**
```
==================================================
1. 텐서 생성
==================================================

빈 텐서 (초기화되지 않음):
tensor([[...]])

랜덤 텐서 (0~1 균등 분포):
tensor([[...]])
...
```

### 3단계: 이론 학습 (3분)

[`torch_tutorial.md`](./torch_tutorial.md)를 열어 다음 섹션을 빠르게 읽어보세요:
1. PyTorch 기초
2. 텐서(Tensor) 기본 연산

## 📚 단계별 학습 경로

### 초급 (1-2시간)

1. **이론**: `torch_tutorial.md`의 섹션 1-2 읽기
2. **실습**: 
   - `examples/01_tensor_basics.py` 실행 및 코드 읽기
   - `examples/02_autograd.py` 실행 및 코드 읽기

**학습 목표:**
- ✓ 텐서 생성 및 조작
- ✓ 자동 미분의 개념 이해

### 중급 (2-3시간)

3. **이론**: `torch_tutorial.md`의 섹션 3-4 읽기
4. **실습**:
   - `examples/03_neural_network.py` 실행 및 코드 읽기
   - `examples/04_training_loop.py` 실행 및 코드 읽기

**학습 목표:**
- ✓ 신경망 구조 이해
- ✓ 학습 파이프라인 구현

### 고급 (3-4시간)

5. **이론**: `torch_tutorial.md`의 섹션 5-7 읽기
6. **실습**:
   - `examples/05_complete_example.py` 실행 및 코드 읽기
   - 파라미터 변경하며 실험

**학습 목표:**
- ✓ 완전한 프로젝트 구현
- ✓ 모델 저장 및 배포

## 🎓 학습 팁

### 효과적인 학습 방법

1. **코드를 직접 실행하세요**
   ```bash
   # 예제 실행
   python examples/01_tensor_basics.py
   ```

2. **코드를 수정해보세요**
   - 텐서 크기 변경
   - 레이어 개수 조정
   - 하이퍼파라미터 튜닝

3. **출력을 관찰하세요**
   - 각 단계에서 텐서의 크기와 값 확인
   - 학습 중 손실과 정확도 변화 관찰

4. **작은 프로젝트를 만들어보세요**
   - 간단한 분류 문제 해결
   - 자신만의 신경망 구조 설계

## 🔧 문제 해결

### PyTorch 설치 오류

**문제**: `ModuleNotFoundError: No module named 'torch'`

**해결책**:
```bash
pip install torch torchvision numpy
```

### CUDA 오류

**문제**: GPU를 사용할 수 없다는 메시지

**해결책**: 
- CPU로 학습하려면 그대로 진행 (속도가 느릴 수 있음)
- GPU를 사용하려면 CUDA가 설치된 PyTorch 버전 필요
  ```bash
  # CUDA 11.8 예시
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### 메모리 부족 오류

**문제**: `RuntimeError: CUDA out of memory`

**해결책**:
- 배치 크기 줄이기
- 모델 크기 줄이기
- CPU 사용으로 전환

## 📖 다음 단계

완전한 학습 자료:
- [`torch_tutorial.md`](./torch_tutorial.md) - 전체 튜토리얼 문서
- [`examples/README.md`](./examples/README.md) - 예제 상세 설명

실전 프로젝트:
1. MNIST 숫자 인식
2. 이미지 분류 (CIFAR-10)
3. 자연어 처리 기초
4. 커스텀 데이터셋으로 학습

## 🤝 도움이 필요하신가요?

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch 포럼](https://discuss.pytorch.org/)

## ✅ 체크리스트

학습 진행 상황을 체크해보세요:

- [ ] PyTorch 설치 완료
- [ ] 첫 번째 예제 실행 완료
- [ ] 텐서 생성 및 연산 이해
- [ ] 자동 미분 개념 이해
- [ ] 간단한 신경망 구현
- [ ] CNN 모델 이해
- [ ] 학습 루프 구현
- [ ] 완전한 예제 실행 완료
- [ ] 자신만의 프로젝트 시작

행운을 빕니다! 🚀
