# 신경망 학습 노트북 (Jupyter Notebooks)

이 디렉토리는 신경망 학습의 수학적 기초를 실행 가능한 코드로 학습할 수 있는 Jupyter 노트북을 포함합니다.

## 📚 노트북 목록

### 1. 기본 단계 (Basic Level)
**파일:** `01_basic_neural_network.ipynb`

**내용:**
- 퍼셉트론 구현 및 수치 검증
- 활성화 함수 (Sigmoid, ReLU, Tanh) 시각화
- 순방향 전파 구현
- 손실 함수 (MSE, Cross-Entropy) 계산 및 시각화

**학습 시간:** 30-45분

**선수 지식:** 
- Python 기초
- NumPy 기본 사용법
- 기본 수학 (미적분 개념)

---

### 2. 중급 단계 (Intermediate Level)
**파일:** `02_backpropagation_from_scratch.ipynb`

**내용:**
- 역전파 알고리즘 처음부터 구현
- 경사하강법 구현
- 완전한 학습 파이프라인
- 그래디언트 검증 (Gradient Checking)
- 결정 경계 시각화

**학습 시간:** 60-90분

**선수 지식:**
- 기본 단계 노트북 완료
- 연쇄 법칙 (Chain Rule) 이해
- 행렬 연산

---

### 3. 고급 단계 (Advanced Level)
**파일:** `03_optimization_techniques.ipynb`

**내용:**
- 고급 최적화 알고리즘 (Adam, RMSprop, AdaGrad)
- 정규화 기법 (L1, L2, Dropout)
- 배치 정규화 (Batch Normalization)
- 학습률 스케줄링
- 조기 종료 (Early Stopping)
- 다양한 기법 비교 및 분석

**학습 시간:** 90-120분

**선수 지식:**
- 중급 단계 노트북 완료
- 최적화 이론 기초
- 과적합/과소적합 개념

---

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install jupyter numpy matplotlib

# 선택적: PyTorch 설치 (고급 노트북에서 사용)
pip install torch torchvision
```

### 2. Jupyter Notebook 실행

```bash
# notebooks 디렉토리로 이동
cd notebooks

# Jupyter Notebook 실행
jupyter notebook
```

### 3. 노트북 순서대로 실행

1. `01_basic_neural_network.ipynb` 부터 시작
2. 각 셀을 순서대로 실행 (Shift + Enter)
3. 코드를 수정하며 실험
4. 다음 노트북으로 진행

---

## 📖 학습 방법

### 권장 학습 전략

1. **이론 먼저 읽기**
   - 각 노트북의 수식과 설명을 먼저 읽습니다
   - 기호 설명(Legend)을 주의 깊게 확인합니다

2. **코드 실행**
   - 셀을 하나씩 실행하며 결과를 확인합니다
   - 출력 값이 수식의 계산 결과와 일치하는지 검증합니다

3. **실험 및 수정**
   - 파라미터 값을 변경해봅니다
   - 학습률, 반복 횟수 등을 조절하며 관찰합니다
   - 다른 활성화 함수를 시도해봅니다

4. **시각화 분석**
   - 그래프를 주의 깊게 관찰합니다
   - 학습 곡선의 변화를 이해합니다
   - 결정 경계의 모양을 분석합니다

---

## 🎯 학습 목표

### 기본 단계 완료 후
- ✓ 퍼셉트론의 동작 원리 이해
- ✓ 활성화 함수의 역할 이해
- ✓ 순방향 전파 과정 구현 가능
- ✓ 손실 함수의 의미 이해

### 중급 단계 완료 후
- ✓ 역전파 알고리즘 완전 이해
- ✓ 그래디언트 계산 가능
- ✓ 신경망 학습 파이프라인 구현 가능
- ✓ 그래디언트 검증 방법 이해

### 고급 단계 완료 후
- ✓ 다양한 최적화 알고리즘 사용 가능
- ✓ 정규화 기법 적용 가능
- ✓ 과적합 방지 방법 이해
- ✓ 실전 문제에 적용 가능

---

## 💡 팁

### 디버깅
- 그래디언트가 폭발하거나 소실되면 학습률을 조정하세요
- 손실이 감소하지 않으면 초기화 방법을 확인하세요
- NaN이 발생하면 수치 안정성을 위한 클리핑을 추가하세요

### 실험
- 다양한 네트워크 구조를 시도해보세요
- 하이퍼파라미터를 체계적으로 변경하며 관찰하세요
- 결과를 노트에 기록하여 비교하세요

### 심화 학습
- 각 노트북의 코드를 확장하여 새로운 기능을 추가해보세요
- 실제 데이터셋 (MNIST, CIFAR-10 등)으로 실험해보세요
- PyTorch나 TensorFlow와 비교해보세요

---

## 🔗 관련 자료

### 이론 문서
- `../01_basic_neural_network_math.md` - 기본 수학 이론
- `../02_intermediate_backpropagation.md` - 역전파 이론
- `../03_advanced_optimization.md` - 고급 최적화 이론

### 예제 코드
- `../examples/` - 추가 PyTorch 예제

### 외부 자료
- [Deep Learning Book](https://www.deeplearningbook.org/) - 이론적 배경
- [CS231n](http://cs231n.stanford.edu/) - Stanford 강의
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - 시각적 설명

---

## ⚠️ 주의사항

1. **실행 순서**: 노트북의 셀을 순서대로 실행하세요. 순서를 건너뛰면 오류가 발생할 수 있습니다.

2. **메모리 관리**: 큰 데이터나 복잡한 모델을 사용할 때는 메모리 사용량을 확인하세요.

3. **랜덤 시드**: 재현성을 위해 `np.random.seed()`를 설정했지만, 다른 시드로도 실험해보세요.

4. **수치 안정성**: 매우 작거나 큰 값을 다룰 때 오버플로우/언더플로우에 주의하세요.

---

## 📝 피드백

노트북에 대한 피드백이나 개선 제안이 있다면 Issues를 통해 공유해주세요!
