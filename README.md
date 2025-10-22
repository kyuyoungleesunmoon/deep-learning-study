# Deep Learning Study - PyTorch 가이드

PyTorch를 사용한 딥러닝 학습 자료입니다.

## 📚 학습 자료

### 🎓 [신경망 학습 필수 수학 공식 교육 시리즈](./neural_network_math_series/) (NEW! 🔥)

**초심자도 이해할 수 있는 8단계 완전 가이드!**

신경망 학습의 수학적 기초를 처음부터 끝까지 체계적으로 학습할 수 있는 완전한 교육 시리즈입니다.

#### 📖 [시리즈 바로가기](./neural_network_math_series/README.md)

**포함된 내용 (총 8 Stages):**
1. ✅ **Stage 1**: 기본 수학 구조 (스칼라, 벡터, 행렬, 선형 변환)
2. ✅ **Stage 2**: 퍼셉트론과 가중합 (z = w·x + b)
3. ✅ **Stage 3**: 활성화 함수 (Sigmoid, ReLU, Tanh) + 미분
4. ✅ **Stage 4**: 손실 함수 (MSE, Cross-Entropy)
5. ✅ **Stage 5**: 미분과 편미분, 그래디언트
6. ✅ **Stage 6**: 경사하강법 (Gradient Descent)
7. ✅ **Stage 7**: 신경망 학습 (1층 → 다층)
8. ✅ **Stage 8**: 최종 정리 (수식만으로 전 과정 설명)

**특징:**
- 📐 LaTeX 수식으로 정확한 표현
- 🔍 모든 수식 기호 상세 설명
- 🌍 실생활 비유로 직관적 이해
- 📊 25+ Python 시각화 (Matplotlib/NumPy)
- 📝 50+ 수치 예제
- 💻 실행 가능한 코드

**난이도**: ⭐⭐~⭐⭐⭐⭐⭐ (초급 → 고급)  
**총 학습 시간**: 약 10-15시간  
**페이지 수**: 약 100페이지

---

### 🧮 신경망 학습의 수학적 기초

신경망(딥러닝) 학습을 수학적으로 완전히 이해하기 위한 교육자료입니다. 이론, 수치 예제, 실행 가능한 코드를 포함합니다.

#### 이론 문서 (Markdown)

**[1. 기본 단계 - 신경망 학습의 수학적 기초](./01_basic_neural_network_math.md)**
- 퍼셉트론 (Perceptron) - 수식과 기호 설명
- 활성화 함수 (Sigmoid, ReLU, Tanh) - 미분 포함
- 순방향 전파 (Forward Propagation) - 수치 예제
- 손실 함수 (MSE, Cross-Entropy) - 계산 과정

**[2. 중급 단계 - 역전파와 경사하강법](./02_intermediate_backpropagation.md)**
- 역전파 알고리즘 - 완전한 수학적 유도
- 경사하강법 변형 (Batch, SGD, Mini-batch)
- 연쇄 법칙 (Chain Rule) - 상세한 예제
- 가중치 업데이트 - 단계별 계산

**[3. 고급 단계 - 최적화 및 정규화](./03_advanced_optimization.md)**
- 고급 최적화 (Adam, RMSprop, AdaGrad) - 수식과 예제
- 정규화 기법 (L1, L2, Dropout) - 수학적 기초
- 배치 정규화 (Batch Normalization) - 상세한 계산
- 고급 아키텍처 (CNN, RNN) - 수학적 정의

#### 실습 노트북 (Jupyter)

**[📔 Jupyter Notebooks](./notebooks/)** - 실행 가능한 코드로 검증

1. **기본**: `01_basic_neural_network.ipynb` - 퍼셉트론, 활성화 함수, 순방향 전파
2. **중급**: `02_backpropagation_from_scratch.ipynb` - 역전파 구현, 그래디언트 검증
3. **고급**: `03_optimization_techniques.ipynb` - Adam, 정규화, 배치 정규화

**학습 수준:** 대학(학부 상위/대학원) ~ 산업 실무자  
**특징:** 모든 수식에 기호 설명(Legend) 포함, 수치 예제로 검증

---

### 🐍 [Python 기초 - 변수와 자료형](./python_basics/)
Python 프로그래밍의 기초인 변수 선언 규칙과 기본 자료형을 학습합니다.

**포함된 내용:**
1. 변수 선언 및 네이밍 규칙
2. 숫자형 자료형 (int, float, complex)
3. 문자열(str) 자료형
4. 자료형 변환
5. 내장 함수 활용
6. 주석 작성법

**예상 학습 시간:** 3시간 30분 (이론 2시간 + 실습 1시간 30분)

### 📖 [PyTorch 문법 가이드](./torch_tutorial.md)
PyTorch의 기본 문법과 개념을 단계별로 설명한 완전한 가이드입니다.

**포함된 내용:**
1. PyTorch 기초
2. 텐서(Tensor) 기본 연산
3. 자동 미분(Autograd)
4. 신경망 구축
5. 데이터 로딩
6. 학습 루프
7. 모델 저장 및 로드

### 📊 [모델 평가와 하이퍼파라미터 튜닝](./model_evaluation_theory.md)
머신러닝 모델의 성능을 정확하게 평가하고 최적화하는 방법을 배웁니다.

**포함된 내용:**
1. 파이프라인을 사용한 효율적인 워크플로
2. k-겹 교차 검증
3. 학습 곡선과 검증 곡선
4. 그리드 서치와 랜덤 서치
5. 다양한 성능 평가 지표
6. 불균형한 클래스 처리

### 🎯 [앙상블 학습 (Ensemble Learning)](./ensemble_learning.md)
다양한 모델을 결합하여 더 강력한 예측 모델을 만드는 앙상블 학습을 배웁니다.

**포함된 내용:**
1. 앙상블 학습 개요 (배깅, 부스팅, 스태킹)
2. 다수결 투표 앙상블 (Voting)
3. 배깅 (Bagging)과 랜덤 포레스트
4. 에이다부스트 (AdaBoost)
5. 그레이디언트 부스팅 & XGBoost
6. 모델 성능 평가 및 비교

**예상 학습 시간:** 4시간 30분 (이론 2시간 + 실습 2시간 30분)

### 💻 [실행 가능한 예제](./examples/)
실제로 실행하고 실험해볼 수 있는 예제 코드들입니다.

**예제 목록:**
- `01_tensor_basics.py` - 텐서 기초 연산
- `02_autograd.py` - 자동 미분
- `03_neural_network.py` - 신경망 구축
- `04_training_loop.py` - 학습 루프
- `05_complete_example.py` - 완전한 이미지 분류 예제
- `06_model_evaluation.py` - 모델 평가와 하이퍼파라미터 튜닝
- `07_ensemble_learning.py` - 앙상블 학습 (Voting, Bagging, Boosting, XGBoost)

## 🚀 시작하기

### 빠른 시작
처음 시작하시나요? **[빠른 시작 가이드](./QUICKSTART.md)**를 읽어보세요! (5분이면 충분합니다)

### 요구사항
```bash
pip install -r requirements.txt
```

또는

```bash
pip install torch torchvision numpy
```

### 학습 방법
1. `torch_tutorial.md`를 읽으며 개념을 이해합니다.
2. `examples/` 디렉토리의 예제를 순서대로 실행해봅니다.
3. 코드를 수정하고 실험하며 이해를 깊게 합니다.

## 📝 학습 순서

1. **이론 학습**: `torch_tutorial.md`에서 각 섹션을 읽습니다.
2. **실습**: 해당 섹션의 예제 코드를 실행합니다.
3. **실험**: 파라미터를 변경하며 결과를 관찰합니다.
4. **응용**: 배운 내용을 바탕으로 자신만의 코드를 작성합니다.

## 📖 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch Examples](https://github.com/pytorch/examples)

## 🤝 기여

이 저장소는 딥러닝 학습을 위한 자료입니다. 개선 사항이나 추가 예제가 있다면 언제든 제안해주세요!