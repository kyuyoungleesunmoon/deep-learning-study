# Deep Learning Study - PyTorch 가이드

PyTorch를 사용한 딥러닝 학습 자료입니다.

## 📚 학습 자료

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

### 💻 [실행 가능한 예제](./examples/)
실제로 실행하고 실험해볼 수 있는 예제 코드들입니다.

**예제 목록:**
- `01_tensor_basics.py` - 텐서 기초 연산
- `02_autograd.py` - 자동 미분
- `03_neural_network.py` - 신경망 구축
- `04_training_loop.py` - 학습 루프
- `05_complete_example.py` - 완전한 이미지 분류 예제
- `06_model_evaluation.py` - 모델 평가와 하이퍼파라미터 튜닝

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