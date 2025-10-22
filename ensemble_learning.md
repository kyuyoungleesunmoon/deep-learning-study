# 다양한 모델을 결합한 앙상블 학습 (Ensemble Learning)

> 🎓 **강사**: 데이터사이언스 및 인공지능 전문가, 퍼실리테이션 기반 교육 전문가  
> 🎯 **목표**: 앙상블 학습의 핵심 개념과 알고리즘을 이론 + 시각화 + 실습으로 완벽 이해  
> 🕒 **예상 학습 시간**: 4시간 30분 (이론 2시간 + 실습 2시간 30분)

---

## 목차
1. [앙상블 학습 개요](#1️⃣-앙상블-학습-개요)
2. [다수결 투표 앙상블 (Voting)](#2️⃣-다수결-투표-앙상블-voting)
3. [배깅 (Bagging)](#3️⃣-배깅-bagging)
4. [에이다부스트 (AdaBoost)](#4️⃣-에이다부스트-adaboost)
5. [그레이디언트 부스팅 & XGBoost](#5️⃣-그레이디언트-부스팅--xgboost)
6. [모델 성능 평가 및 비교](#6️⃣-모델-성능-평가-및-비교)

---

## 1️⃣ 앙상블 학습 개요

### 📖 이론 설명

#### 1.1 단일 모델의 한계

단일 모델은 다음과 같은 문제를 겪을 수 있습니다:

- **과적합(Overfitting)**: 학습 데이터에 너무 맞춰져 새로운 데이터에 대한 성능이 떨어짐
- **과소적합(Underfitting)**: 모델이 너무 단순하여 데이터의 패턴을 제대로 학습하지 못함
- **높은 분산(High Variance)**: 학습 데이터가 조금만 바뀌어도 모델이 크게 달라짐
- **높은 편향(High Bias)**: 모델이 데이터의 진짜 패턴을 포착하지 못함

#### 1.2 집단지성(Ensemble)의 개념

**앙상블 학습**은 여러 개의 모델을 결합하여 단일 모델보다 더 나은 성능을 얻는 방법입니다.

**핵심 아이디어**: "세 사람이 모이면 문수의 지혜보다 낫다"
- 여러 전문가의 의견을 모으면 한 사람의 의견보다 정확할 가능성이 높음
- 각 모델의 실수가 서로 다른 방향이라면, 평균을 내면 오류가 줄어듦

#### 1.3 앙상블 학습의 세 가지 주요 방식

| 방식 | 설명 | 대표 알고리즘 | 주요 목적 |
|------|------|--------------|----------|
| **배깅(Bagging)** | 같은 알고리즘, 다른 데이터셋(부트스트랩) | Random Forest | **분산 감소** |
| **부스팅(Boosting)** | 순차적으로 약한 학습기를 강화 | AdaBoost, XGBoost | **편향 감소** |
| **스태킹(Stacking)** | 다른 알고리즘의 예측을 메타 모델로 학습 | Stacked Generalization | **일반화 성능** |

#### 1.4 편향-분산 트레이드오프

머신러닝 모델의 오차는 다음과 같이 분해됩니다:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**편향(Bias)**:
- 모델이 실제 관계를 얼마나 잘 근사하는가
- 높은 편향 = 과소적합
- 예: 선형 모델로 비선형 데이터 학습

**분산(Variance)**:
- 학습 데이터가 바뀔 때 모델이 얼마나 변하는가
- 높은 분산 = 과적합
- 예: 깊은 결정 트리

**앙상블의 역할**:
- **배깅**: 여러 모델의 평균으로 **분산 감소**
- **부스팅**: 잘못 분류된 샘플에 집중하여 **편향 감소**

### 🔢 수식

#### 평균 앙상블의 분산 감소

N개의 독립적인 모델이 있고, 각 모델의 분산이 $\sigma^2$일 때:

$$\text{Var}(\text{Ensemble}) = \frac{\sigma^2}{N}$$

**해석**: 모델 수가 증가하면 앙상블의 분산이 감소합니다!

#### 실제 상황 (모델이 완전히 독립적이지 않을 때)

상관계수가 $\rho$일 때:

$$\text{Var}(\text{Ensemble}) = \rho\sigma^2 + \frac{1-\rho}{N}\sigma^2$$

**통찰**:
- $\rho = 0$ (완전 독립): 분산이 $1/N$로 감소
- $\rho = 1$ (완전 상관): 분산이 전혀 감소하지 않음
- **따라서 다양성(diversity)이 중요!**

### 💻 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 한글 폰트 설정
rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 생성
np.random.seed(42)
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 단일 모델
single_model = DecisionTreeClassifier(max_depth=3, random_state=42)
single_model.fit(X_train, y_train)

# 앙상블 모델
lr = LogisticRegression(random_state=42)
svm = SVC(kernel='rbf', random_state=42, probability=True)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
ensemble = VotingClassifier(estimators=[('lr', lr), ('svm', svm), ('dt', dt)], voting='soft')
ensemble.fit(X_train, y_train)

# 결정 경계 시각화 함수
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=50)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# 비교 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
plot_decision_boundary(single_model, X_train, y_train, 'Single Model (Decision Tree)')
plt.text(0.5, -1.3, f'Test Accuracy: {single_model.score(X_test, y_test):.3f}', 
         ha='center', fontsize=12, weight='bold')

plt.sca(axes[1])
plot_decision_boundary(ensemble, X_train, y_train, 'Ensemble Model (Voting)')
plt.text(0.5, -1.3, f'Test Accuracy: {ensemble.score(X_test, y_test):.3f}', 
         ha='center', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('ensemble_vs_single.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("Single Model vs Ensemble Comparison")
print("=" * 60)
print(f"Single Model (Decision Tree) - Test Accuracy: {single_model.score(X_test, y_test):.4f}")
print(f"Ensemble Model (Voting)      - Test Accuracy: {ensemble.score(X_test, y_test):.4f}")
print(f"Improvement: {(ensemble.score(X_test, y_test) - single_model.score(X_test, y_test)) * 100:.2f}%")
```

**기대 출력**:
- 단일 모델: 과적합 또는 과소적합으로 결정 경계가 불안정
- 앙상블 모델: 더 부드럽고 안정적인 결정 경계
- 정확도 향상: 일반적으로 3-10% 개선

### 💬 퍼실리테이션 질문

**질문 1**: "여러 모델을 합치면 왜 더 나은 결과가 나올까요?"

**답변 가이드**:
- 각 모델이 다른 실수를 하기 때문
- 평균을 내면 극단적인 예측이 완화됨
- 집단지성: 여러 전문가의 의견이 한 사람보다 정확

**질문 2**: "모든 경우에 앙상블이 항상 더 좋을까요?"

**답변 가이드**:
- 모델들이 너무 비슷하면 효과 감소
- 계산 비용 증가
- 해석력 감소
- 적절한 균형이 필요

### 🧮 루브릭 평가표

| 평가항목 | 우수 (3점) | 보통 (2점) | 미흡 (1점) |
|----------|-----------|-----------|-----------|
| **개념 이해** | 배깅, 부스팅, 스태킹의 차이를 명확히 설명하고 예시 제시 | 3가지 방식의 정의는 알지만 차이점 설명이 불명확 | 개념 정의가 혼란스럽거나 불완전 |
| **편향-분산 이해** | 편향과 분산의 의미를 이해하고 앙상블과의 관계 설명 | 편향-분산 정의는 알지만 앙상블과의 연결이 약함 | 편향-분산 개념 이해 부족 |
| **시각화 해석** | 결정 경계의 차이를 정확히 해석하고 성능 향상 이유 설명 | 시각화 결과는 보지만 깊은 해석 부족 | 시각화를 제대로 이해하지 못함 |

---

## 2️⃣ 다수결 투표 앙상블 (Voting)

### 📖 이론 설명

#### 2.1 투표 앙상블의 개념

투표(Voting) 앙상블은 여러 개의 **서로 다른 알고리즘**을 학습시키고, 그 예측 결과를 투표로 결합합니다.

**핵심 아이디어**: 민주주의 투표처럼 다수결로 최종 결정

#### 2.2 하드 보팅 (Hard Voting)

각 모델의 예측 클래스를 집계하여 **가장 많이 예측된 클래스**를 선택합니다.

**예시**:
- 모델 1: Class A
- 모델 2: Class B
- 모델 3: Class A
- 모델 4: Class A
- **최종 예측: Class A (3표)**

#### 2.3 소프트 보팅 (Soft Voting)

각 모델의 **예측 확률을 평균**하여 가장 높은 확률의 클래스를 선택합니다.

**예시**:
- 모델 1: [0.7, 0.3] → Class 0
- 모델 2: [0.4, 0.6] → Class 1
- 모델 3: [0.6, 0.4] → Class 0
- **평균 확률: [0.567, 0.433]**
- **최종 예측: Class 0**

**장점**: 확률 정보를 활용하여 더 정교한 예측 가능

### 🔢 수식

#### 하드 보팅

$$\hat{y} = \text{mode}\{h_1(x), h_2(x), \ldots, h_M(x)\}$$

여기서:
- $h_i(x)$: i번째 모델의 예측
- $M$: 모델 개수
- $\text{mode}$: 최빈값 (가장 많이 나온 값)

#### 소프트 보팅

$$\hat{y} = \arg\max_c \frac{1}{M} \sum_{i=1}^{M} P_{h_i}(c|x)$$

여기서:
- $P_{h_i}(c|x)$: i번째 모델이 예측한 클래스 c의 확률
- $\arg\max_c$: 확률이 가장 높은 클래스

#### 단계별 계산 예제

3개의 모델이 있고, 2개의 클래스(0, 1)를 분류한다고 가정:

**각 모델의 확률 예측**:
- 모델 1: $P(y=0|x) = 0.8, P(y=1|x) = 0.2$
- 모델 2: $P(y=0|x) = 0.5, P(y=1|x) = 0.5$
- 모델 3: $P(y=0|x) = 0.6, P(y=1|x) = 0.4$

**평균 확률 계산**:
$$P_{\text{ensemble}}(y=0|x) = \frac{0.8 + 0.5 + 0.6}{3} = \frac{1.9}{3} = 0.633$$
$$P_{\text{ensemble}}(y=1|x) = \frac{0.2 + 0.5 + 0.4}{3} = \frac{1.1}{3} = 0.367$$

**최종 예측**: Class 0 (0.633 > 0.367)

### 💻 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# 한글 폰트 설정
rc('font', family='DejaVu Sans')

# 데이터 생성
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 개별 모델 학습
lr = LogisticRegression(random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

# 보팅 앙상블
voting_hard = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('knn', knn)],
    voting='hard'
)
voting_soft = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('knn', knn)],
    voting='soft'
)

voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

# 결정 경계 시각화
def plot_decision_boundary(model, X, y, title, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
              edgecolor='black', s=30, alpha=0.6)
    ax.set_title(title, fontsize=11, weight='bold')
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)

# 5개 서브플롯 생성
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

models = [
    (lr, 'Logistic Regression'),
    (svm, 'SVM (RBF Kernel)'),
    (knn, 'K-Nearest Neighbors'),
    (voting_hard, 'Hard Voting'),
    (voting_soft, 'Soft Voting')
]

for idx, (model, title) in enumerate(models):
    row = idx // 3
    col = idx % 3
    plot_decision_boundary(model, X_train, y_train, title, axes[row, col])
    
    # 정확도 표시
    score = model.score(X_test, y_test)
    axes[row, col].text(0.5, 0.05, f'Test Acc: {score:.3f}', 
                       transform=axes[row, col].transAxes,
                       ha='center', fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 마지막 서브플롯 제거
axes[1, 2].remove()

# 정확도 비교 바 차트 추가
ax_bar = fig.add_subplot(2, 3, 6)
model_names = ['LR', 'SVM', 'KNN', 'Hard\nVoting', 'Soft\nVoting']
accuracies = [model.score(X_test, y_test) for model, _ in models]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
bars = ax_bar.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)

# 값 표시
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.3f}',
               ha='center', va='bottom', fontsize=10, weight='bold')

ax_bar.set_ylabel('Test Accuracy', fontsize=11, weight='bold')
ax_bar.set_title('Model Performance Comparison', fontsize=12, weight='bold')
ax_bar.set_ylim([0.8, 1.0])
ax_bar.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('voting_ensemble.png', dpi=150, bbox_inches='tight')
plt.show()

# 성능 비교 출력
print("=" * 70)
print("VOTING ENSEMBLE PERFORMANCE COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'Train Accuracy':<20} {'Test Accuracy':<20}")
print("-" * 70)
for model, name in models:
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"{name:<25} {train_acc:<20.4f} {test_acc:<20.4f}")
print("=" * 70)
```

**기대 출력**:
- 각 모델의 서로 다른 결정 경계
- 보팅 앙상블의 더 안정적이고 부드러운 경계
- 소프트 보팅이 일반적으로 하드 보팅보다 약간 더 좋은 성능

### 💬 퍼실리테이션 질문

**질문 1**: "투표 규칙을 바꾸면 결과는 어떻게 달라질까요?"

**답변 가이드**:
- **하드 보팅**: 간단하지만 확률 정보 손실
- **소프트 보팅**: 확률을 활용하여 더 정교하지만, 모든 모델이 `predict_proba` 지원 필요
- **가중 보팅**: 성능이 좋은 모델에 더 높은 가중치 부여 가능

**질문 2**: "어떤 종류의 모델들을 조합하는 것이 좋을까요?"

**답변 가이드**:
- **다양성(Diversity)이 중요**: 서로 다른 가정을 가진 모델 조합
- 예: 선형 모델(LR) + 비선형 모델(SVM) + 인스턴스 기반(KNN)
- 너무 비슷한 모델들은 효과 감소

### 🧮 루브릭 평가표

| 평가항목 | 우수 (3점) | 보통 (2점) | 미흡 (1점) |
|----------|-----------|-----------|-----------|
| **보팅 원리 이해** | 하드/소프트 보팅의 차이를 수식과 예시로 명확히 설명 | 두 방식의 차이는 알지만 구체적 설명 부족 | 개념 이해가 불명확 |
| **시각화 정확성** | 결정 경계와 성능 차이를 정확히 해석 | 시각화 결과를 보지만 해석이 피상적 | 시각화 이해 부족 |
| **코드 구현** | VotingClassifier를 올바르게 사용하고 파라미터 이해 | 코드는 실행되지만 파라미터 이해 부족 | 코드 실행 실패 또는 이해 부족 |

---

## 3️⃣ 배깅 (Bagging)

### 📖 이론 설명

#### 3.1 배깅의 개념

**Bagging** = **B**ootstrap **Agg**regat**ing**

배깅은 **같은 알고리즘**을 **다른 데이터셋(부트스트랩 샘플)**으로 여러 번 학습시켜 결합하는 방법입니다.

**핵심 아이디어**:
- 데이터의 다른 부분 집합으로 여러 모델 학습
- 각 모델의 예측을 평균(회귀) 또는 투표(분류)로 결합
- **분산 감소**가 주요 목표

#### 3.2 부트스트랩 샘플링 (Bootstrap Sampling)

부트스트랩은 **복원 추출(sampling with replacement)**로 원본 데이터와 같은 크기의 샘플을 만드는 방법입니다.

**과정**:
1. 원본 데이터셋에서 무작위로 하나의 샘플 선택
2. 선택한 샘플을 다시 데이터셋에 넣음 (복원)
3. 1-2를 원본 데이터 크기만큼 반복

**결과**:
- 어떤 샘플은 여러 번 선택됨
- 어떤 샘플은 한 번도 선택되지 않음 (~36.8%)
- 각 부트스트랩 샘플은 서로 다름

#### 3.3 Random Forest

가장 유명한 배깅 알고리즘은 **Random Forest**입니다.

**Random Forest = Bagging + 특성 랜덤 선택**

추가 기법:
- 각 분할에서 전체 특성 중 일부만 고려
- 트리 간 상관성 감소 → 더 큰 다양성

### 🔢 수식

#### 부트스트랩 샘플링 확률

원본 데이터에 N개의 샘플이 있을 때, 특정 샘플이 부트스트랩 샘플에 **포함되지 않을 확률**:

$$P(\text{not selected}) = \left(1 - \frac{1}{N}\right)^N$$

N이 충분히 클 때:

$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = e^{-1} \approx 0.368$$

**해석**: 각 부트스트랩 샘플은 원본 데이터의 약 63.2%만 포함

#### 배깅 예측

**분류 문제**:
$$\hat{y} = \text{mode}\{h_1(x), h_2(x), \ldots, h_M(x)\}$$

**회귀 문제**:
$$\hat{y} = \frac{1}{M} \sum_{i=1}^{M} h_i(x)$$

여기서:
- $h_i(x)$: i번째 부트스트랩 샘플로 학습한 모델의 예측
- $M$: 부트스트랩 샘플(모델) 개수

### 💻 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA

# 한글 폰트 설정
rc('font', family='DejaVu Sans')

# Wine 데이터셋 로드
wine = load_wine()
X, y = wine.data, wine.target

# PCA로 2차원으로 축소 (시각화를 위해)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

# 모델 정의
single_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)

single_tree.fit(X_train, y_train)
bagging.fit(X_train, y_train)

# 부트스트랩 샘플 시각화
fig = plt.figure(figsize=(16, 10))

# 1. 원본 데이터
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                     cmap='viridis', s=100, alpha=0.6, edgecolor='black')
ax1.set_title('Original Training Data', fontsize=12, weight='bold')
ax1.set_xlabel('PC1', fontsize=10)
ax1.set_ylabel('PC2', fontsize=10)
plt.colorbar(scatter, ax=ax1, label='Class')

# 2-4. 부트스트랩 샘플 3개
np.random.seed(42)
for i in range(3):
    ax = plt.subplot(2, 3, i + 2)
    
    # 부트스트랩 샘플링
    n_samples = len(X_train)
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot = X_train[bootstrap_indices]
    y_boot = y_train[bootstrap_indices]
    
    # 시각화
    scatter = ax.scatter(X_boot[:, 0], X_boot[:, 1], c=y_boot, 
                        cmap='viridis', s=100, alpha=0.6, edgecolor='black')
    ax.set_title(f'Bootstrap Sample {i+1}', fontsize=12, weight='bold')
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    plt.colorbar(scatter, ax=ax, label='Class')

# 5-6. 결정 경계 비교
def plot_decision_boundary(model, X, y, title, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
              edgecolor='black', s=50, alpha=0.7)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)

ax5 = plt.subplot(2, 3, 5)
plot_decision_boundary(single_tree, X_train, y_train, 
                      f'Single Tree (Test Acc: {single_tree.score(X_test, y_test):.3f})', ax5)

ax6 = plt.subplot(2, 3, 6)
plot_decision_boundary(bagging, X_train, y_train, 
                      f'Bagging (Test Acc: {bagging.score(X_test, y_test):.3f})', ax6)

plt.tight_layout()
plt.savefig('bagging_bootstrap.png', dpi=150, bbox_inches='tight')
plt.show()

# 모델 수에 따른 성능 변화
n_estimators_range = range(1, 101, 5)
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_est,
        random_state=42
    )
    bag.fit(X_train, y_train)
    train_scores.append(bag.score(X_train, y_train))
    test_scores.append(bag.score(X_test, y_test))

# 성능 곡선 그리기
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, label='Train Accuracy', 
         marker='o', linewidth=2, markersize=6)
plt.plot(n_estimators_range, test_scores, label='Test Accuracy', 
         marker='s', linewidth=2, markersize=6)
plt.axhline(y=single_tree.score(X_test, y_test), color='r', 
            linestyle='--', label='Single Tree (Test)', linewidth=2)
plt.xlabel('Number of Estimators', fontsize=12, weight='bold')
plt.ylabel('Accuracy', fontsize=12, weight='bold')
plt.title('Bagging Performance vs Number of Estimators', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bagging_performance.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 70)
print("BAGGING ANALYSIS")
print("=" * 70)
print(f"Single Decision Tree:")
print(f"  Train Accuracy: {single_tree.score(X_train, y_train):.4f}")
print(f"  Test Accuracy:  {single_tree.score(X_test, y_test):.4f}")
print(f"\nBagging (100 estimators):")
print(f"  Train Accuracy: {bagging.score(X_train, y_train):.4f}")
print(f"  Test Accuracy:  {bagging.score(X_test, y_test):.4f}")
print(f"\nImprovement: {(bagging.score(X_test, y_test) - single_tree.score(X_test, y_test)) * 100:.2f}%")
print("=" * 70)
```

**기대 출력**:
- 각 부트스트랩 샘플의 서로 다른 분포
- 단일 트리: 과적합된 복잡한 경계
- 배깅: 부드럽고 일반화된 경계
- 모델 수 증가 → 성능 향상 → 포화

### 💬 퍼실리테이션 질문

**질문 1**: "모델 수를 늘리면 왜 분산이 줄어들까요?"

**답변 가이드**:
- 각 모델의 무작위성이 평균으로 상쇄됨
- 중심극한정리: 평균의 분산은 $\sigma^2/N$로 감소
- 그래프에서 테스트 정확도가 안정화되는 것 확인

**질문 2**: "부트스트랩 샘플이 원본과 다른 이유는 무엇일까요?"

**답변 가이드**:
- 복원 추출로 일부 샘플은 중복, 일부는 누락
- 각 샘플은 원본의 약 63.2%만 포함
- 이 차이가 모델의 다양성을 만듦

### 🧮 루브릭 평가표

| 평가항목 | 우수 (3점) | 보통 (2점) | 미흡 (1점) |
|----------|-----------|-----------|-----------|
| **부트스트랩 이해** | 부트스트랩 샘플링 원리와 확률 계산을 정확히 설명 | 개념은 이해하지만 수학적 설명 부족 | 부트스트랩 개념 이해 부족 |
| **실험 설계** | 다양한 파라미터로 체계적인 실험 수행 | 기본 실험은 하지만 심화 분석 부족 | 실험 설계 미흡 |
| **시각화 명확성** | 부트스트랩 샘플과 성능 변화를 명확히 시각화 | 시각화는 있지만 해석 부족 | 시각화 품질 낮음 |

---

## 4️⃣ 에이다부스트 (AdaBoost)

### 📖 이론 설명

#### 4.1 부스팅의 개념

**Boosting**은 **약한 학습기(weak learner)**들을 **순차적으로** 결합하여 강한 학습기를 만드는 방법입니다.

**핵심 아이디어**:
- 이전 모델이 틀린 샘플에 더 집중
- 각 모델은 이전 모델의 실수를 보완
- **편향 감소**가 주요 목표

**배깅 vs 부스팅**:
| 특성 | 배깅 | 부스팅 |
|------|------|--------|
| 학습 방식 | 병렬 (독립적) | 순차적 (의존적) |
| 데이터 샘플링 | 부트스트랩 | 가중치 조정 |
| 주요 목표 | 분산 감소 | 편향 감소 |
| 대표 알고리즘 | Random Forest | AdaBoost, Gradient Boosting |

#### 4.2 AdaBoost 알고리즘

**AdaBoost** = **Ada**ptive **Boost**ing

AdaBoost는 가장 대표적인 부스팅 알고리즘입니다.

**작동 원리**:
1. 모든 샘플에 동일한 가중치 부여
2. 약한 학습기 학습
3. 잘못 분류된 샘플의 가중치 증가
4. 2-3 반복
5. 최종적으로 모든 학습기를 가중 투표로 결합

#### 4.3 약한 학습기 (Weak Learner)

**약한 학습기**: 무작위 추측보다 조금 나은 모델
- 일반적으로 깊이가 1인 결정 트리 (decision stump) 사용
- 정확도 50% 이상이면 충분

**왜 약한 학습기를 사용할까?**
- 과적합 방지
- 빠른 학습
- 순차적 결합으로 점진적 개선

### 🔢 수식

#### AdaBoost 알고리즘 상세

**초기화** (t=0):
$$w_i^{(0)} = \frac{1}{N}, \quad i = 1, 2, \ldots, N$$

여기서 $w_i$는 i번째 샘플의 가중치, N은 전체 샘플 수

**반복** (t = 1, 2, ..., T):

**1단계**: 가중치를 사용하여 약한 학습기 $h_t$ 학습

**2단계**: 가중 오차율 계산
$$\epsilon_t = \frac{\sum_{i=1}^{N} w_i^{(t-1)} \mathbb{1}(h_t(x_i) \neq y_i)}{\sum_{i=1}^{N} w_i^{(t-1)}}$$

여기서 $\mathbb{1}$은 지시함수 (조건이 참이면 1, 거짓이면 0)

**3단계**: 학습기 가중치 계산
$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

**통찰**:
- $\epsilon_t < 0.5$: $\alpha_t > 0$ (좋은 모델은 높은 가중치)
- $\epsilon_t = 0.5$: $\alpha_t = 0$ (무작위 추측 수준)
- $\epsilon_t > 0.5$: $\alpha_t < 0$ (나쁜 모델은 음의 가중치)

**4단계**: 샘플 가중치 업데이트
$$w_i^{(t)} = w_i^{(t-1)} \exp\left(-\alpha_t y_i h_t(x_i)\right)$$

**정규화**:
$$w_i^{(t)} = \frac{w_i^{(t)}}{\sum_{j=1}^{N} w_j^{(t)}}$$

**최종 예측**:
$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

#### 단계별 계산 예제

간단한 예제로 AdaBoost를 이해해봅시다.

**데이터**: 5개 샘플
| 샘플 | 특성 | 실제 레이블 |
|------|------|------------|
| 1 | x₁ | +1 |
| 2 | x₂ | +1 |
| 3 | x₃ | -1 |
| 4 | x₄ | -1 |
| 5 | x₅ | +1 |

**라운드 1**:
- 초기 가중치: $w_1 = w_2 = w_3 = w_4 = w_5 = 0.2$
- 학습기 $h_1$ 예측: [+1, +1, +1, -1, -1]
- 오분류: 샘플 3, 5
- 오차율: $\epsilon_1 = (0.2 + 0.2) / 1.0 = 0.4$
- 모델 가중치: $\alpha_1 = 0.5 \ln(0.6/0.4) = 0.203$
- 가중치 업데이트:
  - 샘플 3: $w_3 = 0.2 \times \exp(0.203) = 0.245$
  - 샘플 5: $w_5 = 0.2 \times \exp(0.203) = 0.245$
  - 나머지: $w = 0.2 \times \exp(-0.203) = 0.163$
- 정규화 후: [0.163, 0.163, 0.245, 0.163, 0.245]

**라운드 2**:
- 이제 샘플 3, 5에 더 집중하여 학습
- 과정 반복...

### 💻 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# 한글 폰트 설정
rc('font', family='DejaVu Sans')

# 데이터 생성
np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          flip_y=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# AdaBoost 학습
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)

# 단계별 학습 과정 시각화
n_estimators_list = [1, 5, 10, 20, 50]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, n_est in enumerate(n_estimators_list):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # 해당 단계까지의 모델
    ada_partial = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_est,
        learning_rate=1.0,
        random_state=42
    )
    ada_partial.fit(X_train, y_train)
    
    # 결정 경계
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = ada_partial.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
              cmap=plt.cm.RdYlBu, edgecolor='black', s=50, alpha=0.7)
    
    train_acc = ada_partial.score(X_train, y_train)
    test_acc = ada_partial.score(X_test, y_test)
    
    ax.set_title(f'n_estimators = {n_est}', fontsize=12, weight='bold')
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.text(0.5, 0.05, f'Test Acc: {test_acc:.3f}', 
           transform=ax.transAxes, ha='center', fontsize=10, weight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 마지막 서브플롯: 학습 곡선
ax_curve = axes[1, 2]
n_range = range(1, 51)
train_scores = []
test_scores = []

for n in n_range:
    ada_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        learning_rate=1.0,
        random_state=42
    )
    ada_temp.fit(X_train, y_train)
    train_scores.append(ada_temp.score(X_train, y_train))
    test_scores.append(ada_temp.score(X_test, y_test))

ax_curve.plot(n_range, train_scores, label='Train Accuracy', 
             marker='o', linewidth=2, markersize=4)
ax_curve.plot(n_range, test_scores, label='Test Accuracy', 
             marker='s', linewidth=2, markersize=4)
ax_curve.set_xlabel('Number of Estimators', fontsize=10, weight='bold')
ax_curve.set_ylabel('Accuracy', fontsize=10, weight='bold')
ax_curve.set_title('Learning Curve', fontsize=12, weight='bold')
ax_curve.legend(fontsize=9)
ax_curve.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adaboost_stages.png', dpi=150, bbox_inches='tight')
plt.show()

# 오차 감소 시각화
plt.figure(figsize=(12, 5))

# 왼쪽: 단계별 오차율
plt.subplot(1, 2, 1)
estimator_errors = ada.estimator_errors_[:30]  # 처음 30개만
plt.plot(range(1, len(estimator_errors) + 1), estimator_errors, 
         marker='o', linewidth=2, markersize=6, color='coral')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess', linewidth=2)
plt.xlabel('Iteration', fontsize=12, weight='bold')
plt.ylabel('Weighted Error Rate', fontsize=12, weight='bold')
plt.title('AdaBoost: Error Rate per Iteration', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 오른쪽: 학습기 가중치
plt.subplot(1, 2, 2)
estimator_weights = ada.estimator_weights_[:30]
plt.bar(range(1, len(estimator_weights) + 1), estimator_weights, 
        color='skyblue', edgecolor='black', linewidth=1.5)
plt.xlabel('Iteration', fontsize=12, weight='bold')
plt.ylabel('Estimator Weight (α)', fontsize=12, weight='bold')
plt.title('AdaBoost: Learner Weights per Iteration', fontsize=14, weight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('adaboost_errors_weights.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 70)
print("ADABOOST ANALYSIS")
print("=" * 70)
print(f"Number of Estimators: {ada.n_estimators}")
print(f"Training Accuracy: {ada.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {ada.score(X_test, y_test):.4f}")
print(f"\nFirst 10 Estimator Errors:")
for i, error in enumerate(ada.estimator_errors_[:10], 1):
    print(f"  Round {i:2d}: {error:.4f}")
print(f"\nFirst 10 Estimator Weights:")
for i, weight in enumerate(ada.estimator_weights_[:10], 1):
    print(f"  Round {i:2d}: {weight:.4f}")
print("=" * 70)
```

**기대 출력**:
- 초기에는 단순한 결정 경계
- 반복이 진행되면서 복잡해지고 정확해짐
- 오차율이 점진적으로 감소
- 좋은 학습기는 높은 가중치를 받음

### 💬 퍼실리테이션 질문

**질문 1**: "왜 오차가 큰 샘플의 가중치를 높이는 걸까요?"

**답변 가이드**:
- 잘못 분류된 샘플은 어려운 케이스
- 다음 모델이 이 샘플에 집중하도록 유도
- 점진적으로 어려운 문제 해결
- 전체적인 성능 향상

**질문 2**: "AdaBoost가 과적합될 수 있을까요?"

**답변 가이드**:
- 이론적으로 부스팅은 과적합에 강함
- 하지만 노이즈가 많으면 과적합 가능
- 학습률(learning_rate) 조정으로 완화
- 너무 많은 반복은 피해야 함

### 🧮 루브릭 평가표

| 평가항목 | 우수 (3점) | 보통 (2점) | 미흡 (1점) |
|----------|-----------|-----------|-----------|
| **알고리즘 이해** | 가중치 업데이트 과정을 수식과 함께 명확히 설명 | 개념은 이해하지만 수식 설명 부족 | 알고리즘 이해 부족 |
| **해석 정확도** | 오차율과 가중치 변화를 정확히 해석 | 그래프는 보지만 해석이 피상적 | 결과 해석 불가 |
| **코드 구현** | AdaBoostClassifier를 올바르게 사용하고 파라미터 이해 | 코드 실행은 되지만 파라미터 이해 부족 | 코드 실행 실패 |

---

## 5️⃣ 그레이디언트 부스팅 & XGBoost

### 📖 이론 설명

#### 5.1 그레이디언트 부스팅의 개념

**Gradient Boosting**은 손실 함수의 **그레이디언트(기울기)**를 이용하여 부스팅하는 방법입니다.

**핵심 아이디어**:
- 이전 모델의 **잔차(residual)**를 다음 모델이 학습
- 경사하강법처럼 손실 함수를 최소화하는 방향으로 학습
- 매우 강력하지만 과적합 위험

**AdaBoost vs Gradient Boosting**:
| 특성 | AdaBoost | Gradient Boosting |
|------|----------|------------------|
| 가중치 조정 | 샘플 가중치 | 잔차 학습 |
| 손실 함수 | 지수 손실 | 임의의 미분 가능 손실 |
| 유연성 | 분류에 특화 | 분류/회귀 모두 가능 |

#### 5.2 그레이디언트 부스팅 알고리즘

**과정**:
1. 초기 예측 $F_0(x)$ 설정 (보통 평균값)
2. 현재 모델의 잔차 계산: $r_i = y_i - F_{m-1}(x_i)$
3. 잔차를 타겟으로 새로운 모델 $h_m$ 학습
4. 모델 업데이트: $F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$
5. 2-4 반복

여기서 $\nu$는 학습률(learning rate)

#### 5.3 XGBoost (eXtreme Gradient Boosting)

XGBoost는 Gradient Boosting의 **최적화된 구현**입니다.

**주요 개선점**:
- **정규화**: L1, L2 정규화로 과적합 방지
- **병렬 처리**: 트리 구축 시 병렬화로 속도 향상
- **결측치 처리**: 자동으로 최적의 방향 학습
- **가지치기**: 손실 감소가 없으면 분할 중단
- **조기 종료**: 검증 성능이 악화되면 학습 중단

#### 5.4 학습률의 역할

**학습률(Learning Rate, $\nu$)**:
- 각 트리의 기여도 조절
- 낮은 학습률: 천천히 학습, 더 많은 트리 필요, 과적합 방지
- 높은 학습률: 빠르게 학습, 적은 트리, 과적합 위험

**일반적인 전략**:
- $\nu = 0.1$: 중간 속도, 균형잡힌 성능
- $\nu = 0.01$: 느린 학습, 높은 정확도
- 트리 수와 반비례: 학습률 ↓ → 트리 수 ↑

### 🔢 수식

#### 그레이디언트 부스팅 수식

**목표**: 손실 함수 $L$을 최소화하는 함수 $F$를 찾기

$$F^* = \arg\min_F \sum_{i=1}^{N} L(y_i, F(x_i))$$

**초기화**:
$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{N} L(y_i, \gamma)$$

회귀 문제에서 평균제곱오차 사용 시: $F_0(x) = \bar{y}$

**반복** (m = 1, 2, ..., M):

**1단계**: 음의 그레이디언트 (의사 잔차) 계산
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

평균제곱오차의 경우: $r_{im} = y_i - F_{m-1}(x_i)$

**2단계**: 잔차에 대해 회귀 트리 $h_m$ 학습
$$h_m = \arg\min_h \sum_{i=1}^{N} (r_{im} - h(x_i))^2$$

**3단계**: 모델 업데이트
$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

**최종 모델**:
$$F_M(x) = F_0(x) + \nu \sum_{m=1}^{M} h_m(x)$$

#### XGBoost 목적 함수

XGBoost는 다음을 최소화:

$$\text{Obj}^{(t)} = \sum_{i=1}^{N} L(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)$$

여기서:
- $L$: 손실 함수
- $\Omega(f)$: 정규화 항 (복잡도 제어)
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$
  - $T$: 리프 노드 수
  - $w_j$: 리프 j의 가중치
  - $\gamma$, $\lambda$: 정규화 파라미터

### 💻 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Some visualizations will be skipped.")

# 한글 폰트 설정
rc('font', family='DejaVu Sans')

# ===== 회귀 문제: 잔차 학습 시각화 =====
np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(100, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, X_reg.shape[0])

# 간단한 그레이디언트 부스팅 구현 (설명용)
n_estimators = 5
learning_rate = 0.5

# 초기 예측: 평균
predictions = np.full(len(y_reg), np.mean(y_reg))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(n_estimators):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # 잔차 계산
    residuals = y_reg - predictions
    
    # 잔차에 대해 트리 학습
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X_reg, residuals)
    
    # 예측 업데이트
    predictions += learning_rate * tree.predict(X_reg)
    
    # 시각화
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    y_plot = np.full(len(X_plot), np.mean(y_reg))
    
    # 현재까지의 예측
    temp_pred = np.full(len(X_plot), np.mean(y_reg))
    for j in range(i + 1):
        if j == 0:
            residuals_temp = y_reg - temp_pred[:len(y_reg)]
            tree_temp = DecisionTreeRegressor(max_depth=3, random_state=42+j)
            tree_temp.fit(X_reg, residuals_temp)
            temp_pred += learning_rate * tree_temp.predict(X_plot)
        else:
            # 재학습 과정
            pass
    
    ax.scatter(X_reg, y_reg, alpha=0.5, s=30, label='Data')
    ax.plot(X_plot, temp_pred, 'r-', linewidth=2, label=f'Prediction (step {i+1})')
    ax.plot(X_reg, predictions, 'g.', markersize=8, label='Current fit', alpha=0.7)
    ax.set_title(f'Boosting Round {i+1}', fontsize=12, weight='bold')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 마지막 서브플롯: 최종 결과
ax = axes[1, 2]
gb_reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, 
                                   max_depth=3, random_state=42)
gb_reg.fit(X_reg, y_reg)
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
y_plot = gb_reg.predict(X_plot)

ax.scatter(X_reg, y_reg, alpha=0.5, s=30, label='Data')
ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='GB (50 estimators)')
ax.plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, label='True function', alpha=0.7)
ax.set_title('Final Gradient Boosting Model', fontsize=12, weight='bold')
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_boosting_residuals.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== 분류 문제: GB vs XGBoost 비교 =====
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, 
                                   n_informative=15, n_redundant=5,
                                   random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, 
                                                    test_size=0.3, random_state=42)

# 학습률 효과 분석
learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

plt.figure(figsize=(14, 6))

# 왼쪽: 학습률별 성능
plt.subplot(1, 2, 1)
for lr, color in zip(learning_rates, colors):
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, 
                                   max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    
    # 학습 과정의 누적 점수
    train_scores = []
    test_scores = []
    for i, (train_pred, test_pred) in enumerate(zip(
        gb.staged_predict(X_train), gb.staged_predict(X_test))):
        train_scores.append(np.mean(train_pred == y_train))
        test_scores.append(np.mean(test_pred == y_test))
    
    plt.plot(range(1, len(test_scores) + 1), test_scores, 
            label=f'LR = {lr}', linewidth=2, color=color)

plt.xlabel('Number of Estimators', fontsize=12, weight='bold')
plt.ylabel('Test Accuracy', fontsize=12, weight='bold')
plt.title('Effect of Learning Rate on GB Performance', fontsize=14, weight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 오른쪽: GB vs XGBoost 비교
plt.subplot(1, 2, 2)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                               max_depth=3, random_state=42)
gb.fit(X_train, y_train)

models = [('Gradient Boosting', gb)]
model_names = ['GB']
train_accs = [gb.score(X_train, y_train)]
test_accs = [gb.score(X_test, y_test)]

if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, 
                       max_depth=3, random_state=42, use_label_encoder=False,
                       eval_metric='logloss')
    xgb.fit(X_train, y_train)
    models.append(('XGBoost', xgb))
    model_names.append('XGB')
    train_accs.append(xgb.score(X_train, y_train))
    test_accs.append(xgb.score(X_test, y_test))

x = np.arange(len(model_names))
width = 0.35

bars1 = plt.bar(x - width/2, train_accs, width, label='Train', 
               color='skyblue', edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x + width/2, test_accs, width, label='Test', 
               color='coral', edgecolor='black', linewidth=1.5)

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

plt.xlabel('Model', fontsize=12, weight='bold')
plt.ylabel('Accuracy', fontsize=12, weight='bold')
plt.title('Gradient Boosting vs XGBoost', fontsize=14, weight='bold')
plt.xticks(x, model_names)
plt.legend(fontsize=10)
plt.ylim([0.85, 1.0])
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_boosting_xgboost.png', dpi=150, bbox_inches='tight')
plt.show()

# 특성 중요도
plt.figure(figsize=(10, 6))
feature_importance = gb.feature_importances_
sorted_idx = np.argsort(feature_importance)[-10:]  # 상위 10개

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], 
        color='teal', edgecolor='black', linewidth=1.5)
plt.yticks(range(len(sorted_idx)), [f'Feature {i}' for i in sorted_idx])
plt.xlabel('Importance', fontsize=12, weight='bold')
plt.title('Top 10 Feature Importances (Gradient Boosting)', fontsize=14, weight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 70)
print("GRADIENT BOOSTING & XGBOOST COMPARISON")
print("=" * 70)
print(f"Gradient Boosting:")
print(f"  Train Accuracy: {gb.score(X_train, y_train):.4f}")
print(f"  Test Accuracy:  {gb.score(X_test, y_test):.4f}")
if XGBOOST_AVAILABLE:
    print(f"\nXGBoost:")
    print(f"  Train Accuracy: {xgb.score(X_train, y_train):.4f}")
    print(f"  Test Accuracy:  {xgb.score(X_test, y_test):.4f}")
print("\nLearning Rate Effects:")
for lr in learning_rates:
    gb_temp = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, 
                                        max_depth=3, random_state=42)
    gb_temp.fit(X_train, y_train)
    print(f"  LR = {lr:4.2f}: Test Acc = {gb_temp.score(X_test, y_test):.4f}")
print("=" * 70)
```

**기대 출력**:
- 잔차 학습 과정의 단계별 시각화
- 학습률에 따른 성능 변화 곡선
- GB와 XGBoost의 성능 비교
- 특성 중요도 시각화

### 💬 퍼실리테이션 질문

**질문 1**: "학습률이 너무 크거나 작으면 어떤 문제가 생길까요?"

**답변 가이드**:
- **너무 큰 학습률 (예: 1.0)**:
  - 빠르게 수렴하지만 과적합 위험
  - 최적점을 넘어설 수 있음
  - 불안정한 학습
- **너무 작은 학습률 (예: 0.001)**:
  - 천천히 수렴, 매우 많은 트리 필요
  - 학습 시간 증가
  - 과소적합 위험
- **적절한 균형** (예: 0.1): 안정적이고 효율적

**질문 2**: "XGBoost가 일반 Gradient Boosting보다 좋은 이유는?"

**답변 가이드**:
- 정규화로 과적합 방지
- 병렬 처리로 속도 향상
- 결측치 자동 처리
- 조기 종료로 효율성 증가
- 대규모 데이터에 적합

### 🧮 루브릭 평가표

| 평가항목 | 우수 (3점) | 보통 (2점) | 미흡 (1점) |
|----------|-----------|-----------|-----------|
| **모델 비교 분석** | GB와 XGBoost의 차이를 이론과 실험으로 명확히 설명 | 기본 차이는 알지만 깊은 분석 부족 | 차이점 이해 부족 |
| **설명 명확성** | 잔차 학습과 학습률의 역할을 명확히 설명 | 개념은 이해하지만 설명이 불완전 | 핵심 개념 이해 부족 |
| **파라미터 튜닝** | 학습률과 트리 수의 관계를 이해하고 최적값 탐색 | 기본 파라미터만 사용 | 파라미터 의미 모름 |

---

## 6️⃣ 모델 성능 평가 및 비교

### 📖 이론 설명

#### 6.1 앙상블 방식별 특징 요약

각 앙상블 방식은 서로 다른 강점과 약점을 가지고 있습니다.

| 앙상블 방식 | 주요 목적 | 장점 | 단점 | 적합한 상황 |
|------------|----------|------|------|-----------|
| **Voting** | 다양성 활용 | 간단, 해석 가능 | 모델 선택에 의존 | 서로 다른 알고리즘 조합 |
| **Bagging** | 분산 감소 | 과적합 방지, 병렬 가능 | 편향 감소 어려움 | 고분산 모델(깊은 트리) |
| **Random Forest** | 분산 감소 + 다양성 | 안정적, 특성 중요도 | 해석력 낮음 | 범용적 사용 |
| **AdaBoost** | 편향 감소 | 약한 학습기 강화 | 노이즈에 민감 | 단순 모델 개선 |
| **Gradient Boosting** | 편향 감소 | 높은 정확도 | 과적합 위험, 느림 | 정확도 최우선 |
| **XGBoost** | 편향 감소 + 정규화 | 매우 높은 성능, 빠름 | 하이퍼파라미터 많음 | 대규모 데이터, 경진대회 |

#### 6.2 평가 지표

**분류 문제**:
- **정확도(Accuracy)**: 전체 중 맞춘 비율
- **정밀도(Precision)**: 양성 예측 중 실제 양성 비율
- **재현율(Recall)**: 실제 양성 중 양성으로 예측한 비율
- **F1 점수**: 정밀도와 재현율의 조화평균
- **ROC-AUC**: 거짓양성률 대비 참양성률

**회귀 문제**:
- **MSE (Mean Squared Error)**: 평균 제곱 오차
- **RMSE (Root Mean Squared Error)**: MSE의 제곱근
- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **R² 점수**: 결정계수 (1에 가까울수록 좋음)

#### 6.3 모델 선택 가이드

**프로젝트 요구사항에 따른 선택**:

1. **해석 가능성이 중요**: Voting, Single Tree
2. **높은 정확도 필요**: XGBoost, Gradient Boosting
3. **빠른 학습 필요**: Random Forest (병렬), Voting
4. **적은 데이터**: Bagging, Random Forest
5. **많은 데이터**: XGBoost, Gradient Boosting
6. **노이즈가 많음**: Random Forest, Bagging
7. **불균형 데이터**: 가중치 조정 + XGBoost

### 🔢 수식

#### 교차 검증 점수

K-Fold 교차 검증의 평균 점수:

$$\text{CV Score} = \frac{1}{K} \sum_{k=1}^{K} \text{Score}_k$$

여기서:
- $K$: 폴드 수 (일반적으로 5 또는 10)
- $\text{Score}_k$: k번째 폴드의 성능 점수

#### F1 점수

정밀도와 재현율의 조화평균:

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

여기서:
- TP (True Positive): 참 양성
- FP (False Positive): 거짓 양성
- FN (False Negative): 거짓 음성

### 💻 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, 
                             RandomForestClassifier, AdaBoostClassifier, 
                             GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix)
import time

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# 한글 폰트 설정
rc('font', family='DejaVu Sans')

# 데이터 로드
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

# 모델 정의
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Voting': VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    ),
    'Bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, 
                                                    random_state=42)
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42,
                                     use_label_encoder=False, eval_metric='logloss')

# 성능 평가
results = {
    'Model': [],
    'Train Acc': [],
    'Test Acc': [],
    'Precision': [],
    'Recall': [],
    'F1': [],
    'ROC-AUC': [],
    'Train Time (s)': []
}

for name, model in models.items():
    print(f"Training {name}...")
    
    # 학습 시간 측정
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 메트릭 계산
    results['Model'].append(name)
    results['Train Acc'].append(accuracy_score(y_train, model.predict(X_train)))
    results['Test Acc'].append(accuracy_score(y_test, y_pred))
    results['Precision'].append(precision_score(y_test, y_pred))
    results['Recall'].append(recall_score(y_test, y_pred))
    results['F1'].append(f1_score(y_test, y_pred))
    results['ROC-AUC'].append(roc_auc_score(y_test, y_proba) if y_proba is not None else 0)
    results['Train Time (s)'].append(train_time)

# 결과를 DataFrame으로
df_results = pd.DataFrame(results)
print("\n" + "=" * 100)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 100)
print(df_results.to_string(index=False))
print("=" * 100)

# 시각화 1: 성능 비교 바 차트
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Test Acc', 'F1', 'ROC-AUC', 'Train Time (s)']
titles = ['Test Accuracy', 'F1 Score', 'ROC-AUC Score', 'Training Time (seconds)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    sorted_df = df_results.sort_values(metric, ascending=(metric == 'Train Time (s)'))
    
    bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=color, 
                   edgecolor='black', linewidth=1.5)
    
    # 값 표시
    for bar, value in zip(bars, sorted_df[metric]):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{value:.3f}' if metric != 'Train Time (s)' else f'{value:.2f}s',
               ha='left', va='center', fontsize=9, weight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(title, fontsize=11, weight='bold')
    ax.set_title(f'{title} Comparison', fontsize=12, weight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 시각화 2: 정확도-시간 트레이드오프
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_results['Train Time (s)'], df_results['Test Acc'], 
                     s=200, c=df_results['F1'], cmap='viridis', 
                     edgecolor='black', linewidth=2, alpha=0.7)

# 모델 이름 표시
for idx, row in df_results.iterrows():
    plt.annotate(row['Model'], 
                (row['Train Time (s)'], row['Test Acc']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))

plt.colorbar(scatter, label='F1 Score')
plt.xlabel('Training Time (seconds)', fontsize=12, weight='bold')
plt.ylabel('Test Accuracy', fontsize=12, weight='bold')
plt.title('Accuracy vs Training Time Trade-off', fontsize=14, weight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_time_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()

# 시각화 3: 레이더 차트 (상위 5개 모델)
from math import pi

top_models = df_results.nlargest(5, 'Test Acc')

categories = ['Test Acc', 'Precision', 'Recall', 'F1', 'ROC-AUC']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for idx, row in top_models.iterrows():
    values = [row['Test Acc'], row['Precision'], row['Recall'], 
             row['F1'], row['ROC-AUC']]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, weight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
ax.grid(True)
ax.set_title('Top 5 Models - Multi-Metric Comparison', 
            fontsize=14, weight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('radar_chart_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# GridSearchCV를 활용한 하이퍼파라미터 튜닝 예제
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING EXAMPLE (Random Forest)")
print("=" * 70)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', 
                          n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# GridSearch 결과 시각화
cv_results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 6))

# n_estimators 효과
for depth in [None, 10, 20, 30]:
    mask = cv_results['param_max_depth'] == depth
    subset = cv_results[mask]
    plt.plot(subset['param_n_estimators'], subset['mean_test_score'], 
            marker='o', label=f'max_depth={depth}', linewidth=2)

plt.xlabel('Number of Estimators', fontsize=12, weight='bold')
plt.ylabel('Mean CV Accuracy', fontsize=12, weight='bold')
plt.title('GridSearch Results: n_estimators vs max_depth', fontsize=14, weight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gridsearch_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 70)
```

**기대 출력**:
- 모든 앙상블 방법의 상세 성능 비교 표
- 성능 지표별 바 차트
- 정확도-시간 트레이드오프 산점도
- 상위 모델의 레이더 차트
- GridSearch 하이퍼파라미터 튜닝 결과

### 💬 퍼실리테이션 질문

**질문**: "우리 팀이 선택한 최적의 앙상블 모델은 무엇이며, 그 이유는?"

**토론 가이드**:
1. **프로젝트 요구사항 분석**:
   - 정확도가 최우선인가?
   - 해석 가능성이 필요한가?
   - 실시간 예측이 필요한가?
   - 학습 시간 제약이 있는가?

2. **데이터 특성 고려**:
   - 데이터 크기는?
   - 노이즈 수준은?
   - 클래스 불균형 여부?
   - 특성 개수는?

3. **성능-복잡도 트레이드오프**:
   - 약간의 정확도 향상을 위해 복잡도 증가가 정당화되는가?
   - 유지보수 비용을 고려했는가?

4. **실험 결과 기반 의사결정**:
   - 교차 검증 점수가 안정적인가?
   - 테스트 성능이 학습 성능과 비슷한가? (과적합 확인)

### 🧮 루브릭 평가표

| 평가항목 | 우수 (3점) | 보통 (2점) | 미흡 (1점) |
|----------|-----------|-----------|-----------|
| **실험 설계** | 체계적인 비교 실험과 명확한 평가 기준 제시 | 기본 실험은 했으나 일부 메트릭 누락 | 실험 설계가 비체계적이거나 불완전 |
| **결과 해석** | 데이터 기반으로 모델 선택 근거를 명확히 설명 | 결과를 보고하지만 해석이 피상적 | 결과 해석 없이 수치만 나열 |
| **모델 선택 정당화** | 프로젝트 요구사항과 실험 결과를 종합하여 최적 모델 선택 | 일부 요소만 고려하여 모델 선택 | 근거 없는 주관적 선택 |
| **하이퍼파라미터 튜닝** | GridSearch/RandomSearch로 체계적 튜닝 수행 | 기본 파라미터만 사용하거나 수동 조정 | 파라미터 튜닝 없음 |
| **시각화 품질** | 다양한 관점의 명확하고 유용한 시각화 | 기본 시각화는 있으나 통찰력 부족 | 시각화 부족하거나 부적절 |

---

## 📚 학습 요약 및 실전 가이드

### ✅ 핵심 개념 체크리스트

**기본 개념**:
- [ ] 앙상블 학습의 기본 원리 이해
- [ ] 편향-분산 트레이드오프 개념 이해
- [ ] 배깅, 부스팅, 스태킹의 차이점 설명 가능

**개별 알고리즘**:
- [ ] Voting: 하드/소프트 보팅 차이 이해
- [ ] Bagging: 부트스트랩 샘플링 원리 이해
- [ ] Random Forest: 특성 랜덤 선택의 효과 이해
- [ ] AdaBoost: 샘플 가중치 업데이트 과정 이해
- [ ] Gradient Boosting: 잔차 학습 개념 이해
- [ ] XGBoost: 정규화와 최적화 기법 이해

**실전 스킬**:
- [ ] scikit-learn 앙상블 모델 사용 가능
- [ ] 하이퍼파라미터 튜닝 수행 가능
- [ ] 교차 검증으로 모델 평가 가능
- [ ] 적절한 평가 지표 선택 가능
- [ ] 시각화를 통한 결과 해석 가능

### 🎯 실전 프로젝트 가이드

#### 단계 1: 문제 정의
```python
# 1. 문제 유형 확인 (분류 vs 회귀)
# 2. 평가 지표 결정 (정확도, F1, MSE 등)
# 3. 성능 목표 설정
```

#### 단계 2: 데이터 준비
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### 단계 3: 베이스라인 모델
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 단순 모델로 시작
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
print(f"Baseline: {baseline_score:.4f}")
```

#### 단계 4: 앙상블 모델 실험
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.4f}")
```

#### 단계 5: 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.5]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

#### 단계 6: 최종 모델 평가
```python
from sklearn.metrics import classification_report, confusion_matrix

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### 💡 실무 팁

1. **항상 베이스라인부터**:
   - 단순 모델로 시작
   - 복잡한 모델은 나중에

2. **교차 검증 필수**:
   - 단일 train/test split은 불안정
   - 최소 5-fold CV 사용

3. **과적합 경계**:
   - Train/Test 성능 차이 모니터링
   - 정규화 파라미터 조정

4. **특성 엔지니어링 우선**:
   - 좋은 특성 > 복잡한 모델
   - 도메인 지식 활용

5. **앙상블의 앙상블**:
   - Voting으로 여러 앙상블 결합 가능
   - Stacking으로 메타 학습 가능

### 📖 추가 학습 자료

**온라인 리소스**:
- [scikit-learn 앙상블 가이드](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/)
- [Kaggle 앙상블 튜토리얼](https://www.kaggle.com/learn/intro-to-machine-learning)

**추천 논문**:
- Breiman, L. (1996). "Bagging Predictors"
- Freund, Y., & Schapire, R. E. (1997). "A Decision-Theoretic Generalization of On-Line Learning"
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"

**도서**:
- "Ensemble Methods: Foundations and Algorithms" - Zhi-Hua Zhou
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Aurélien Géron

---

## 🎓 종합 평가 과제

### 프로젝트: 와인 품질 예측 앙상블 시스템

**목표**: 와인의 화학적 특성으로 품질을 예측하는 최적의 앙상블 모델 구축

**데이터셋**: UCI Wine Quality Dataset (Red Wine)

**요구사항**:
1. 최소 5가지 앙상블 방법 비교
2. GridSearch로 최적 하이퍼파라미터 탐색
3. 교차 검증으로 성능 평가
4. 결과를 시각화로 표현
5. 최종 모델 선택 근거 작성

**평가 기준**:
- 실험 설계 체계성 (30%)
- 코드 품질 및 완성도 (25%)
- 시각화 명확성 (20%)
- 결과 분석 및 해석 (25%)

**제출물**:
- Python 코드 (.py 또는 .ipynb)
- 결과 리포트 (Markdown 또는 PDF)
- 시각화 이미지

---

## 🙏 마무리

축하합니다! 앙상블 학습의 핵심 개념과 실전 활용법을 모두 학습하셨습니다.

**여러분이 배운 내용**:
- ✅ 앙상블 학습의 이론적 배경 (편향-분산 트레이드오프)
- ✅ 5가지 주요 앙상블 알고리즘 (Voting, Bagging, AdaBoost, GB, XGBoost)
- ✅ 각 알고리즘의 수학적 원리와 작동 방식
- ✅ Python으로 앙상블 모델 구현 및 평가
- ✅ 하이퍼파라미터 튜닝 및 모델 선택 전략

**다음 단계**:
1. 실제 데이터셋으로 프로젝트 진행
2. Kaggle 경진대회 참여
3. 앙상블의 앙상블(Stacking) 학습
4. 딥러닝과 앙상블 결합

**마지막 조언**:
> "완벽한 모델은 없지만, 더 나은 모델은 항상 있습니다.  
> 계속 실험하고, 학습하고, 개선하세요!"

행운을 빕니다! 🚀

---

**문서 작성 정보**:
- 작성일: 2024
- 버전: 1.0
- Python 버전: 3.10+
- 주요 라이브러리: scikit-learn 1.3.0+, matplotlib 3.7.0+, xgboost 2.0.0+
- 예상 학습 시간: 4시간 30분

**라이선스**: MIT License
**기여**: 개선 제안 및 오류 보고 환영합니다!
