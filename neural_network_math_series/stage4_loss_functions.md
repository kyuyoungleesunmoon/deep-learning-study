# Stage 4: 손실 함수 (Loss Functions)

## 📚 목차
1. [손실 함수란?](#1-손실-함수란)
2. [평균 제곱 오차 (MSE)](#2-평균-제곱-오차-mse)
3. [교차 엔트로피 (Cross-Entropy)](#3-교차-엔트로피-cross-entropy)
4. [손실 함수 선택 기준](#4-손실-함수-선택-기준)
5. [Python 시각화](#5-python-시각화)

---

## 1. 손실 함수란?

### 1.1 정의
손실 함수(Loss Function 또는 Cost Function)는 신경망의 **예측값과 실제값의 차이**를 측정하는 함수입니다. 학습의 목표는 이 손실을 최소화하는 것입니다.

### 1.2 수학적 표현

**단일 샘플에 대한 손실:**
$$
L(\hat{y}, y) = \text{distance}(\hat{y}, y)
$$

**전체 데이터셋에 대한 손실 (비용 함수):**
$$
J(\mathbf{W}, \mathbf{b}) = \frac{1}{m}\sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})
$$

**기호 설명:**
- $\hat{y}$: 예측값 (predicted value)
- $y$: 실제값 (true value)
- $L$: 손실 함수
- $J$: 비용 함수 (전체 손실의 평균)
- $m$: 샘플 개수
- $\mathbf{W}, \mathbf{b}$: 신경망의 가중치와 편향
- $(i)$: $i$번째 샘플

### 1.3 왜 필요한가?

**학습 = 최적화:**
$$
\mathbf{W}^*, \mathbf{b}^* = \arg\min_{\mathbf{W}, \mathbf{b}} J(\mathbf{W}, \mathbf{b})
$$

- 손실 함수가 **학습의 목표**(objective)를 정의
- 손실이 작을수록 예측이 정확함
- 경사하강법으로 손실을 최소화

### 1.4 실생활 비유

**과녁 맞추기:**
- **예측값**: 화살이 맞은 위치
- **실제값**: 과녁의 중심
- **손실**: 화살과 중심 사이의 거리
- **학습**: 거리를 줄이는 방법 찾기

**온도 예측:**
- 실제 온도: 25°C
- 예측 온도: 28°C
- 손실: $(28-25)^2 = 9$ (오차의 제곱)

---

## 2. 평균 제곱 오차 (MSE)

### 2.1 수학적 정의

**단일 출력:**
$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2
$$

**다중 출력:**
$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{n}(y_j^{(i)} - \hat{y}_j^{(i)})^2
$$

**기호 설명:**
- $m$: 샘플 수
- $n$: 출력 차원
- $(y - \hat{y})^2$: 오차의 제곱

### 2.2 특성

#### 값의 범위:
$$
\text{MSE} \geq 0
$$

- **최소값**: 0 (완벽한 예측)
- **제곱 사용**: 큰 오차에 더 큰 페널티

#### 미분:
$$
\frac{\partial \text{MSE}}{\partial \hat{y}} = -\frac{2}{m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})
$$

### 2.3 수치 예제

**예제 1: 회귀 문제**

실제값과 예측값:
| 샘플 | 실제값 $y$ | 예측값 $\hat{y}$ | 오차 $(y-\hat{y})$ | 오차² |
|------|-----------|-----------------|-------------------|-------|
| 1    | 10        | 9               | 1                 | 1     |
| 2    | 20        | 22              | -2                | 4     |
| 3    | 30        | 28              | 2                 | 4     |
| 4    | 40        | 41              | -1                | 1     |

**계산:**
$$
\text{MSE} = \frac{1}{4}(1 + 4 + 4 + 1) = \frac{10}{4} = 2.5
$$

**예제 2: 주택 가격 예측**

| 주택 | 실제 가격 (억) | 예측 가격 (억) | 오차² |
|------|--------------|--------------|-------|
| A    | 5.0          | 5.2          | 0.04  |
| B    | 3.5          | 3.0          | 0.25  |
| C    | 7.2          | 7.5          | 0.09  |

$$
\text{MSE} = \frac{1}{3}(0.04 + 0.25 + 0.09) = \frac{0.38}{3} = 0.127 \text{ 억}^2
$$

**RMSE (Root MSE):**
$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{0.127} = 0.356 \text{ 억} = 3,560\text{만원}
$$

### 2.4 MSE의 변형

#### 2.4.1 MAE (Mean Absolute Error)
$$
\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}|y^{(i)} - \hat{y}^{(i)}|
$$

- 이상치(outlier)에 덜 민감
- 절댓값 사용

#### 2.4.2 Huber Loss
$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

- MSE와 MAE의 조합
- 이상치에 강건(robust)

### 2.5 장점 & 단점

#### 장점:
- ✅ **수학적으로 깔끔**: 미분이 간단
- ✅ **볼록 함수**: 전역 최소값 존재
- ✅ **직관적**: "오차의 평균"

#### 단점:
- ❌ **이상치에 민감**: 큰 오차가 손실을 크게 증가
- ❌ **단위 문제**: 제곱하면 단위가 달라짐 (m → m²)
- ❌ **분류에 부적합**: 확률 출력에 맞지 않음

### 2.6 실생활 응용

**회귀 문제에 사용:**
- 주택 가격 예측
- 주식 가격 예측
- 온도 예측
- 매출 예측

---

## 3. 교차 엔트로피 (Cross-Entropy)

### 3.1 수학적 정의

#### 3.1.1 이진 교차 엔트로피 (Binary Cross-Entropy)
$$
L(y, \hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

**전체 샘플:**
$$
J = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]
$$

**기호 설명:**
- $y \in \{0, 1\}$: 실제 레이블
- $\hat{y} \in (0, 1)$: 예측 확률
- $\log$: 자연로그 (밑이 $e$)

#### 3.1.2 범주형 교차 엔트로피 (Categorical Cross-Entropy)
$$
L(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^{K} y_j \log(\hat{y}_j)
$$

**전체 샘플:**
$$
J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{K} y_j^{(i)} \log(\hat{y}_j^{(i)})
$$

**기호 설명:**
- $K$: 클래스 개수
- $\mathbf{y}$: 원-핫 인코딩된 실제 레이블
- $\hat{\mathbf{y}}$: Softmax 출력 (확률 분포)

### 3.2 특성

#### 값의 범위:
$$
0 \leq L \leq \infty
$$

- **최소값**: 0 (완벽한 예측: $\hat{y}=y$)
- **최대값**: $\infty$ (확신있게 틀림: $\hat{y} \to 0$ when $y=1$)

#### 미분 (이진 분류):
$$
\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}
$$

Sigmoid와 함께 사용하면:
$$
\frac{\partial L}{\partial z} = \hat{y} - y
$$

매우 깔끔!

### 3.3 수치 예제

**예제 1: 이진 분류 (스팸 필터)**

| 샘플 | 실제 $y$ | 예측 $\hat{y}$ | $y\log\hat{y}$ | $(1-y)\log(1-\hat{y})$ | $L$ |
|------|---------|---------------|---------------|----------------------|-----|
| 1    | 1       | 0.9           | -0.105        | 0                    | 0.105 |
| 2    | 0       | 0.2           | 0             | -0.223               | 0.223 |
| 3    | 1       | 0.7           | -0.357        | 0                    | 0.357 |
| 4    | 0       | 0.1           | 0             | -0.105               | 0.105 |

**샘플 1 계산:**
$$
L = -(1 \times \log(0.9) + 0 \times \log(0.1)) = -\log(0.9) = 0.105
$$

**평균 손실:**
$$
J = \frac{1}{4}(0.105 + 0.223 + 0.357 + 0.105) = 0.198
$$

**예제 2: 다중 클래스 (동물 분류)**

실제: 고양이 (원-핫: [1, 0, 0])  
예측: [0.7, 0.2, 0.1]

$$
L = -(1 \times \log(0.7) + 0 \times \log(0.2) + 0 \times \log(0.1))
$$
$$
= -\log(0.7) = 0.357
$$

만약 예측이 [0.9, 0.05, 0.05]라면:
$$
L = -\log(0.9) = 0.105 \quad \text{(더 낮음 = 더 좋음)}
$$

### 3.4 정보 이론적 해석

교차 엔트로피는 **두 확률 분포의 차이**를 측정:

$$
H(p, q) = -\sum_x p(x)\log q(x)
$$

- $p$: 실제 분포 (true distribution)
- $q$: 예측 분포 (predicted distribution)

**엔트로피 (Entropy):**
$$
H(p) = -\sum_x p(x)\log p(x)
$$

**KL 발산 (KL Divergence):**
$$
D_{KL}(p \| q) = H(p, q) - H(p)
$$

교차 엔트로피를 최소화 = KL 발산 최소화 = 분포를 가깝게

### 3.5 장점 & 단점

#### 장점:
- ✅ **확률에 적합**: 분류 문제에 자연스러움
- ✅ **빠른 학습**: 오차가 클 때 기울기도 큼
- ✅ **정보 이론적 의미**: 분포 차이 측정

#### 단점:
- ❌ **수치 안정성**: $\log(0)$은 정의 안됨 → 클리핑 필요
- ❌ **불균형 데이터**: 한 클래스가 매우 많으면 편향

### 3.6 실생활 응용

**분류 문제에 사용:**
- 이미지 분류 (개/고양이)
- 스팸 메일 필터
- 감정 분석 (긍정/부정)
- 질병 진단 (양성/음성)
- 객체 인식 (여러 클래스)

---

## 4. 손실 함수 선택 기준

### 4.1 문제 유형별 선택

| 문제 유형 | 출력층 활성화 | 손실 함수 | 예시 |
|----------|-------------|----------|------|
| **이진 분류** | Sigmoid | Binary Cross-Entropy | 스팸/정상 |
| **다중 클래스** | Softmax | Categorical Cross-Entropy | 동물 분류 |
| **회귀** | Linear | MSE | 주택 가격 |
| **회귀 (이상치 많음)** | Linear | MAE 또는 Huber | 매출 예측 |

### 4.2 비교표

| 특징 | MSE | Cross-Entropy |
|------|-----|---------------|
| **용도** | 회귀 | 분류 |
| **출력 범위** | $(-\infty, \infty)$ | $[0, 1]$ (확률) |
| **최적 활성화** | Linear | Sigmoid/Softmax |
| **이상치 민감도** | 높음 (제곱) | 중간 (로그) |
| **수렴 속도** | 느림 | 빠름 |

### 4.3 시각적 비교

**오차 vs 손실:**

| 오차 | MSE | BCE (y=1) |
|------|-----|-----------|
| 0.1  | 0.01 | 0.105 |
| 0.3  | 0.09 | 0.357 |
| 0.5  | 0.25 | 0.693 |
| 0.9  | 0.81 | 2.303 |

**관찰:**
- MSE: 오차가 커지면 제곱으로 증가
- BCE: 오차가 커지면 로그로 급격히 증가 (특히 확신있게 틀릴 때)

---

## 5. Python 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'

# ========== 그림 1: 손실 함수 비교 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. MSE
ax1 = axes[0, 0]
y_true = 10
y_pred = np.linspace(5, 15, 100)
mse = (y_true - y_pred)**2

ax1.plot(y_pred, mse, 'b-', linewidth=2.5)
ax1.axvline(x=y_true, color='red', linestyle='--', linewidth=2, label=f'True value = {y_true}')
ax1.scatter([y_true], [0], color='red', s=100, zorder=5)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Predicted Value', fontsize=13)
ax1.set_ylabel('MSE Loss', fontsize=13)
ax1.set_title('Mean Squared Error', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.text(12, 15, r'$L = (y - \hat{y})^2$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 2. MAE
ax2 = axes[0, 1]
mae = np.abs(y_true - y_pred)

ax2.plot(y_pred, mae, 'g-', linewidth=2.5)
ax2.axvline(x=y_true, color='red', linestyle='--', linewidth=2, label=f'True value = {y_true}')
ax2.scatter([y_true], [0], color='red', s=100, zorder=5)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Predicted Value', fontsize=13)
ax2.set_ylabel('MAE Loss', fontsize=13)
ax2.set_title('Mean Absolute Error', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.text(12, 3, r'$L = |y - \hat{y}|$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# 3. Binary Cross-Entropy (y=1)
ax3 = axes[1, 0]
y_true_bce = 1
y_pred_bce = np.linspace(0.01, 0.99, 100)
bce = -np.log(y_pred_bce)

ax3.plot(y_pred_bce, bce, 'r-', linewidth=2.5)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Predicted Probability', fontsize=13)
ax3.set_ylabel('BCE Loss', fontsize=13)
ax3.set_title('Binary Cross-Entropy (y=1)', fontsize=15, fontweight='bold')
ax3.set_ylim(0, 5)
ax3.text(0.2, 3, r'$L = -\log(\hat{y})$ when $y=1$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5)

# 4. Binary Cross-Entropy (y=0)
ax4 = axes[1, 1]
y_true_bce = 0
bce_neg = -np.log(1 - y_pred_bce)

ax4.plot(y_pred_bce, bce_neg, 'purple', linewidth=2.5)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('Predicted Probability', fontsize=13)
ax4.set_ylabel('BCE Loss', fontsize=13)
ax4.set_title('Binary Cross-Entropy (y=0)', fontsize=15, fontweight='bold')
ax4.set_ylim(0, 5)
ax4.text(0.6, 3, r'$L = -\log(1-\hat{y})$ when $y=0$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage4_loss_functions.png', dpi=300, bbox_inches='tight')
print("✅ Loss functions visualization saved!")
plt.close()

# ========== 그림 2: 실제 데이터 예제 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 회귀 예제
ax1 = axes[0]
np.random.seed(42)
x = np.linspace(0, 10, 50)
y_true_reg = 2 * x + 1 + np.random.randn(50) * 2
y_pred_reg = 2 * x + 1

ax1.scatter(x, y_true_reg, alpha=0.5, label='True values', s=50)
ax1.plot(x, y_pred_reg, 'r-', linewidth=2, label='Predictions')

# 오차 선 표시
for i in [5, 15, 25, 35]:
    ax1.plot([x[i], x[i]], [y_true_reg[i], y_pred_reg[i]], 'g--', alpha=0.5, linewidth=1.5)
    
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('y', fontsize=13)
ax1.set_title('Regression with MSE Loss', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)

mse_value = np.mean((y_true_reg - y_pred_reg)**2)
ax1.text(2, 20, f'MSE = {mse_value:.2f}', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# 2. 분류 예제
ax2 = axes[1]
# 데이터 생성
np.random.seed(42)
class_0 = np.random.randn(50, 2) + np.array([-2, -2])
class_1 = np.random.randn(50, 2) + np.array([2, 2])

ax2.scatter(class_0[:, 0], class_0[:, 1], c='blue', marker='o', s=50, alpha=0.6, label='Class 0')
ax2.scatter(class_1[:, 0], class_1[:, 1], c='red', marker='x', s=50, alpha=0.6, label='Class 1')

# 결정 경계
xx = np.linspace(-5, 5, 100)
yy = xx
ax2.plot(xx, yy, 'k-', linewidth=2, label='Decision boundary')

ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Feature 1', fontsize=13)
ax2.set_ylabel('Feature 2', fontsize=13)
ax2.set_title('Binary Classification with BCE Loss', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage4_examples.png', dpi=300, bbox_inches='tight')
print("✅ Examples visualization saved!")
plt.close()

print("\n🎉 All Stage 4 visualizations completed successfully!")
```

### 5.1 시각화 결과 해설

#### 그림 1: 손실 함수 형태
1. **좌상단 - MSE**: 포물선 형태, 실제값에서 멀어질수록 제곱으로 증가
2. **우상단 - MAE**: V자 형태, 실제값에서 멀어질수록 선형 증가
3. **좌하단 - BCE (y=1)**: 예측 확률이 0에 가까우면 손실이 무한대로
4. **우하단 - BCE (y=0)**: 예측 확률이 1에 가까우면 손실이 무한대로

#### 그림 2: 실제 응용
1. **좌측 - 회귀**: 예측 직선과 실제 점들 사이의 수직 거리가 오차
2. **우측 - 분류**: 결정 경계로 두 클래스를 분리

---

## 📝 핵심 요약

### 손실 함수 한눈에 보기

| 손실 함수 | 수식 | 용도 | 특징 |
|----------|------|------|------|
| **MSE** | $\frac{1}{m}\sum(y-\hat{y})^2$ | 회귀 | 제곱 페널티 |
| **MAE** | $\frac{1}{m}\sum\|y-\hat{y}\|$ | 회귀 | 이상치에 강건 |
| **BCE** | $-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$ | 이진 분류 | 확률 기반 |
| **CCE** | $-\sum y_j\log\hat{y}_j$ | 다중 분류 | 원-핫 인코딩 |

### 선택 가이드

**간단한 규칙:**
1. **회귀 → MSE** (기본)
2. **이진 분류 → Binary Cross-Entropy**
3. **다중 클래스 → Categorical Cross-Entropy**
4. **이상치 많음 → MAE 또는 Huber**

### 실생활 비유
- **MSE**: 과녁 맞추기에서 중심까지의 거리의 제곱
- **MAE**: 과녁 맞추기에서 중심까지의 실제 거리
- **Cross-Entropy**: 확률 예보의 정확도 (날씨 예보)

---

## 🎯 다음 단계 예고

**Stage 5**에서는 손실을 최소화하기 위해 필요한 **미분과 편미분**을 배웁니다:
- 미분의 기하학적 의미
- 편미분과 그래디언트
- 연쇄 법칙

미분을 이해해야 역전파(backpropagation)와 경사하강법을 완전히 이해할 수 있습니다!

---

**작성 완료 시각**: 2024년 기준  
**난이도**: ⭐⭐⭐☆☆ (중급)  
**예상 학습 시간**: 60-75분
