# Stage 3: 활성화 함수 (Activation Functions)

## 📚 목차
1. [활성화 함수란?](#1-활성화-함수란)
2. [시그모이드 (Sigmoid)](#2-시그모이드-sigmoid)
3. [ReLU (Rectified Linear Unit)](#3-relu-rectified-linear-unit)
4. [Tanh (Hyperbolic Tangent)](#4-tanh-hyperbolic-tangent)
5. [기타 활성화 함수](#5-기타-활성화-함수)
6. [활성화 함수의 선택 기준](#6-활성화-함수의-선택-기준)
7. [Python 시각화](#7-python-시각화)

---

## 1. 활성화 함수란?

### 1.1 정의
활성화 함수(Activation Function)는 퍼셉트론의 가중합 $z$를 최종 출력 $y$로 변환하는 비선형 함수입니다.

### 1.2 수학적 표현
$$
y = f(z) = f(\mathbf{w}^T\mathbf{x} + b)
$$

**기호 설명:**
- $z$: 가중합 (weighted sum)
- $f(\cdot)$: 활성화 함수
- $y$: 최종 출력 (activation)

### 1.3 왜 필요한가?

#### ❌ 활성화 함수가 없다면?
신경망이 단순히:
$$
\mathbf{h}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1
$$
$$
\mathbf{y} = \mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2
$$

이를 풀면:
$$
\begin{align}
\mathbf{y} &= \mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 \\
&= \mathbf{W}_2\mathbf{W}_1\mathbf{x} + \mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2 \\
&= \mathbf{W}'\mathbf{x} + \mathbf{b}'
\end{align}
$$

**결과**: 여러 층이 하나의 선형 변환으로 축소됨 → 깊은 신경망의 의미 상실

#### ✅ 활성화 함수가 있다면?
$$
\mathbf{h}_1 = f(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)
$$
$$
\mathbf{y} = f(\mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2)
$$

**결과**: 각 층에서 비선형 변환 → 복잡한 패턴 학습 가능

### 1.4 실생활 비유

**스위치 vs 조광기:**
- **계단 함수** (Step): 전등 스위치 (켜짐/꺼짐만)
- **시그모이드**: 조광기 (0~100% 부드럽게 조절)
- **ReLU**: 역류 방지 밸브 (한 방향만 흐름)

---

## 2. 시그모이드 (Sigmoid)

### 2.1 수학적 정의
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**기호 설명:**
- $\sigma(z)$: 시그모이드 함수 (그리스 문자 시그마)
- $e$: 자연상수 (약 2.71828)
- $e^{-z}$: 자연상수의 $-z$ 제곱

### 2.2 특성

#### 값의 범위:
$$
0 < \sigma(z) < 1
$$

**극한값:**
- $\lim_{z \to +\infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma(0) = 0.5$

#### 대칭성:
$$
\sigma(-z) = 1 - \sigma(z)
$$

### 2.3 수치 예제

| $z$ | $e^{-z}$ | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
|-----|---------|----------------------------------|
| -5  | 148.41  | 0.0067                          |
| -2  | 7.39    | 0.1192                          |
| 0   | 1.00    | 0.5000                          |
| 2   | 0.135   | 0.8808                          |
| 5   | 0.0067  | 0.9933                          |

**계산 예시 ($z=2$):**
$$
\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} = 0.8808
$$

### 2.4 미분 (Derivative)

시그모이드의 미분은 매우 특별한 성질을 가집니다:

$$
\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z))
$$

**기호 설명:**
- $\frac{d\sigma(z)}{dz}$: $z$에 대한 $\sigma(z)$의 미분
- $\cdot$: 곱셈

**미분 유도:**
$$
\begin{align}
\frac{d}{dz}\left(\frac{1}{1+e^{-z}}\right) &= \frac{d}{dz}(1+e^{-z})^{-1} \\
&= -(1+e^{-z})^{-2} \cdot (-e^{-z}) \\
&= \frac{e^{-z}}{(1+e^{-z})^2} \\
&= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} \\
&= \sigma(z) \cdot \frac{e^{-z}}{1+e^{-z}} \\
&= \sigma(z) \cdot (1 - \sigma(z))
\end{align}
$$

**미분 예제:**
$z=2$일 때, $\sigma(2) = 0.8808$이므로:
$$
\sigma'(2) = 0.8808 \times (1 - 0.8808) = 0.8808 \times 0.1192 = 0.105
$$

### 2.5 장점 & 단점

#### 장점:
- ✅ **출력이 확률**: $[0, 1]$ 범위 → 이진 분류에 적합
- ✅ **부드러운 함수**: 미분 가능, 연속
- ✅ **직관적**: S자 형태

#### 단점:
- ❌ **기울기 소실**: $|z|$가 크면 미분값이 0에 가까움
- ❌ **출력이 0 중심 아님**: 항상 양수
- ❌ **계산 비용**: 지수 함수 사용

### 2.6 실생활 응용

**스팸 메일 확률:**
가중합 $z = 3.5$인 메일:
$$
P(\text{스팸}) = \sigma(3.5) = \frac{1}{1 + e^{-3.5}} = 0.9707
$$
→ 97% 확률로 스팸

**합격 확률:**
입학 점수 $z = -1.2$:
$$
P(\text{합격}) = \sigma(-1.2) = \frac{1}{1 + e^{1.2}} = 0.2315
$$
→ 23% 확률로 합격

---

## 3. ReLU (Rectified Linear Unit)

### 3.1 수학적 정의
$$
\text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**기호 설명:**
- $\max(a, b)$: $a$와 $b$ 중 큰 값

### 3.2 특성

#### 값의 범위:
$$
\text{ReLU}(z) \geq 0
$$

**예시:**
- $\text{ReLU}(3.5) = 3.5$
- $\text{ReLU}(-2.1) = 0$
- $\text{ReLU}(0) = 0$

### 3.3 수치 예제

| $z$ | $\max(0, z)$ |
|-----|--------------|
| -3  | 0            |
| -1  | 0            |
| 0   | 0            |
| 1   | 1            |
| 3   | 3            |

### 3.4 미분 (Derivative)

$$
\frac{d\text{ReLU}(z)}{dz} = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z < 0 \\
\text{undefined} & \text{if } z = 0
\end{cases}
$$

**실무에서는:** $z=0$일 때 미분을 0 또는 1로 정의 (보통 0)

**미분 예제:**
- $z=5$: $\text{ReLU}'(5) = 1$
- $z=-2$: $\text{ReLU}'(-2) = 0$

### 3.5 장점 & 단점

#### 장점:
- ✅ **계산 효율**: 단순 비교와 선택만
- ✅ **기울기 소실 완화**: $z > 0$일 때 미분값이 1
- ✅ **희소성**: 음수 입력을 0으로 만들어 네트워크 단순화

#### 단점:
- ❌ **Dying ReLU**: $z < 0$일 때 뉴런이 완전히 "죽음" (항상 0 출력)
- ❌ **출력 범위 무한**: 제한 없음

### 3.6 ReLU 변형

#### 3.6.1 Leaky ReLU
음수 영역에 작은 기울기 부여:
$$
\text{LeakyReLU}(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0
\end{cases}
$$

보통 $\alpha = 0.01$

**예시:**
- $\text{LeakyReLU}(2) = 2$
- $\text{LeakyReLU}(-3) = -0.03$

#### 3.6.2 ELU (Exponential Linear Unit)
$$
\text{ELU}(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha(e^z - 1) & \text{if } z \leq 0
\end{cases}
$$

### 3.7 실생활 비유

**역류 방지 밸브:**
물(신호)이 한 방향으로만 흐르도록 함. 역방향은 차단(0).

**다이오드:**
전류가 양의 방향으로만 흐름. 음의 전압은 차단.

---

## 4. Tanh (Hyperbolic Tangent)

### 4.1 수학적 정의
$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}
$$

**시그모이드와의 관계:**
$$
\tanh(z) = 2\sigma(2z) - 1
$$

### 4.2 특성

#### 값의 범위:
$$
-1 < \tanh(z) < 1
$$

**극한값:**
- $\lim_{z \to +\infty} \tanh(z) = 1$
- $\lim_{z \to -\infty} \tanh(z) = -1$
- $\tanh(0) = 0$

#### 대칭성:
$$
\tanh(-z) = -\tanh(z)
$$

### 4.3 수치 예제

| $z$ | $e^z$ | $e^{-z}$ | $\tanh(z)$ |
|-----|-------|---------|-----------|
| -2  | 0.135 | 7.389   | -0.964    |
| -1  | 0.368 | 2.718   | -0.762    |
| 0   | 1.000 | 1.000   | 0.000     |
| 1   | 2.718 | 0.368   | 0.762     |
| 2   | 7.389 | 0.135   | 0.964     |

**계산 예시 ($z=1$):**
$$
\tanh(1) = \frac{e^1 - e^{-1}}{e^1 + e^{-1}} = \frac{2.718 - 0.368}{2.718 + 0.368} = \frac{2.350}{3.086} = 0.762
$$

### 4.4 미분 (Derivative)
$$
\frac{d\tanh(z)}{dz} = 1 - \tanh^2(z)
$$

**미분 예제:**
$z=1$일 때, $\tanh(1) = 0.762$이므로:
$$
\tanh'(1) = 1 - (0.762)^2 = 1 - 0.581 = 0.419
$$

### 4.5 장점 & 단점

#### 장점:
- ✅ **0 중심 출력**: $[-1, 1]$ 범위
- ✅ **시그모이드보다 강한 기울기**: 학습이 더 빠름

#### 단점:
- ❌ **기울기 소실**: 시그모이드와 유사한 문제
- ❌ **계산 비용**: 지수 함수 사용

### 4.6 실생활 응용

**온도 편차:**
평균 기온 대비 편차를 $[-1, 1]$로 정규화
- $\tanh(2) = 0.964$: 매우 더움
- $\tanh(0) = 0$: 평균
- $\tanh(-2) = -0.964$: 매우 추움

---

## 5. 기타 활성화 함수

### 5.1 Softmax

**다중 클래스 분류**에 사용:
$$
\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**특성:**
- 모든 출력의 합이 1: $\sum_{i=1}^{K} \text{Softmax}(\mathbf{z})_i = 1$
- 각 출력은 확률로 해석 가능

**예시:**
$$
\mathbf{z} = [2.0, 1.0, 0.1]
$$

계산:
$$
\sum e^{z_j} = e^{2.0} + e^{1.0} + e^{0.1} = 7.389 + 2.718 + 1.105 = 11.212
$$

$$
\text{Softmax}(\mathbf{z}) = \left[\frac{7.389}{11.212}, \frac{2.718}{11.212}, \frac{1.105}{11.212}\right] = [0.659, 0.242, 0.099]
$$

**해석**: 클래스 1일 확률 66%, 클래스 2일 확률 24%, 클래스 3일 확률 10%

### 5.2 Swish

Google이 제안한 활성화 함수:
$$
\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

**특성:**
- ReLU와 유사하지만 부드러움
- 음수 영역에서도 0이 아닌 값

---

## 6. 활성화 함수의 선택 기준

### 6.1 비교표

| 함수 | 범위 | 미분 | 계산 비용 | 주요 용도 |
|------|------|------|-----------|----------|
| **Sigmoid** | $(0, 1)$ | $\sigma(1-\sigma)$ | 높음 | 이진 분류 출력층 |
| **Tanh** | $(-1, 1)$ | $1-\tanh^2$ | 높음 | RNN 은닉층 |
| **ReLU** | $[0, \infty)$ | $\{0, 1\}$ | 낮음 | CNN/DNN 은닉층 |
| **Leaky ReLU** | $(-\infty, \infty)$ | $\{\alpha, 1\}$ | 낮음 | Dying ReLU 방지 |
| **Softmax** | $(0, 1)$, 합=1 | 복잡 | 중간 | 다중 클래스 출력층 |

### 6.2 선택 가이드

#### 은닉층 (Hidden Layer):
1. **기본 선택**: ReLU
   - 계산 빠름, 학습 효율적
2. **ReLU 문제 발생 시**: Leaky ReLU 또는 ELU
3. **RNN/LSTM**: Tanh

#### 출력층 (Output Layer):
1. **이진 분류**: Sigmoid
2. **다중 클래스 분류**: Softmax
3. **회귀**: 없음 (선형 출력) 또는 ReLU (양수만)

### 6.3 실무 팁

**문제별 추천:**
- **이미지 분류 (CNN)**: 은닉층 ReLU, 출력층 Softmax
- **자연어 처리 (RNN)**: 은닉층 Tanh/Sigmoid, 출력층 Softmax
- **이진 감정 분석**: 은닉층 ReLU, 출력층 Sigmoid
- **회귀 (주택 가격)**: 은닉층 ReLU, 출력층 Linear

---

## 7. Python 시각화

아래 코드는 모든 주요 활성화 함수를 시각화하고 비교합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'

# z 값 범위
z = np.linspace(-5, 5, 400)

# ========== 활성화 함수 정의 ==========
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

# ========== 그림 1: 주요 활성화 함수 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Sigmoid
ax1 = axes[0, 0]
ax1.plot(z, sigmoid(z), 'b-', linewidth=2.5, label='Sigmoid')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('z', fontsize=13)
ax1.set_ylabel('σ(z)', fontsize=13)
ax1.set_title('Sigmoid Activation Function', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.text(2, 0.2, r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.text(2, 0.05, 'Range: (0, 1)', fontsize=11, color='red')

# ReLU
ax2 = axes[0, 1]
ax2.plot(z, relu(z), 'r-', linewidth=2.5, label='ReLU')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('z', fontsize=13)
ax2.set_ylabel('ReLU(z)', fontsize=13)
ax2.set_title('ReLU Activation Function', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.text(2, 1, r'$ReLU(z) = \max(0, z)$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
ax2.text(2, 0.3, r'Range: $[0, \infty)$', fontsize=11, color='red')

# Tanh
ax3 = axes[1, 0]
ax3.plot(z, tanh(z), 'g-', linewidth=2.5, label='Tanh')
ax3.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('z', fontsize=13)
ax3.set_ylabel('tanh(z)', fontsize=13)
ax3.set_title('Tanh Activation Function', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.text(2, -0.5, r'$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax3.text(2, -0.8, 'Range: (-1, 1)', fontsize=11, color='red')

# Comparison
ax4 = axes[1, 1]
ax4.plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid', alpha=0.8)
ax4.plot(z, relu(z), 'r-', linewidth=2, label='ReLU', alpha=0.8)
ax4.plot(z, tanh(z), 'g-', linewidth=2, label='Tanh', alpha=0.8)
ax4.plot(z, leaky_relu(z), 'm--', linewidth=2, label='Leaky ReLU', alpha=0.8)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('z', fontsize=13)
ax4.set_ylabel('f(z)', fontsize=13)
ax4.set_title('Comparison of Activation Functions', fontsize=15, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylim(-1.5, 5)

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage3_activation_functions.png', dpi=300, bbox_inches='tight')
print("✅ Activation functions visualization saved!")
plt.close()

# ========== 그림 2: 미분 시각화 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Sigmoid derivative
ax1 = axes[0, 0]
ax1.plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid', alpha=0.5)
ax1_twin = ax1.twinx()
ax1_twin.plot(z, sigmoid_derivative(z), 'r--', linewidth=2.5, label="Sigmoid'")
ax1.set_xlabel('z', fontsize=13)
ax1.set_ylabel('σ(z)', fontsize=13, color='b')
ax1_twin.set_ylabel("σ'(z)", fontsize=13, color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1.grid(True, alpha=0.3)
ax1.set_title("Sigmoid and Its Derivative", fontsize=15, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1_twin.legend(loc='upper right', fontsize=10)
ax1.text(0, 0.7, r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$", fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# ReLU derivative
ax2 = axes[0, 1]
ax2.plot(z, relu(z), 'r-', linewidth=2, label='ReLU', alpha=0.5)
ax2_twin = ax2.twinx()
ax2_twin.plot(z, relu_derivative(z), 'b--', linewidth=2.5, label="ReLU'")
ax2.set_xlabel('z', fontsize=13)
ax2.set_ylabel('ReLU(z)', fontsize=13, color='r')
ax2_twin.set_ylabel("ReLU'(z)", fontsize=13, color='b')
ax2.tick_params(axis='y', labelcolor='r')
ax2_twin.tick_params(axis='y', labelcolor='b')
ax2.grid(True, alpha=0.3)
ax2.set_title("ReLU and Its Derivative", fontsize=15, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2_twin.legend(loc='lower right', fontsize=10)
ax2_twin.set_ylim(-0.2, 1.5)

# Tanh derivative
ax3 = axes[1, 0]
ax3.plot(z, tanh(z), 'g-', linewidth=2, label='Tanh', alpha=0.5)
ax3_twin = ax3.twinx()
ax3_twin.plot(z, tanh_derivative(z), 'purple', linestyle='--', linewidth=2.5, label="Tanh'")
ax3.set_xlabel('z', fontsize=13)
ax3.set_ylabel('tanh(z)', fontsize=13, color='g')
ax3_twin.set_ylabel("tanh'(z)", fontsize=13, color='purple')
ax3.tick_params(axis='y', labelcolor='g')
ax3_twin.tick_params(axis='y', labelcolor='purple')
ax3.grid(True, alpha=0.3)
ax3.set_title("Tanh and Its Derivative", fontsize=15, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3_twin.legend(loc='upper right', fontsize=10)
ax3.text(0, 0.2, r"$\tanh'(z) = 1 - \tanh^2(z)$", fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Derivative comparison
ax4 = axes[1, 1]
ax4.plot(z, sigmoid_derivative(z), 'b-', linewidth=2, label="Sigmoid'", alpha=0.8)
ax4.plot(z, relu_derivative(z), 'r-', linewidth=2, label="ReLU'", alpha=0.8)
ax4.plot(z, tanh_derivative(z), 'g-', linewidth=2, label="Tanh'", alpha=0.8)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('z', fontsize=13)
ax4.set_ylabel("f'(z)", fontsize=13)
ax4.set_title('Comparison of Derivatives', fontsize=15, fontweight='bold')
ax4.legend(fontsize=11)
ax4.text(2, 0.5, 'ReLU: constant gradient\nwhen z > 0', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage3_derivatives.png', dpi=300, bbox_inches='tight')
print("✅ Derivatives visualization saved!")
plt.close()

print("\n🎉 All Stage 3 visualizations completed successfully!")
```

### 7.1 시각화 결과 해설

#### 그림 1: 활성화 함수
1. **좌상단 - Sigmoid**: S자 곡선, 0~1 범위
2. **우상단 - ReLU**: 음수는 0, 양수는 그대로
3. **좌하단 - Tanh**: S자 곡선, -1~1 범위, 0 중심
4. **우하단 - 비교**: 모든 함수를 한 그래프에 표시

#### 그림 2: 미분
1. **좌상단 - Sigmoid 미분**: 최대값이 0.25 (z=0에서), 양 끝에서 0에 수렴 → 기울기 소실 문제
2. **우상단 - ReLU 미분**: z>0에서 1, z<0에서 0 → 간단하고 명확
3. **좌하단 - Tanh 미분**: 최대값이 1 (z=0에서), 시그모이드보다 큼
4. **우하단 - 미분 비교**: ReLU가 가장 단순하고 일정한 기울기 유지

---

## 📝 핵심 요약

### 활성화 함수 한눈에 보기

| 특징 | Sigmoid | ReLU | Tanh |
|------|---------|------|------|
| **수식** | $\frac{1}{1+e^{-z}}$ | $\max(0,z)$ | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ |
| **범위** | (0, 1) | [0, ∞) | (-1, 1) |
| **미분** | $\sigma(1-\sigma)$ | {0, 1} | $1-\tanh^2$ |
| **중심** | 0.5 | 0 | 0 |
| **장점** | 확률 해석 | 빠름, 간단 | 0 중심 |
| **단점** | 기울기 소실 | Dying ReLU | 기울기 소실 |

### 선택 기준 요약

**간단한 규칙:**
1. **시작은 ReLU**: 대부분의 은닉층
2. **이진 분류 출력**: Sigmoid
3. **다중 클래스 출력**: Softmax
4. **RNN**: Tanh

### 실생활 비유 요약
- **Sigmoid**: 조광기 (부드러운 밝기 조절)
- **ReLU**: 역류 방지 밸브 (한 방향만)
- **Tanh**: 온도계 (음수/양수 모두)
- **Softmax**: 투표 시스템 (확률 합=100%)

---

## 🎯 다음 단계 예고

**Stage 4**에서는 신경망의 학습 목표를 정의하는 **손실 함수**를 배웁니다:
- **MSE**: 회귀 문제용
- **Cross-Entropy**: 분류 문제용

손실 함수는 "신경망이 얼마나 틀렸는지"를 수치화하여, 학습의 방향을 제시합니다!

---

**작성 완료 시각**: 2024년 기준  
**난이도**: ⭐⭐⭐☆☆ (중급)  
**예상 학습 시간**: 60-80분
