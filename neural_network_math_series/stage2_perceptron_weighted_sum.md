# Stage 2: 퍼셉트론과 가중합 수식

## 📚 목차
1. [퍼셉트론의 개념](#1-퍼셉트론의-개념)
2. [가중합 수식 (z = w·x + b)](#2-가중합-수식-z--wx--b)
3. [퍼셉트론의 동작 원리](#3-퍼셉트론의-동작-원리)
4. [실생활 비유](#4-실생활-비유)
5. [Python 시각화](#5-python-시각화)

---

## 1. 퍼셉트론의 개념

### 1.1 퍼셉트론이란?
퍼셉트론(Perceptron)은 **1958년 Frank Rosenblatt**가 고안한 가장 간단한 형태의 인공 신경망입니다. 생물학적 뉴런의 작동 방식을 모방하여 만들어졌습니다.

### 1.2 생물학적 뉴런 vs 인공 퍼셉트론

#### 생물학적 뉴런:
1. **수상돌기(Dendrites)**: 신호 입력
2. **세포체(Cell body)**: 신호 처리
3. **축삭(Axon)**: 신호 출력

#### 인공 퍼셉트론:
1. **입력(Input)**: $x_1, x_2, \ldots, x_n$
2. **가중치(Weights)**: $w_1, w_2, \ldots, w_n$
3. **가중합(Weighted sum)**: $z = \sum w_i x_i + b$
4. **활성화 함수(Activation function)**: $y = f(z)$

### 1.3 퍼셉트론의 구조

```
        입력층        가중합        활성화        출력
         
         x₁ ─────w₁───→ \
         x₂ ─────w₂───→  \
         x₃ ─────w₃───→   Σ + b → z → f(z) → y
          ⋮       ⋮    /
         xₙ ─────wₙ───→/
                        ↑
                        b (편향)
```

---

## 2. 가중합 수식 (z = w·x + b)

### 2.1 수학적 표현

퍼셉트론의 핵심은 **가중합(weighted sum)** 계산입니다.

#### 스칼라 형태:
$$
z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$

#### 시그마 표기법 (Sigma notation):
$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

#### 벡터 내적 형태:
$$
z = \mathbf{w}^T \mathbf{x} + b = \mathbf{w} \cdot \mathbf{x} + b
$$

### 2.2 기호 설명 (Symbol Legend)

| 기호 | 이름 | 의미 | 예시 |
|------|------|------|------|
| $x_i$ | 입력 (input) | $i$번째 특성 값 | 온도, 습도 등 |
| $w_i$ | 가중치 (weight) | $i$번째 입력의 중요도 | 0.5, -0.3 등 |
| $b$ | 편향 (bias) | 기준점 조정 값 | 0.1, -2 등 |
| $n$ | 입력 개수 | 특성의 수 | 2, 10, 784 등 |
| $z$ | 가중합 | 선형 조합 결과 | 실수 값 |
| $\mathbf{w}$ | 가중치 벡터 | 모든 가중치 모음 | $[w_1, w_2, \ldots, w_n]^T$ |
| $\mathbf{x}$ | 입력 벡터 | 모든 입력 모음 | $[x_1, x_2, \ldots, x_n]^T$ |
| $\mathbf{w}^T$ | 전치 벡터 | 행 벡터로 변환 | 내적 계산용 |
| $\sum$ | 합 (summation) | 모든 항을 더함 | $\sum_{i=1}^{3} i = 1+2+3=6$ |

### 2.3 수치 예제 1: 간단한 2차원 입력

**주어진 값:**
- 입력: $\mathbf{x} = [x_1, x_2] = [2, 3]$
- 가중치: $\mathbf{w} = [w_1, w_2] = [0.5, 0.3]$
- 편향: $b = 0.1$

**계산 과정:**

**방법 1 - 스칼라 형태:**
$$
\begin{align}
z &= w_1 x_1 + w_2 x_2 + b \\
  &= (0.5)(2) + (0.3)(3) + 0.1 \\
  &= 1.0 + 0.9 + 0.1 \\
  &= 2.0
\end{align}
$$

**방법 2 - 벡터 내적:**
$$
\begin{align}
z &= \mathbf{w}^T \mathbf{x} + b \\
  &= \begin{bmatrix} 0.5 & 0.3 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix} + 0.1 \\
  &= (0.5 \times 2) + (0.3 \times 3) + 0.1 \\
  &= 2.0
\end{align}
$$

**해석:**
- $z = 2.0$은 양수이므로 "활성화될 가능성이 높음"

### 2.4 수치 예제 2: 3차원 입력

**주어진 값:**
- 입력: $\mathbf{x} = [1.5, 2.0, -0.5]$
- 가중치: $\mathbf{w} = [0.8, -0.4, 0.6]$
- 편향: $b = -0.2$

**계산:**
$$
\begin{align}
z &= \sum_{i=1}^{3} w_i x_i + b \\
  &= (0.8)(1.5) + (-0.4)(2.0) + (0.6)(-0.5) + (-0.2) \\
  &= 1.2 - 0.8 - 0.3 - 0.2 \\
  &= -0.1
\end{align}
$$

**해석:**
- $z = -0.1$은 음수지만 0에 가까움
- "약하게 비활성화"

---

## 3. 퍼셉트론의 동작 원리

### 3.1 전체 수식

퍼셉트론의 출력은 다음과 같이 계산됩니다:

$$
y = f(z) = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

여기서 $f(\cdot)$는 활성화 함수 (Stage 3에서 자세히 다룸)

### 3.2 단계별 동작

**단계 1: 가중합 계산**
$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$

**단계 2: 활성화 함수 적용**
$$
y = f(z)
$$

**예시: 계단 함수 (Step function)**
$$
f(z) = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

### 3.3 구체적 예제: 논리 AND 게이트

논리 AND는 두 입력이 모두 1일 때만 1을 출력합니다.

**진리표:**
| $x_1$ | $x_2$ | $y$ (AND) |
|-------|-------|-----------|
| 0     | 0     | 0         |
| 0     | 1     | 0         |
| 1     | 0     | 0         |
| 1     | 1     | 1         |

**퍼셉트론 설정:**
- $w_1 = 0.5$, $w_2 = 0.5$
- $b = -0.7$
- $f(z) = $ 계단 함수

**검증:**

**Case 1:** $x_1=0, x_2=0$
$$
z = 0.5(0) + 0.5(0) - 0.7 = -0.7 < 0 \Rightarrow y = 0 \quad \checkmark
$$

**Case 2:** $x_1=0, x_2=1$
$$
z = 0.5(0) + 0.5(1) - 0.7 = -0.2 < 0 \Rightarrow y = 0 \quad \checkmark
$$

**Case 3:** $x_1=1, x_2=0$
$$
z = 0.5(1) + 0.5(0) - 0.7 = -0.2 < 0 \Rightarrow y = 0 \quad \checkmark
$$

**Case 4:** $x_1=1, x_2=1$
$$
z = 0.5(1) + 0.5(1) - 0.7 = 0.3 > 0 \Rightarrow y = 1 \quad \checkmark
$$

### 3.4 가중치와 편향의 역할

#### 가중치 ($w_i$)의 역할:
- **크기**: 해당 입력의 중요도
  - $|w_i|$가 크면: 입력 $x_i$가 출력에 큰 영향
  - $|w_i|$가 작으면: 입력 $x_i$가 출력에 작은 영향
- **부호**: 영향의 방향
  - $w_i > 0$: 양의 상관관계 (증가 → 증가)
  - $w_i < 0$: 음의 상관관계 (증가 → 감소)

#### 편향 ($b$)의 역할:
- **임계값 조정**: 활성화되는 기준점 이동
  - $b > 0$: 활성화가 쉬워짐 (왼쪽으로 이동)
  - $b < 0$: 활성화가 어려워짐 (오른쪽으로 이동)
- **기준점 없이 판단 가능**: 입력이 모두 0이어도 출력 가능

**예시:**
$$
z = 2x - 3
$$
- $b = -3$이므로 $x > 1.5$일 때만 $z > 0$

$$
z = 2x + 3
$$
- $b = 3$이므로 $x > -1.5$일 때 $z > 0$ (활성화 쉬움)

---

## 4. 실생활 비유

### 4.1 대학 입학 심사 시스템

대학이 학생을 선발할 때 여러 요소를 고려한다고 가정:

**입력 ($\mathbf{x}$):**
- $x_1$: 수능 점수 (0~100)
- $x_2$: 내신 등급 (1~9, 역으로 9~1로 변환)
- $x_3$: 면접 점수 (0~100)

**가중치 ($\mathbf{w}$):**
- $w_1 = 0.5$ (수능이 가장 중요)
- $w_2 = 0.3$ (내신도 중요)
- $w_3 = 0.2$ (면접은 보조)

**편향 ($b$):**
- $b = -60$ (합격 기준을 높게 설정)

**계산:**
지원자 A: 수능 85점, 내신 2등급(변환 8점), 면접 70점
$$
\begin{align}
z &= 0.5(85) + 0.3(8) + 0.2(70) - 60 \\
  &= 42.5 + 2.4 + 14 - 60 \\
  &= -1.1
\end{align}
$$

**결과:** $z < 0$ → 불합격 (편향 때문에 기준이 높음)

### 4.2 스팸 메일 필터

이메일이 스팸인지 판단하는 시스템:

**입력:**
- $x_1$: "무료"라는 단어 횟수
- $x_2$: "클릭"이라는 단어 횟수
- $x_3$: 이메일 길이 (단어 수)

**가중치:**
- $w_1 = 2.0$ (무료는 스팸 가능성 높임)
- $w_2 = 1.5$ (클릭도 스팸 신호)
- $w_3 = -0.1$ (긴 이메일은 스팸 가능성 낮음)

**편향:**
- $b = -5.0$ (기본적으로 스팸 아님으로 가정)

**계산:**
이메일 X: "무료" 3회, "클릭" 2회, 길이 50단어
$$
\begin{align}
z &= 2.0(3) + 1.5(2) - 0.1(50) - 5.0 \\
  &= 6.0 + 3.0 - 5.0 - 5.0 \\
  &= -1.0
\end{align}
$$

**결과:** $z < 0$ → 정상 메일 (경계선)

### 4.3 신용 승인 시스템

은행이 대출을 승인할지 결정:

**입력:**
- $x_1$: 연소득 (만원 단위)
- $x_2$: 신용 점수 (0~1000)
- $x_3$: 기존 대출 금액 (만원 단위)

**가중치:**
- $w_1 = 0.01$ (소득이 높을수록 좋음)
- $w_2 = 0.05$ (신용 점수가 높을수록 좋음)
- $w_3 = -0.02$ (기존 대출이 많으면 나쁨)

**편향:**
- $b = -30$ (기본 승인 기준)

**계산:**
지원자 B: 연소득 5000만원, 신용점수 750, 기존 대출 1000만원
$$
\begin{align}
z &= 0.01(5000) + 0.05(750) - 0.02(1000) - 30 \\
  &= 50 + 37.5 - 20 - 30 \\
  &= 37.5
\end{align}
$$

**결과:** $z > 0$ → 대출 승인! (큰 양수이므로 확실)

---

## 5. Python 시각화

### 5.1 2차원 입력의 퍼셉트론 결정 경계

아래 코드는 2차원 입력 공간에서 퍼셉트론의 결정 경계를 시각화합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== 1. 퍼셉트론 구조 다이어그램 ==========
ax1 = axes[0, 0]
ax1.axis('off')

# 입력 노드
input_x = 0.2
input_y_positions = [0.7, 0.5, 0.3]
input_labels = ['x₁', 'x₂', 'x₃']

for i, (y, label) in enumerate(zip(input_y_positions, input_labels)):
    circle = plt.Circle((input_x, y), 0.04, color='lightblue', ec='black', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(input_x - 0.12, y, label, fontsize=14, ha='right', fontweight='bold')

# 가중합 노드
sum_x = 0.5
sum_y = 0.5
circle = plt.Circle((sum_x, sum_y), 0.06, color='yellow', ec='black', linewidth=2)
ax1.add_patch(circle)
ax1.text(sum_x, sum_y, 'Σ', fontsize=18, ha='center', va='center', fontweight='bold')

# 활성화 함수 노드
act_x = 0.7
act_y = 0.5
circle = plt.Circle((act_x, act_y), 0.05, color='lightcoral', ec='black', linewidth=2)
ax1.add_patch(circle)
ax1.text(act_x, act_y, 'f', fontsize=14, ha='center', va='center', fontweight='bold')

# 출력 노드
output_x = 0.9
output_y = 0.5
circle = plt.Circle((output_x, output_y), 0.04, color='lightgreen', ec='black', linewidth=2)
ax1.add_patch(circle)
ax1.text(output_x + 0.08, output_y, 'y', fontsize=14, ha='left', fontweight='bold')

# 연결선 (가중치)
weights = ['w₁', 'w₂', 'w₃']
for i, (y, w_label) in enumerate(zip(input_y_positions, weights)):
    ax1.plot([input_x + 0.04, sum_x - 0.06], [y, sum_y], 'gray', linewidth=2)
    mid_x = (input_x + sum_x) / 2
    mid_y = (y + sum_y) / 2
    ax1.text(mid_x, mid_y + 0.02, w_label, fontsize=10, ha='center', color='red', fontweight='bold')

# 편향
ax1.plot([sum_x, sum_x], [sum_y - 0.15, sum_y - 0.06], 'gray', linewidth=2)
ax1.text(sum_x, sum_y - 0.18, 'b', fontsize=12, ha='center', color='red', fontweight='bold')

# 연결선 (활성화)
ax1.arrow(sum_x + 0.06, sum_y, act_x - sum_x - 0.11, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')
ax1.text((sum_x + act_x) / 2, sum_y + 0.05, 'z', fontsize=12, ha='center', color='blue', fontweight='bold')

# 연결선 (출력)
ax1.arrow(act_x + 0.05, act_y, output_x - act_x - 0.09, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('Perceptron Structure', fontsize=16, fontweight='bold', pad=20)

# 수식 표시
ax1.text(0.5, 0.1, r'z = w₁x₁ + w₂x₂ + w₃x₃ + b', fontsize=13, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.text(0.5, 0.02, r'y = f(z)', fontsize=13, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ========== 2. 결정 경계 시각화 (AND 게이트) ==========
ax2 = axes[0, 1]

# AND 게이트 파라미터
w1, w2 = 0.5, 0.5
b = -0.7

# 데이터 포인트
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# 결정 경계: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
x1_line = np.linspace(-0.5, 1.5, 100)
x2_line = -(w1 * x1_line + b) / w2

# 배경 색칠
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
Z = w1 * xx + w2 * yy + b
ax2.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['lightcoral', 'lightblue'], alpha=0.3)

# 결정 경계선
ax2.plot(x1_line, x2_line, 'k-', linewidth=3, label='Decision Boundary')

# 데이터 포인트
for i, (x, y) in enumerate(X_and):
    color = 'blue' if y_and[i] == 1 else 'red'
    marker = 'o' if y_and[i] == 1 else 'x'
    ax2.scatter(x, y, c=color, marker=marker, s=200, edgecolors='black', linewidth=2, 
               label=f'Class {y_and[i]}' if i == 0 or i == 3 else '')

ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel('x₁', fontsize=14)
ax2.set_ylabel('x₂', fontsize=14)
ax2.set_title('AND Gate Decision Boundary', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 1.3, f'w₁={w1}, w₂={w2}, b={b}', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ========== 3. 가중치 변화에 따른 결정 경계 ==========
ax3 = axes[1, 0]

# 여러 가중치 설정
weights_list = [(1, 1, -0.5), (1, 0.5, -0.3), (0.5, 1, -0.3)]
colors = ['red', 'blue', 'green']
labels = ['w=[1,1], b=-0.5', 'w=[1,0.5], b=-0.3', 'w=[0.5,1], b=-0.3']

for (w1, w2, b), color, label in zip(weights_list, colors, labels):
    x1_line = np.linspace(-0.5, 2, 100)
    x2_line = -(w1 * x1_line + b) / w2
    ax3.plot(x1_line, x2_line, linewidth=2, color=color, label=label)

ax3.set_xlim(-0.5, 2)
ax3.set_ylim(-0.5, 2)
ax3.set_xlabel('x₁', fontsize=14)
ax3.set_ylabel('x₂', fontsize=14)
ax3.set_title('Effect of Different Weights on Decision Boundary', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ========== 4. 편향 변화에 따른 결정 경계 ==========
ax4 = axes[1, 1]

# 고정된 가중치, 다양한 편향
w1, w2 = 1, 1
biases = [-1, -0.5, 0, 0.5]
colors = ['red', 'orange', 'blue', 'green']

for b, color in zip(biases, colors):
    x1_line = np.linspace(-0.5, 2, 100)
    x2_line = -(w1 * x1_line + b) / w2
    ax4.plot(x1_line, x2_line, linewidth=2, color=color, label=f'b={b}')

ax4.set_xlim(-0.5, 2)
ax4.set_ylim(-0.5, 2)
ax4.set_xlabel('x₁', fontsize=14)
ax4.set_ylabel('x₂', fontsize=14)
ax4.set_title('Effect of Bias on Decision Boundary', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.text(1, 1.7, 'w₁=1, w₂=1 (fixed)', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 화살표로 편향 증가 방향 표시
ax4.annotate('Bias increasing →', xy=(0.5, 0.3), xytext=(0.2, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'),
            fontsize=11, color='purple', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage2_perceptron_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Perceptron visualization saved!")
plt.close()

# ========== 추가: 3D 가중합 시각화 ==========
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# ========== 1. 3D 가중합 평면 ==========
ax1 = fig.add_subplot(121, projection='3d')

# 파라미터
w1, w2, b = 0.8, 0.6, -1.0

# 그리드 생성
x1 = np.linspace(-2, 2, 30)
x2 = np.linspace(-2, 2, 30)
X1, X2 = np.meshgrid(x1, x2)
Z = w1 * X1 + w2 * X2 + b

# 평면 그리기
surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7, edgecolor='none')

# z=0 평면 (결정 경계)
ax1.plot_surface(X1, X2, np.zeros_like(Z), alpha=0.3, color='red')

ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_zlabel('z = w₁x₁ + w₂x₂ + b', fontsize=12)
ax1.set_title('3D Weighted Sum Surface', fontsize=14, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# ========== 2. 등고선 플롯 ==========
ax2 = fig.add_subplot(122)

# 등고선
contour = ax2.contour(X1, X2, Z, levels=15, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)

# z=0 결정 경계 강조
contour_zero = ax2.contour(X1, X2, Z, levels=[0], colors='red', linewidths=3)
ax2.clabel(contour_zero, inline=True, fontsize=10)

ax2.set_xlabel('x₁', fontsize=12)
ax2.set_ylabel('x₂', fontsize=12)
ax2.set_title('Contour Plot of Weighted Sum', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0, 1.7, f'w₁={w1}, w₂={w2}, b={b}', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage2_weighted_sum_3d.png', dpi=300, bbox_inches='tight')
print("✅ 3D weighted sum visualization saved!")
plt.close()

print("\n🎉 All Stage 2 visualizations completed successfully!")
```

### 5.2 시각화 결과 해설

#### 그림 1: 퍼셉트론 기본 시각화
1. **좌상단 - 퍼셉트론 구조**: 입력, 가중합, 활성화, 출력의 흐름을 다이어그램으로 표현
2. **우상단 - AND 게이트**: 논리 AND 게이트의 결정 경계. 빨간 영역(Class 0)과 파란 영역(Class 1)으로 구분
3. **좌하단 - 가중치 변화**: 서로 다른 가중치가 결정 경계의 기울기를 어떻게 변화시키는지 표시
4. **우하단 - 편향 변화**: 편향 값이 증가하면 결정 경계가 평행 이동하는 것을 보여줌

#### 그림 2: 3D 가중합 시각화
1. **좌측 - 3D 표면**: 가중합 $z = w_1x_1 + w_2x_2 + b$를 3차원 공간에 표현. 빨간 평면(z=0)이 결정 경계
2. **우측 - 등고선**: 같은 가중합 값을 가지는 점들을 선으로 연결. 빨간 선(z=0)이 결정 경계

#### 주요 관찰:
- **결정 경계는 항상 직선** (2D의 경우) 또는 **초평면** (고차원의 경우)
- **가중치**는 경계의 **방향(기울기)** 결정
- **편향**은 경계의 **위치** 결정
- 퍼셉트론은 **선형 분류기**: 직선으로 나눌 수 있는 문제만 해결 가능

---

## 📝 핵심 요약

### 가중합 수식의 3가지 표현

| 표현 방식 | 수식 | 장점 |
|-----------|------|------|
| **스칼라** | $z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b$ | 직관적, 이해하기 쉬움 |
| **시그마** | $z = \sum_{i=1}^{n} w_ix_i + b$ | 간결함, 일반화 |
| **벡터** | $z = \mathbf{w}^T\mathbf{x} + b$ | 효율적 계산, 행렬 연산 |

### 각 요소의 역할

- **입력 ($\mathbf{x}$)**: 판단의 재료 (특성, 특징)
- **가중치 ($\mathbf{w}$)**: 각 입력의 중요도와 영향 방향
- **편향 ($b$)**: 활성화 기준점 조정
- **가중합 ($z$)**: 모든 정보를 종합한 점수

### 실생활 패턴

1. **의사결정 문제**: 여러 요소를 고려해 최종 결정
   - 대학 입학, 대출 승인, 채용 결정 등

2. **분류 문제**: 입력을 범주로 구분
   - 스팸 필터, 질병 진단, 품질 검사 등

3. **선형 조합**: 여러 신호의 가중 평균
   - 포트폴리오 수익률, 종합 점수 계산 등

---

## 🎯 다음 단계 예고

**Stage 3**에서는 가중합 $z$를 비선형 출력으로 변환하는 **활성화 함수**를 배웁니다:
- **Sigmoid**: 확률 출력 (0~1)
- **ReLU**: 음수 제거
- **Tanh**: 대칭적 출력 (-1~1)

활성화 함수가 없으면 신경망은 단순한 선형 모델에 불과합니다. 활성화 함수가 신경망에 강력한 비선형성을 부여하는 방법을 살펴보겠습니다!

---

**작성 완료 시각**: 2024년 기준  
**난이도**: ⭐⭐☆☆☆ (초급-중급)  
**예상 학습 시간**: 50-70분
