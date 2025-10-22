# Stage 5: 미분과 편미분 (Differentiation & Partial Derivatives)

## 📚 목차
1. [미분이란?](#1-미분이란)
2. [기하학적 의미](#2-기하학적-의미)
3. [편미분 (Partial Derivatives)](#3-편미분-partial-derivatives)
4. [그래디언트 (Gradient)](#4-그래디언트-gradient)
5. [연쇄 법칙 (Chain Rule)](#5-연쇄-법칙-chain-rule)
6. [Python 시각화](#6-python-시각화)

---

## 1. 미분이란?

### 1.1 정의
미분(Differentiation)은 함수의 **변화율**을 측정하는 수학적 도구입니다.

### 1.2 수학적 정의

**극한을 이용한 정의:**
$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

**기호 표기:**
$$
\frac{df}{dx} = \frac{d}{dx}f(x) = f'(x) = Df(x)
$$

**기호 설명:**
- $f'(x)$: $f$의 도함수 (derivative)
- $\frac{df}{dx}$: $x$에 대한 $f$의 미분
- $h$: 매우 작은 변화량
- $\lim$: 극한 (limit)

### 1.3 실생활 비유

**속도와 위치:**
- **위치**: $s(t) = 5t^2$ (시간 $t$에 따른 위치)
- **속도**: $v(t) = s'(t) = 10t$ (위치의 변화율)
- $t=3$일 때 속도: $v(3) = 30$ m/s

**경제학:**
- **총비용**: $C(x) = 100 + 5x + 0.1x^2$
- **한계비용**: $C'(x) = 5 + 0.2x$ (추가 생산비용)

### 1.4 기본 미분 공식

| 함수 | 도함수 | 예시 |
|------|--------|------|
| $f(x) = c$ | $f'(x) = 0$ | $f(x) = 5 \Rightarrow f'(x) = 0$ |
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ | $f(x) = x^3 \Rightarrow f'(x) = 3x^2$ |
| $f(x) = e^x$ | $f'(x) = e^x$ | - |
| $f(x) = \log(x)$ | $f'(x) = \frac{1}{x}$ | - |
| $f(x) = \sin(x)$ | $f'(x) = \cos(x)$ | - |

### 1.5 수치 예제

**예제 1:** $f(x) = x^2$

$$
f'(x) = 2x
$$

- $x=3$에서: $f'(3) = 2(3) = 6$
- 의미: $x=3$일 때 함수는 초당 6 단위로 증가

**예제 2:** $f(x) = 3x^2 + 2x - 1$

$$
f'(x) = 6x + 2
$$

- $x=1$에서: $f'(1) = 6(1) + 2 = 8$

---

## 2. 기하학적 의미

### 2.1 접선의 기울기

미분은 **곡선 위의 한 점에서의 접선 기울기**입니다.

**접선 방정식:**
$$
y - f(a) = f'(a)(x - a)
$$

### 2.2 증가/감소 판정

- $f'(x) > 0$: 함수 증가 ↗
- $f'(x) = 0$: 극값 또는 변곡점
- $f'(x) < 0$: 함수 감소 ↘

### 2.3 예시: $f(x) = x^2 - 4x + 3$

$$
f'(x) = 2x - 4
$$

- $x < 2$: $f'(x) < 0$ → 감소
- $x = 2$: $f'(x) = 0$ → 최솟값
- $x > 2$: $f'(x) > 0$ → 증가

최솟값: $f(2) = 4 - 8 + 3 = -1$

---

## 3. 편미분 (Partial Derivatives)

### 3.1 정의
편미분은 **다변수 함수에서 한 변수에 대해서만 미분**하는 것입니다. 다른 변수는 상수로 취급합니다.

### 3.2 수학적 표현

함수 $f(x, y)$의 편미분:

**$x$에 대한 편미분:**
$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}
$$

**$y$에 대한 편미분:**
$$
\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h}
$$

**기호 설명:**
- $\partial$: 편미분 기호 (partial)
- $\frac{\partial f}{\partial x}$: $x$로만 미분, $y$는 상수

### 3.3 수치 예제

**예제 1:** $f(x, y) = x^2 + 3xy + y^2$

$$
\frac{\partial f}{\partial x} = 2x + 3y \quad \text{($y$를 상수로)}
$$

$$
\frac{\partial f}{\partial y} = 3x + 2y \quad \text{($x$를 상수로)}
$$

점 $(2, 3)$에서:
$$
\frac{\partial f}{\partial x}\bigg|_{(2,3)} = 2(2) + 3(3) = 4 + 9 = 13
$$

$$
\frac{\partial f}{\partial y}\bigg|_{(2,3)} = 3(2) + 2(3) = 6 + 6 = 12
$$

**예제 2:** $z = 2x^2y + xy^3$

$$
\frac{\partial z}{\partial x} = 4xy + y^3
$$

$$
\frac{\partial z}{\partial y} = 2x^2 + 3xy^2
$$

### 3.4 실생활 비유

**온도 분포:**
$T(x, y) = $ 위치 $(x, y)$에서의 온도

- $\frac{\partial T}{\partial x}$: 동쪽으로 이동할 때 온도 변화율
- $\frac{\partial T}{\partial y}$: 북쪽으로 이동할 때 온도 변화율

---

## 4. 그래디언트 (Gradient)

### 4.1 정의
그래디언트는 **모든 편미분을 모은 벡터**입니다.

### 4.2 수학적 표현

$$
\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

**기호 설명:**
- $\nabla$ (nabla): 그래디언트 연산자
- $\nabla f$: $f$의 그래디언트 벡터

### 4.3 기하학적 의미

그래디언트는:
1. **가장 가파른 상승 방향**을 가리킴
2. **등고선에 수직**

**경사하강법의 핵심:**
$$
\mathbf{x}_{new} = \mathbf{x}_{old} - \alpha \nabla f(\mathbf{x}_{old})
$$
그래디언트 **반대 방향**으로 이동 → 최솟값 찾기

### 4.4 수치 예제

**예제:** $f(x, y) = x^2 + 2y^2$

$$
\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y}
\end{bmatrix} = \begin{bmatrix}
2x \\
4y
\end{bmatrix}
$$

점 $(3, 1)$에서:
$$
\nabla f(3, 1) = \begin{bmatrix} 6 \\ 4 \end{bmatrix}
$$

의미: 벡터 $[6, 4]$ 방향으로 가장 가파르게 증가

---

## 5. 연쇄 법칙 (Chain Rule)

### 5.1 정의
연쇄 법칙은 **합성 함수의 미분**을 계산하는 규칙입니다.

### 5.2 수학적 표현

**1변수:**
$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

여기서 $y = f(u)$, $u = g(x)$

**다변수:**
$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial x}
$$

### 5.3 신경망에서의 연쇄 법칙

**순방향:**
$$
x \xrightarrow{W_1, b_1} z_1 \xrightarrow{\sigma} a_1 \xrightarrow{W_2, b_2} z_2 \xrightarrow{\sigma} \hat{y} \xrightarrow{L} \mathcal{L}
$$

**역전파 (그래디언트 계산):**
$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}
$$

### 5.4 수치 예제

**예제 1:** $y = (3x + 2)^2$

$u = 3x + 2$, $y = u^2$

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x + 2)
$$

$x=1$에서: $\frac{dy}{dx} = 6(5) = 30$

**예제 2: 신경망**
$$
z = wx + b, \quad a = \sigma(z), \quad L = (a - y)^2
$$

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

$$
= 2(a-y) \cdot \sigma'(z) \cdot x
$$

**구체적 값:**
- $w=0.5, x=2, b=0.1, y=1$
- $z = 0.5(2) + 0.1 = 1.1$
- $a = \sigma(1.1) = 0.750$
- $L = (0.750 - 1)^2 = 0.0625$

$$
\frac{\partial L}{\partial w} = 2(0.750-1) \cdot 0.750(1-0.750) \cdot 2 = -0.1875
$$

---

## 6. Python 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.family'] = 'DejaVu Sans'

# ========== 그림 1: 미분의 기하학적 의미 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 접선
ax1 = axes[0]
x = np.linspace(-2, 4, 100)
f = x**2 - 2*x + 1

# 점 x=1에서 접선
x0 = 1
y0 = x0**2 - 2*x0 + 1
slope = 2*x0 - 2
tangent = slope * (x - x0) + y0

ax1.plot(x, f, 'b-', linewidth=2, label='$f(x) = x^2 - 2x + 1$')
ax1.plot(x, tangent, 'r--', linewidth=2, label=f"Tangent at x={x0}")
ax1.scatter([x0], [y0], color='red', s=100, zorder=5)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('f(x)', fontsize=13)
ax1.set_title('Derivative as Tangent Line Slope', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.text(2.5, 3, f"$f'({x0}) = {slope}$", fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# 2. 증가/감소
ax2 = axes[1]
x2 = np.linspace(-3, 3, 100)
f2 = x2**3 - 3*x2
f2_prime = 3*x2**2 - 3

ax2_twin = ax2.twinx()
ax2.plot(x2, f2, 'b-', linewidth=2.5, label='$f(x) = x^3 - 3x$')
ax2_twin.plot(x2, f2_prime, 'r--', linewidth=2, label="$f'(x) = 3x^2 - 3$")
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2_twin.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 극값 표시
critical_pts = [-1, 1]
for pt in critical_pts:
    ax2.scatter([pt], [pt**3 - 3*pt], color='red', s=100, zorder=5)

ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel('f(x)', fontsize=13, color='b')
ax2_twin.set_ylabel("f'(x)", fontsize=13, color='r')
ax2.tick_params(axis='y', labelcolor='b')
ax2_twin.tick_params(axis='y', labelcolor='r')
ax2.set_title('Critical Points Where Derivative = 0', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2_twin.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('stage5_derivatives_geometric.png', dpi=300, bbox_inches='tight')
print("✅ Geometric interpretation saved!")
plt.close()

# ========== 그림 2: 편미분과 그래디언트 ==========
fig = plt.figure(figsize=(14, 6))

# 1. 3D 표면과 편미분
ax1 = fig.add_subplot(121, projection='3d')
x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)
Z = X**2 + 2*Y**2

surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_zlabel('f(x,y)', fontsize=12)
ax1.set_title('$f(x,y) = x^2 + 2y^2$', fontsize=14, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 2. 그래디언트 벡터장
ax2 = fig.add_subplot(122)
x_grad = np.linspace(-2, 2, 15)
y_grad = np.linspace(-2, 2, 15)
X_grad, Y_grad = np.meshgrid(x_grad, y_grad)

# 그래디언트 계산
U = 2 * X_grad  # ∂f/∂x
V = 4 * Y_grad  # ∂f/∂y

# 등고선
contour = ax2.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.5)
ax2.clabel(contour, inline=True, fontsize=8)

# 그래디언트 벡터
ax2.quiver(X_grad, Y_grad, U, V, alpha=0.6, width=0.003, color='red')

ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel('y', fontsize=13)
ax2.set_title('Gradient Vector Field', fontsize=14, fontweight='bold')
ax2.text(0, -1.7, r'$\nabla f = [2x, 4y]$', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stage5_gradient.png', dpi=300, bbox_inches='tight')
print("✅ Gradient visualization saved!")
plt.close()

# ========== 그림 3: 연쇄 법칙 ==========
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('off')

# 신경망 다이어그램으로 연쇄 법칙 표현
positions = {
    'x': (0.1, 0.5),
    'z': (0.3, 0.5),
    'a': (0.5, 0.5),
    'L': (0.7, 0.5)
}

labels = {
    'x': 'x',
    'z': 'z = wx+b',
    'a': 'a = σ(z)',
    'L': 'L = (a-y)²'
}

# 노드 그리기
for key, (x_pos, y_pos) in positions.items():
    circle = plt.Circle((x_pos, y_pos), 0.06, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x_pos, y_pos, key, fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(x_pos, y_pos - 0.12, labels[key], fontsize=10, ha='center')

# 화살표와 미분 표시
arrows = [
    ('x', 'z', r'$\frac{\partial z}{\partial w} = x$'),
    ('z', 'a', r'$\frac{\partial a}{\partial z} = \sigma\'(z)$'),
    ('a', 'L', r'$\frac{\partial L}{\partial a} = 2(a-y)$')
]

for start, end, label in arrows:
    x1, y1 = positions[start]
    x2, y2 = positions[end]
    ax.annotate('', xy=(x2-0.06, y2), xytext=(x1+0.06, y1),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, y1 + 0.1, label, fontsize=11, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 연쇄 법칙 수식
ax.text(0.4, 0.2, 'Chain Rule:', fontsize=13, ha='center', fontweight='bold')
ax.text(0.4, 0.1, r'$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$',
       fontsize=12, ha='center',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.8)
ax.set_title('Chain Rule in Neural Networks', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('stage5_chain_rule.png', dpi=300, bbox_inches='tight')
print("✅ Chain rule visualization saved!")
plt.close()

print("\n🎉 All Stage 5 visualizations completed successfully!")
```

---

## 📝 핵심 요약

### 미분 개념 요약

| 개념 | 기호 | 의미 | 예시 |
|------|------|------|------|
| **미분** | $\frac{df}{dx}$ | 변화율 | 속도 = 위치의 변화율 |
| **편미분** | $\frac{\partial f}{\partial x}$ | 한 변수만 미분 | 동쪽 방향 온도 변화 |
| **그래디언트** | $\nabla f$ | 모든 편미분 벡터 | 가장 가파른 방향 |
| **연쇄 법칙** | $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$ | 합성 함수 미분 | 역전파 |

### 실생활 비유
- **미분**: 자동차 속도계 (위치 변화율)
- **편미분**: 등산할 때 동쪽/북쪽 방향별 경사
- **그래디언트**: 가장 가파른 오르막 방향
- **연쇄 법칙**: 환율 계산 (원→달러→유로)

---

## 🎯 다음 단계 예고

**Stage 6**에서는 그래디언트를 이용하여 손실을 최소화하는 **경사하강법**을 배웁니다:
- 경사하강법의 직관적 이해
- 학습률의 역할
- 수식 유도와 구현

미분을 이해했으니, 이제 학습 알고리즘의 핵심을 배울 준비가 되었습니다!

---

**작성 완료 시각**: 2024년 기준  
**난이도**: ⭐⭐⭐⭐☆ (중상급)  
**예상 학습 시간**: 70-90분
