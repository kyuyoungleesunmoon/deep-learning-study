# Stage 1: 인공 신경망 학습에서 사용하는 기본 수학 구조 개요

## 📚 목차
1. [스칼라 (Scalar)](#1-스칼라-scalar)
2. [벡터 (Vector)](#2-벡터-vector)
3. [행렬 (Matrix)](#3-행렬-matrix)
4. [선형 변환 (Linear Transformation)](#4-선형-변환-linear-transformation)
5. [Python 시각화](#5-python-시각화)

---

## 1. 스칼라 (Scalar)

### 1.1 정의
스칼라는 하나의 수치로 표현되는 값입니다. 크기(magnitude)만 가지고 방향은 없습니다.

### 1.2 수학적 표현
$$
s \in \mathbb{R}
$$

**기호 설명:**
- $s$: 스칼라 값
- $\mathbb{R}$: 실수 집합 (Real numbers)
- $\in$: "속한다" (belongs to)

### 1.3 실생활 예시
- **온도**: 25°C (단순히 크기만 나타냄)
- **몸무게**: 70kg
- **나이**: 30세
- **신경망에서**: 학습률(learning rate), 편향(bias) 값

### 1.4 수치 예제
```
s₁ = 3.14
s₂ = -2.5
s₃ = 0.001
```

---

## 2. 벡터 (Vector)

### 2.1 정의
벡터는 여러 개의 수치를 순서대로 나열한 것입니다. 크기와 방향을 모두 가집니다.

### 2.2 수학적 표현

**열 벡터 (Column Vector):**

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
$$

**행 벡터 (Row Vector):**

$$
\mathbf{v}^T = \begin{bmatrix} v_1 & v_2 & \cdots & v_n \end{bmatrix}
$$

**기호 설명:**

- $\mathbf{v}$: 벡터 (굵은 글씨로 표기)
- $v_i$: 벡터의 $i$번째 원소 (element)
- $n$: 벡터의 차원 (dimension)
- $\mathbb{R}^n$: $n$차원 실수 공간
- $\mathbf{v}^T$: 벡터의 전치 (transpose)

### 2.3 실생활 예시
- **위치**: 집의 좌표 (위도, 경도) = [37.5, 127.0]
- **속도**: 자동차의 속도 (동쪽 50km/h, 북쪽 30km/h) = [50, 30]
- **RGB 색상**: 빨강, 초록, 파랑 = [255, 0, 128]
- **신경망에서**: 입력 데이터, 특성(features), 가중치(weights)

### 2.4 수치 예제

**2차원 벡터:**

$$
\mathbf{x} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

**벡터의 크기 (Norm):**

$$
\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2} = \sqrt{3^2 + 4^2} = \sqrt{25} = 5
$$

**기호 설명:**

- $\|\mathbf{x}\|$: 벡터의 크기 (또는 노름, norm)
- $\sqrt{\cdot}$: 제곱근

### 2.5 벡터 연산

#### 2.5.1 벡터 덧셈

$$
\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}
$$

**예제:**

$$
\begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 1 \\ 4 \end{bmatrix} = \begin{bmatrix} 3 \\ 7 \end{bmatrix}
$$

#### 2.5.2 스칼라 곱 (Scalar Multiplication)

$$
c \cdot \mathbf{v} = c \cdot \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \end{bmatrix}
$$

**예제:**

$$
2 \cdot \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 8 \end{bmatrix}
$$

#### 2.5.3 내적 (Dot Product / Inner Product)

$$
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n = \sum_{i=1}^{n} a_i b_i
$$

**기호 설명:**

- $\cdot$: 내적 연산자
- $\sum$: 합 (summation)
- $i=1$부터 $n$까지: 인덱스의 범위

**예제:**

$$
\begin{bmatrix} 2 \\ 3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\ 5 \end{bmatrix} = (2 \times 4) + (3 \times 5) = 8 + 15 = 23
$$

**실생활 의미**: 두 벡터가 얼마나 같은 방향을 가리키는지 측정
- 내적 > 0: 같은 방향
- 내적 = 0: 수직 (직교)
- 내적 < 0: 반대 방향

---

## 3. 행렬 (Matrix)

### 3.1 정의
행렬은 숫자들을 2차원 배열로 배치한 것입니다. 여러 벡터를 모아놓은 것으로 볼 수 있습니다.

### 3.2 수학적 표현

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

**기호 설명:**

- $\mathbf{A}$: 행렬 (대문자 굵은 글씨)
- $a_{ij}$: $i$번째 행(row), $j$번째 열(column)의 원소
- $m$: 행의 개수
- $n$: 열의 개수
- $\mathbb{R}^{m \times n}$: $m \times n$ 크기의 실수 행렬 공간

### 3.3 실생활 예시

- **이미지**: 픽셀을 행렬로 표현 (예: 28x28 이미지 = 28x28 행렬)
- **학생 성적표**: 학생(행) × 과목(열)
- **거리 행렬**: 도시 간 거리
- **신경망에서**: 가중치 행렬 (여러 입력을 여러 출력으로 변환)

### 3.4 수치 예제

**2×3 행렬:**

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

- 2개의 행, 3개의 열
- $a_{11} = 1$, $a_{12} = 2$, $a_{23} = 6$

### 3.5 행렬 연산

#### 3.5.1 행렬 덧셈

$$
\mathbf{A} + \mathbf{B} = \begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} \\
a_{21} + b_{21} & a_{22} + b_{22}
\end{bmatrix}
$$

**조건**: 두 행렬의 크기가 같아야 함

**예제:**

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}
$$

#### 3.5.2 행렬-벡터 곱 (Matrix-Vector Multiplication)

$$
\mathbf{A}\mathbf{x} = \begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix}
a_{11}x_1 + a_{12}x_2 \\
a_{21}x_1 + a_{22}x_2
\end{bmatrix}
$$

**예제:**

$$
\begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 2 \cdot 1 + 3 \cdot 2 \\ 4 \cdot 1 + 5 \cdot 2 \end{bmatrix} = \begin{bmatrix} 8 \\ 14 \end{bmatrix}
$$

**해석**: 각 행과 벡터의 내적 결과를 모은 것

#### 3.5.3 행렬 곱셈 (Matrix Multiplication)

$$
\mathbf{C} = \mathbf{A}\mathbf{B}
$$

여기서 $c_{ij}$는 $\mathbf{A}$의 $i$번째 행과 $\mathbf{B}$의 $j$번째 열의 내적:
$$
c_{ij} = \sum_{k=1}^{p} a_{ik} b_{kj}
$$

**조건**: $\mathbf{A} \in \mathbb{R}^{m \times p}$, $\mathbf{B} \in \mathbb{R}^{p \times n}$ → $\mathbf{C} \in \mathbb{R}^{m \times n}$

**예제:**

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

---

## 4. 선형 변환 (Linear Transformation)

### 4.1 정의
선형 변환은 벡터 공간의 벡터를 다른 벡터 공간의 벡터로 매핑하는 함수입니다.

### 4.2 수학적 표현

$$
\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}
$$

**기호 설명:**

- $\mathbf{x} \in \mathbb{R}^n$: 입력 벡터
- $\mathbf{A} \in \mathbb{R}^{m \times n}$: 변환 행렬 (가중치)
- $\mathbf{b} \in \mathbb{R}^m$: 편향 벡터 (bias)
- $\mathbf{y} \in \mathbb{R}^m$: 출력 벡터

### 4.3 선형 변환의 성질

함수 $f(\mathbf{x})$가 선형 변환이려면 다음 두 조건을 만족해야 합니다:

**1. 가법성 (Additivity):**

$$
f(\mathbf{x} + \mathbf{y}) = f(\mathbf{x}) + f(\mathbf{y})
$$

**2. 동차성 (Homogeneity):**

$$
f(c\mathbf{x}) = c \cdot f(\mathbf{x})
$$

여기서 $c$는 스칼라

### 4.4 실생활 예시

- **화면 회전**: 이미지를 90도 회전시키기
- **크기 조정**: 이미지를 2배로 확대하기
- **좌표 변환**: GPS 좌표를 화면 좌표로 변환
- **신경망에서**: 입력층에서 은닉층으로 데이터 변환 ($\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}$)

### 4.5 수치 예제

2차원 공간에서 회전 변환 (45도 반시계방향):

**회전 행렬:**

$$
\mathbf{R} = \begin{bmatrix}
\cos(45°) & -\sin(45°) \\
\sin(45°) & \cos(45°)
\end{bmatrix} \approx \begin{bmatrix}
0.707 & -0.707 \\
0.707 & 0.707
\end{bmatrix}
$$

**입력 벡터:**

$$
\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

**변환된 벡터:**

$$
\mathbf{y} = \mathbf{R}\mathbf{x} = \begin{bmatrix}
0.707 & -0.707 \\
0.707 & 0.707
\end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix}
$$

**해석**: 점 (1, 0)이 45도 회전하여 (0.707, 0.707)로 이동

### 4.6 신경망에서의 선형 변환

신경망의 각 층은 선형 변환으로 볼 수 있습니다:

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

**예제: 3개 입력 → 2개 출력**

$$
\begin{bmatrix} z_1 \\ z_2 \end{bmatrix} = \begin{bmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23}
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
$$

**구체적 값:**

$$
\begin{bmatrix} z_1 \\ z_2 \end{bmatrix} = \begin{bmatrix}
0.5 & 0.3 & 0.2 \\
0.7 & 0.4 & 0.6
\end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}
$$

**계산:**

$$
z_1 = (0.5 \times 1.0) + (0.3 \times 2.0) + (0.2 \times 3.0) + 0.1 = 0.5 + 0.6 + 0.6 + 0.1 = 1.8
$$
$$
z_2 = (0.7 \times 1.0) + (0.4 \times 2.0) + (0.6 \times 3.0) + 0.2 = 0.7 + 0.8 + 1.8 + 0.2 = 3.5
$$

$$
\mathbf{z} = \begin{bmatrix} 1.8 \\ 3.5 \end{bmatrix}
$$

---

## 5. Python 시각화

다음은 위에서 배운 개념들을 Python으로 시각화하는 코드입니다.

### 5.1 벡터 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'

# 그림 크기 설정
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== 1. 벡터 표현 ==========
ax1 = axes[0, 0]
# 원점
origin = [0, 0]
# 벡터 정의
v1 = [3, 4]
v2 = [2, -1]

# 벡터 그리기
ax1.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='blue', width=0.006, label='v1 = [3, 4]')
ax1.quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, label='v2 = [2, -1]')

# 격자 및 축 설정
ax1.set_xlim(-1, 5)
ax1.set_ylim(-2, 5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Vectors in 2D Space', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)

# 벡터 크기 표시
magnitude_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
ax1.text(1.5, 2.5, f'||v1|| = {magnitude_v1:.2f}', fontsize=10, color='blue')

# ========== 2. 벡터 덧셈 ==========
ax2 = axes[0, 1]
a = [2, 3]
b = [1, 4]
c = [a[0] + b[0], a[1] + b[1]]  # 벡터 합

ax2.quiver(*origin, *a, angles='xy', scale_units='xy', scale=1, color='blue', width=0.006, label='a = [2, 3]')
ax2.quiver(*a, *b, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, label='b = [1, 4]')
ax2.quiver(*origin, *c, angles='xy', scale_units='xy', scale=1, color='green', width=0.008, label='a + b = [3, 7]', linestyle='--')

ax2.set_xlim(-1, 5)
ax2.set_ylim(-1, 8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Vector Addition', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)

# ========== 3. 내적 시각화 ==========
ax3 = axes[1, 0]
v_a = np.array([4, 2])
v_b = np.array([3, 3])
dot_product = np.dot(v_a, v_b)

ax3.quiver(*origin, *v_a, angles='xy', scale_units='xy', scale=1, color='blue', width=0.006, label=f'a = {v_a}')
ax3.quiver(*origin, *v_b, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, label=f'b = {v_b}')

# 각도 계산
cos_angle = dot_product / (np.linalg.norm(v_a) * np.linalg.norm(v_b))
angle = np.arccos(cos_angle) * 180 / np.pi

ax3.set_xlim(-1, 5)
ax3.set_ylim(-1, 5)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Dot Product', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.text(2, 4, f'a · b = {dot_product}', fontsize=12, color='purple', fontweight='bold')
ax3.text(2, 3.5, f'angle = {angle:.1f}°', fontsize=10, color='purple')

# ========== 4. 선형 변환 (회전) ==========
ax4 = axes[1, 1]
# 45도 회전 행렬
theta = np.pi / 4  # 45도
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# 원본 벡터들
original_vectors = np.array([[1, 0], [0, 1], [1, 1]]).T
# 변환된 벡터들
transformed_vectors = R @ original_vectors

# 원본 벡터 그리기
for i in range(original_vectors.shape[1]):
    ax4.quiver(*origin, *original_vectors[:, i], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.005, alpha=0.5, linestyle='--')

# 변환된 벡터 그리기
for i in range(transformed_vectors.shape[1]):
    ax4.quiver(*origin, *transformed_vectors[:, i], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.006)

ax4.set_xlim(-0.5, 1.5)
ax4.set_ylim(-0.5, 1.5)
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('Linear Transformation (45° Rotation)', fontsize=14, fontweight='bold')
ax4.text(0.1, 1.3, 'Blue: Original', fontsize=10, color='blue')
ax4.text(0.1, 1.2, 'Red: Rotated', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage1_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved to stage1_visualization.png")
plt.show()
```

### 5.2 행렬 연산 시각화 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ========== 1. 행렬-벡터 곱 시각화 ==========
ax1 = axes[0]
A = np.array([[2, 3], [4, 5]])
x = np.array([1, 2])
y = A @ x

# 입력 벡터
ax1.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label=f'x = {x}')
# 출력 벡터
ax1.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.008, label=f'Ax = {y}')

ax1.set_xlim(-1, 10)
ax1.set_ylim(-1, 15)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Matrix-Vector Multiplication', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.text(4, 12, f'A = {A.tolist()}', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========== 2. 행렬 곱셈 결과 히트맵 ==========
ax2 = axes[1]
A_mat = np.array([[1, 2], [3, 4]])
B_mat = np.array([[5, 6], [7, 8]])
C_mat = A_mat @ B_mat

im = ax2.imshow(C_mat, cmap='YlOrRd', aspect='auto')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['col 0', 'col 1'])
ax2.set_yticklabels(['row 0', 'row 1'])

# 값 표시
for i in range(2):
    for j in range(2):
        text = ax2.text(j, i, f'{C_mat[i, j]}',
                       ha="center", va="center", color="black", fontsize=14, fontweight='bold')

ax2.set_title('Matrix Multiplication Result C = A × B', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2)

# 수식 표시
ax2.text(0.5, -0.3, f'A = {A_mat.tolist()}, B = {B_mat.tolist()}', 
         ha='center', transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ========== 3. 신경망 선형 변환 ==========
ax3 = axes[2]
# 3개 입력 -> 2개 출력
W = np.array([[0.5, 0.3, 0.2], [0.7, 0.4, 0.6]])
x_input = np.array([1.0, 2.0, 3.0])
b = np.array([0.1, 0.2])
z_output = W @ x_input + b

# 신경망 구조 시각화
input_layer_x = 0.2
hidden_layer_x = 0.8

# 입력 노드
for i in range(3):
    y = 0.2 + i * 0.3
    circle = plt.Circle((input_layer_x, y), 0.05, color='lightblue', ec='black', linewidth=2)
    ax3.add_patch(circle)
    ax3.text(input_layer_x - 0.15, y, f'x{i+1}={x_input[i]}', fontsize=10, ha='right')

# 출력 노드
for i in range(2):
    y = 0.35 + i * 0.3
    circle = plt.Circle((hidden_layer_x, y), 0.05, color='lightcoral', ec='black', linewidth=2)
    ax3.add_patch(circle)
    ax3.text(hidden_layer_x + 0.15, y, f'z{i+1}={z_output[i]:.2f}', fontsize=10, ha='left')

# 연결선 (가중치)
for i in range(3):
    for j in range(2):
        y_in = 0.2 + i * 0.3
        y_out = 0.35 + j * 0.3
        ax3.plot([input_layer_x + 0.05, hidden_layer_x - 0.05], [y_in, y_out], 
                'gray', alpha=0.3, linewidth=1)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Neural Network Linear Transformation', fontsize=14, fontweight='bold')
ax3.text(0.5, 0.05, 'z = Wx + b', fontsize=12, ha='center', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage1_matrix_operations.png', dpi=300, bbox_inches='tight')
print("Matrix operations visualization saved to stage1_matrix_operations.png")
plt.show()
```

### 5.3 시각화 결과 해설

#### 그림 1: 벡터 연산
1. **좌상단 - 벡터 표현**: 2차원 공간에서 두 벡터의 방향과 크기를 화살표로 표현
2. **우상단 - 벡터 덧셈**: 평행사변형 법칙을 보여줌. 벡터 a 끝점에서 시작하는 b와 원점에서 시작하는 합 벡터
3. **좌하단 - 내적**: 두 벡터의 내적 값과 사이각을 계산하여 표시
4. **우하단 - 선형 변환**: 45도 회전 변환을 적용한 벡터들의 변화

#### 그림 2: 행렬 연산
1. **좌측 - 행렬-벡터 곱**: 입력 벡터가 행렬에 의해 변환되어 다른 방향과 크기의 벡터로 변환
2. **중앙 - 행렬 곱셈**: 두 행렬의 곱셈 결과를 히트맵으로 시각화
3. **우측 - 신경망**: 3개 입력이 가중치 행렬을 통해 2개 출력으로 변환되는 과정

---

## 핵심 요약

### 수학 구조별 요약

| 구조 | 차원 | 표기 | 신경망 역할 | 예시 |
|------|------|------|------------|------|
| **스칼라** | 0차원 | $s$ | 학습률, 편향 | $0.01$ |
| **벡터** | 1차원 | $\mathbf{v}$ | 입력, 특성, 가중치 | $[1, 2, 3]$ |
| **행렬** | 2차원 | $\mathbf{A}$ | 가중치 행렬 | $$ \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix} $$  |
| **선형 변환** | 함수 | $\mathbf{y}=\mathbf{Ax}+\mathbf{b}$ | 층 간 변환 | 입력→은닉층 |


### 실생활 비유
- **스칼라**: 온도계의 온도 하나
- **벡터**: GPS 좌표 (위도, 경도)
- **행렬**: 교실의 학생 성적표 (학생×과목)
- **선형 변환**: 번역기 (한국어 문장 → 영어 문장)

### 신경망 연결
신경망은 이러한 수학 구조들의 조합입니다:
1. **입력**: 벡터 $\mathbf{x}$
2. **가중치**: 행렬 $\mathbf{W}$
3. **편향**: 벡터 $\mathbf{b}$
4. **변환**: $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ (선형 변환)
5. **활성화**: $\mathbf{a} = f(\mathbf{z})$ (다음 Stage에서 다룰 내용)

---

## 다음 단계 예고

**Stage 2**에서는 퍼셉트론과 가중합 수식 ($z = \mathbf{w} \cdot \mathbf{x} + b$)을 깊이 있게 다룰 예정입니다. 오늘 배운 벡터 내적과 선형 변환이 어떻게 퍼셉트론의 핵심 연산이 되는지 살펴보겠습니다!
