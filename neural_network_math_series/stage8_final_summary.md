# Stage 8: 최종 정리 - 신경망 학습 전 과정을 수식만으로 설명하기

## 🎯 목표
신경망 학습의 전체 과정을 **수식만으로** 완벽하게 설명합니다.

---

## 📐 Part 1: 기본 구성 요소

### 1.1 벡터와 행렬

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n, \quad
\mathbf{W} = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1n} \\
w_{21} & w_{22} & \cdots & w_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \cdots & w_{mn}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

### 1.2 선형 변환

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}, \quad \mathbf{z} \in \mathbb{R}^m, \ \mathbf{b} \in \mathbb{R}^m
$$

---

## 🔥 Part 2: 활성화 함수

$$
\begin{align}
\text{Sigmoid: } & \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z)) \\
\text{ReLU: } & f(z) = \max(0, z), \quad f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases} \\
\text{Tanh: } & \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)
\end{align}
$$

---

## 📊 Part 3: 손실 함수

$$
\begin{align}
\text{MSE: } & L = \frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2 \\
\text{Binary CE: } & L = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})] \\
\text{Categorical CE: } & L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{K} y_j^{(i)} \log(\hat{y}_j^{(i)})
\end{align}
$$

---

## 🔄 Part 4: 순방향 전파

**층 $l$의 계산 ($l = 1, 2, \ldots, L$):**

$$
\begin{align}
\mathbf{z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{a}^{[l]} &= f^{[l]}(\mathbf{z}^{[l]})
\end{align}
$$

**초기 조건:** $\mathbf{a}^{[0]} = \mathbf{x}$

**최종 출력:** $\hat{\mathbf{y}} = \mathbf{a}^{[L]}$

---

## ⬅️ Part 5: 역전파

### 5.1 출력층 그래디언트

$$
\delta^{[L]} = \frac{\partial L}{\partial \mathbf{z}^{[L]}} = \frac{\partial L}{\partial \mathbf{a}^{[L]}} \odot f'^{[L]}(\mathbf{z}^{[L]})
$$

### 5.2 은닉층 그래디언트 (층 $l = L-1, L-2, \ldots, 1$)

$$
\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot f'^{[l]}(\mathbf{z}^{[l]})
$$

### 5.3 파라미터 그래디언트

$$
\begin{align}
\frac{\partial L}{\partial \mathbf{W}^{[l]}} &= \frac{1}{m} \delta^{[l]} (\mathbf{a}^{[l-1]})^T \\
\frac{\partial L}{\partial \mathbf{b}^{[l]}} &= \frac{1}{m} \sum_{i=1}^{m} \delta^{[l](i)}
\end{align}
$$

---

## ⚙️ Part 6: 경사하강법

$$
\begin{align}
\mathbf{W}^{[l]} &:= \mathbf{W}^{[l]} - \alpha \frac{\partial L}{\partial \mathbf{W}^{[l]}} \\
\mathbf{b}^{[l]} &:= \mathbf{b}^{[l]} - \alpha \frac{\partial L}{\partial \mathbf{b}^{[l]}}
\end{align}
$$

여기서 $\alpha$는 학습률 (learning rate)

---

## 🔁 Part 7: 완전한 학습 알고리즘

```
입력: 학습 데이터 {(x^(i), y^(i))}_{i=1}^m, 학습률 α, 에폭 수 E

1. 파라미터 초기화: W^[l], b^[l] for l = 1, ..., L

2. For epoch = 1 to E:
   
   For each mini-batch:
   
      3. 순방향 전파:
         For l = 1 to L:
            z^[l] = W^[l] a^[l-1] + b^[l]
            a^[l] = f^[l](z^[l])
      
      4. 손실 계산:
         L = loss(a^[L], y)
      
      5. 역전파:
         δ^[L] = ∂L/∂z^[L]
         For l = L-1 down to 1:
            δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ f'^[l](z^[l])
      
      6. 그래디언트 계산:
         For l = 1 to L:
            ∂L/∂W^[l] = (1/m) δ^[l] (a^[l-1])^T
            ∂L/∂b^[l] = (1/m) Σ δ^[l]
      
      7. 파라미터 업데이트:
         For l = 1 to L:
            W^[l] := W^[l] - α ∂L/∂W^[l]
            b^[l] := b^[l] - α ∂L/∂b^[l]

출력: 최적화된 파라미터 W^[l], b^[l]
```

---

## 💡 Part 8: 핵심 수식 모음

### 8.1 순방향 전파 (한 샘플)

$$
\boxed{
\mathbf{a}^{[0]} = \mathbf{x} \quad \xrightarrow{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}} \quad \mathbf{z}^{[1]} \quad \xrightarrow{f^{[1]}} \quad \mathbf{a}^{[1]} \quad \xrightarrow{\mathbf{W}^{[2]}, \mathbf{b}^{[2]}} \quad \cdots \quad \xrightarrow{f^{[L]}} \quad \mathbf{a}^{[L]} = \hat{\mathbf{y}}
}
$$

### 8.2 역전파 (그래디언트 흐름)

$$
\boxed{
\frac{\partial L}{\partial \hat{\mathbf{y}}} \quad \xleftarrow{f'^{[L]}} \quad \delta^{[L]} \quad \xleftarrow{(\mathbf{W}^{[L]})^T, f'^{[L-1]}} \quad \delta^{[L-1]} \quad \xleftarrow{} \quad \cdots \quad \xleftarrow{} \quad \delta^{[1]}
}
$$

### 8.3 연쇄 법칙의 완전한 형태

$$
\boxed{
\frac{\partial L}{\partial \mathbf{W}^{[1]}} = \frac{\partial L}{\partial \mathbf{a}^{[L]}} \cdot \frac{\partial \mathbf{a}^{[L]}}{\partial \mathbf{z}^{[L]}} \cdot \frac{\partial \mathbf{z}^{[L]}}{\partial \mathbf{a}^{[L-1]}} \cdots \frac{\partial \mathbf{z}^{[2]}}{\partial \mathbf{a}^{[1]}} \cdot \frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} \cdot \frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}}
}
$$

---

## 🧩 Part 9: 구체적 예제 (2-3-1 신경망)

### 구조
- 입력: 2차원
- 은닉층: 3 뉴런 (Sigmoid)
- 출력: 1 뉴런 (Sigmoid)
- 손실: MSE

### 순방향

$$
\begin{align}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} \in \mathbb{R}^{3} \\
\mathbf{a}^{[1]} &= \sigma(\mathbf{z}^{[1]}) \in \mathbb{R}^{3} \\
z^{[2]} &= (\mathbf{w}^{[2]})^T \mathbf{a}^{[1]} + b^{[2]} \in \mathbb{R} \\
\hat{y} &= \sigma(z^{[2]}) \in \mathbb{R} \\
L &= \frac{1}{2}(\hat{y} - y)^2
\end{align}
$$

### 역전파

$$
\begin{align}
\delta^{[2]} &= (\hat{y} - y) \cdot \sigma'(z^{[2]}) \\
\boldsymbol{\delta}^{[1]} &= \mathbf{w}^{[2]} \delta^{[2]} \odot \sigma'(\mathbf{z}^{[1]}) \\
\frac{\partial L}{\partial \mathbf{W}^{[1]}} &= \boldsymbol{\delta}^{[1]} \mathbf{x}^T \\
\frac{\partial L}{\partial \mathbf{w}^{[2]}} &= \delta^{[2]} \mathbf{a}^{[1]}
\end{align}
$$

---

## 🎓 Part 10: 최종 통합 수식

**신경망 학습의 본질을 하나의 최적화 문제로:**

$$
\boxed{
\begin{align}
&\text{최소화: } J(\mathbf{W}, \mathbf{b}) = \frac{1}{m}\sum_{i=1}^{m} L(f_{\mathbf{W}, \mathbf{b}}(\mathbf{x}^{(i)}), \mathbf{y}^{(i)}) \\
&\text{제약 조건: } f_{\mathbf{W}, \mathbf{b}}(\mathbf{x}) = f^{[L]}(\mathbf{W}^{[L]} f^{[L-1]}(\cdots f^{[1]}(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) \cdots) + \mathbf{b}^{[L]}) \\
&\text{해법: } \mathbf{W}^{*}, \mathbf{b}^{*} = \arg\min_{\mathbf{W}, \mathbf{b}} J(\mathbf{W}, \mathbf{b})
\end{align}
}
$$

**경사하강법으로 해결:**

$$
\boxed{
(\mathbf{W}^{[l]}, \mathbf{b}^{[l]})_{t+1} = (\mathbf{W}^{[l]}, \mathbf{b}^{[l]})_t - \alpha \nabla_{(\mathbf{W}^{[l]}, \mathbf{b}^{[l]})} J
}
$$

---

## 📖 Part 11: 기호 정리

| 기호 | 의미 | 차원 |
|------|------|------|
| $\mathbf{x}$ | 입력 벡터 | $\mathbb{R}^{n_0}$ |
| $\mathbf{W}^{[l]}$ | 층 $l$의 가중치 행렬 | $\mathbb{R}^{n_l \times n_{l-1}}$ |
| $\mathbf{b}^{[l]}$ | 층 $l$의 편향 벡터 | $\mathbb{R}^{n_l}$ |
| $\mathbf{z}^{[l]}$ | 층 $l$의 가중합 | $\mathbb{R}^{n_l}$ |
| $\mathbf{a}^{[l]}$ | 층 $l$의 활성화 | $\mathbb{R}^{n_l}$ |
| $f^{[l]}$ | 층 $l$의 활성화 함수 | $\mathbb{R}^{n_l} \to \mathbb{R}^{n_l}$ |
| $\delta^{[l]}$ | 층 $l$의 오차 | $\mathbb{R}^{n_l}$ |
| $L$ | 손실 함수 | $\mathbb{R}$ |
| $\alpha$ | 학습률 | $\mathbb{R}^+$ |
| $m$ | 배치 크기 | $\mathbb{N}$ |
| $L$ | 총 층 수 | $\mathbb{N}$ |

---

## 🚀 Part 12: 결론

신경망 학습은 결국:

1. **목적**: 손실 함수 $J(\mathbf{W}, \mathbf{b})$를 최소화
2. **방법**: 경사하강법 $\theta := \theta - \alpha \nabla J(\theta)$
3. **핵심**: 역전파로 효율적인 그래디언트 계산

**한 문장 요약:**

> 신경망 학습은 **역전파**를 통해 계산한 **그래디언트**를 이용하여 **경사하강법**으로 **손실 함수를 최소화**하는 **최적화 과정**이다.

**수식으로 요약:**

$$
\boxed{
\min_{\theta} J(\theta) \quad \text{s.t.} \quad \theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
}
$$

---

## 🎯 최종 메시지

당신은 이제 신경망 학습의 수학적 기초를 **완전히** 이해했습니다!

$$
\Large\boxed{\text{축하합니다! 🎉}}
$$

---

**작성 완료 시각**: 2024년 기준  
**난이도**: ⭐⭐⭐⭐⭐ (마스터)  
**예상 학습 시간**: 모든 Stage 복습 포함 4-6시간
