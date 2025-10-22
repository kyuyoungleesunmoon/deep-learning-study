# Stage 7: 신경망 학습 - 1층에서 다층까지

## 📚 목차
1. [단층 신경망 학습](#1-단층-신경망-학습)
2. [다층 신경망 구조](#2-다층-신경망-구조)
3. [역전파 알고리즘](#3-역전파-알고리즘)
4. [전체 학습 과정](#4-전체-학습-과정)
5. [구현 예제](#5-구현-예제)

---

## 1. 단층 신경망 학습

### 1.1 구조

**입력 → 은닉층 → 출력**

$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}
$$
$$
\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})
$$
$$
\hat{y} = \mathbf{a}^{[1]}
$$

### 1.2 순방향 전파 (Forward Propagation)

**단계:**
1. 입력 $\mathbf{x}$
2. 가중합: $\mathbf{z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$
3. 활성화: $\mathbf{a}^{[1]} = f(\mathbf{z}^{[1]})$
4. 손실: $L = \frac{1}{2}(\mathbf{a}^{[1]} - \mathbf{y})^2$

### 1.3 역전파 (Backpropagation)

**그래디언트 계산:**

$$
\frac{\partial L}{\partial \mathbf{a}^{[1]}} = \mathbf{a}^{[1]} - \mathbf{y}
$$

$$
\frac{\partial L}{\partial \mathbf{z}^{[1]}} = \frac{\partial L}{\partial \mathbf{a}^{[1]}} \odot f'(\mathbf{z}^{[1]})
$$

$$
\frac{\partial L}{\partial \mathbf{W}^{[1]}} = \frac{\partial L}{\partial \mathbf{z}^{[1]}} \mathbf{x}^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{[1]}} = \frac{\partial L}{\partial \mathbf{z}^{[1]}}
$$

**기호 설명:**
- $\odot$: 원소별 곱 (element-wise multiplication)
- $f'$: 활성화 함수의 미분

### 1.4 파라미터 업데이트

$$
\mathbf{W}^{[1]} := \mathbf{W}^{[1]} - \alpha \frac{\partial L}{\partial \mathbf{W}^{[1]}}
$$

$$
\mathbf{b}^{[1]} := \mathbf{b}^{[1]} - \alpha \frac{\partial L}{\partial \mathbf{b}^{[1]}}
$$

---

## 2. 다층 신경망 구조

### 2.1 2층 신경망

**입력 → 은닉층 1 → 은닉층 2 → 출력**

**순방향:**
$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}
$$
$$
\mathbf{a}^{[1]} = f^{[1]}(\mathbf{z}^{[1]})
$$
$$
\mathbf{z}^{[2]} = \mathbf{W}^{[2]}\mathbf{a}^{[1]} + \mathbf{b}^{[2]}
$$
$$
\mathbf{a}^{[2]} = f^{[2]}(\mathbf{z}^{[2]})
$$
$$
\hat{\mathbf{y}} = \mathbf{a}^{[2]}
$$

### 2.2 일반화된 $L$층 신경망

**층 $l$의 계산:**
$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$
$$
\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})
$$

여기서 $\mathbf{a}^{[0]} = \mathbf{x}$ (입력)

---

## 3. 역전파 알고리즘

### 3.1 출력층에서 시작

**손실 함수의 그래디언트:**
$$
\delta^{[L]} = \frac{\partial L}{\partial \mathbf{z}^{[L]}} = \frac{\partial L}{\partial \mathbf{a}^{[L]}} \odot f'^{[L]}(\mathbf{z}^{[L]})
$$

### 3.2 역방향으로 전파

**층 $l$의 그래디언트:**
$$
\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot f'^{[l]}(\mathbf{z}^{[l]})
$$

### 3.3 파라미터 그래디언트

$$
\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}
$$

### 3.4 수치 예제

**2층 신경망:** 2 입력 → 3 은닉 → 1 출력

**주어진 값:**
- $\mathbf{x} = [1, 2]^T$
- $\mathbf{y} = 1$
- 모든 가중치와 편향은 0.5로 초기화
- 활성화: Sigmoid
- 손실: MSE

**순방향:**
1. $\mathbf{z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$
2. $\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})$
3. $\mathbf{z}^{[2]} = \mathbf{W}^{[2]}\mathbf{a}^{[1]} + \mathbf{b}^{[2]}$
4. $\hat{y} = \sigma(\mathbf{z}^{[2]})$
5. $L = \frac{1}{2}(\hat{y} - y)^2$

**역전파:**
1. $\delta^{[2]} = (\hat{y} - y) \cdot \sigma'(\mathbf{z}^{[2]})$
2. $\delta^{[1]} = (\mathbf{W}^{[2]})^T \delta^{[2]} \odot \sigma'(\mathbf{z}^{[1]})$
3. 그래디언트 계산 및 업데이트

---

## 4. 전체 학습 과정

### 4.1 알고리즘

```
초기화: W, b를 랜덤 값으로
for epoch in 1 to max_epochs:
    for each mini-batch:
        # 순방향 전파
        for l in 1 to L:
            z[l] = W[l] @ a[l-1] + b[l]
            a[l] = f[l](z[l])
        
        # 손실 계산
        L = loss(a[L], y)
        
        # 역전파
        δ[L] = ∂L/∂z[L]
        for l in L-1 down to 1:
            δ[l] = (W[l+1])^T @ δ[l+1] ⊙ f'[l](z[l])
        
        # 파라미터 업데이트
        for l in 1 to L:
            W[l] -= α * ∂L/∂W[l]
            b[l] -= α * ∂L/∂b[l]
```

### 4.2 핵심 방정식 요약

**순방향:**
$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \quad \mathbf{a}^{[l]} = f(\mathbf{z}^{[l]})
$$

**역전파:**
$$
\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot f'(\mathbf{z}^{[l]})
$$

**업데이트:**
$$
\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \delta^{[l]} (\mathbf{a}^{[l-1]})^T
$$

---

## 5. 구현 예제

### 5.1 Python 구현 (NumPy)

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dims):
        self.L = len(layer_dims) - 1
        self.parameters = {}
        
        # 초기화
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        self.cache = {'A0': X}
        A = X
        
        for l in range(1, self.L + 1):
            Z = self.parameters['W' + str(l)] @ A + self.parameters['b' + str(l)]
            A = self.sigmoid(Z)
            self.cache['Z' + str(l)] = Z
            self.cache['A' + str(l)] = A
        
        return A
    
    def backward(self, Y):
        m = Y.shape[1]
        grads = {}
        
        # 출력층
        dZ = self.cache['A' + str(self.L)] - Y
        grads['dW' + str(self.L)] = (1/m) * dZ @ self.cache['A' + str(self.L-1)].T
        grads['db' + str(self.L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # 은닉층
        for l in reversed(range(1, self.L)):
            dA = self.parameters['W' + str(l+1)].T @ dZ
            dZ = dA * self.sigmoid_derivative(self.cache['Z' + str(l)])
            grads['dW' + str(l)] = (1/m) * dZ @ self.cache['A' + str(l-1)].T
            grads['db' + str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return grads
    
    def update(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    
    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            # 순방향
            A = self.forward(X)
            
            # 손실 계산
            loss = np.mean((A - Y)**2)
            
            # 역전파
            grads = self.backward(Y)
            
            # 업데이트
            self.update(grads, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 사용 예
nn = NeuralNetwork([2, 4, 1])  # 2 입력, 4 은닉, 1 출력
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # XOR 입력
Y = np.array([[0, 1, 1, 0]])  # XOR 출력
nn.train(X, Y, epochs=1000, learning_rate=0.5)
```

---

## 📝 핵심 요약

### 신경망 학습의 3단계

1. **순방향 전파**: 입력 → 출력 계산
2. **손실 계산**: 예측과 실제값 비교
3. **역전파**: 그래디언트 계산 및 파라미터 업데이트

### 수학적 핵심

$$
\boxed{
\begin{align}
&\text{순방향: } \mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \quad \mathbf{a}^{[l]} = f(\mathbf{z}^{[l]}) \\
&\text{역전파: } \delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot f'(\mathbf{z}^{[l]}) \\
&\text{업데이트: } \mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \frac{\partial L}{\partial \mathbf{W}^{[l]}}
\end{align}
}
$$

---

## 🎯 다음 단계 예고

**Stage 8**에서는 지금까지 배운 모든 내용을 **수식만으로 완전히 정리**하는 최종 마스터 문서를 제공합니다!

---

**작성 완료 시각**: 2024년 기준  
**난이도**: ⭐⭐⭐⭐⭐ (고급)  
**예상 학습 시간**: 90-120분
