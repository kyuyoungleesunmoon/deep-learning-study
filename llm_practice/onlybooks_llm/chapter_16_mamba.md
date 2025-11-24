# 📖 Chapter 16: Mamba 아키텍처

## 📋 개요

이 챕터에서는 Transformer의 대안 아키텍처인 Mamba를 학습합니다.
- State Space Models (SSM)
- Selective State Space (S6)
- 선형 시간 복잡도의 장점

## 🔬 핵심 알고리즘

### 1. Transformer의 한계

**Self-Attention의 문제점**:
```
시간 복잡도: O(n²) (n: 시퀀스 길이)
공간 복잡도: O(n²)
```

예: 100K 토큰 처리 시 10억 번 연산 필요!

**Mamba의 해결책**:
```
시간 복잡도: O(n) 선형
공간 복잡도: O(1) 상수
```

### 2. State Space Models (SSM)

**연속 시간 SSM**:
```
h'(t) = A·h(t) + B·x(t)
y(t) = C·h(t) + D·x(t)
```

**이산화 (Discretization)**:
```
h_k = Ā·h_{k-1} + B̄·x_k
y_k = C·h_k
```

**수식 설명**:
- `h`: 은닉 상태 (hidden state)
- `x`: 입력
- `y`: 출력
- `A, B, C, D`: 학습 가능한 파라미터
- `Ā, B̄`: 이산화된 행렬

### 3. S6 (Selective State Space)

**기존 SSM의 문제**:
- 파라미터 A, B, C가 입력에 무관 (time-invariant)
- 컨텍스트 인식 능력 부족

**Mamba의 해결책**:
- 파라미터를 입력 의존적으로 변환
- 선택적 정보 필터링

```python
# 입력 x에서 파라미터 생성
Δ = softplus(Linear(x))  # 시간 간격
B = Linear(x)            # 입력 행렬
C = Linear(x)            # 출력 행렬
```

### 4. Mamba Block 구조

```
Input (b, l, d)
    │
    ├──────────────────┐
    │                  │
    ▼                  │
Linear (2 × d_inner)   │
    │                  │
    ├─────┬────────────┤
    │     │            │
    ▼     ▼            │
   SiLU   x            │
    │     │            │
    ▼     │            │
Conv1D   │            │
    │     │            │
    ▼     │            │
  SiLU    │            │
    │     │            │
    ▼     │            │
  SSM     │            │
    │     │            │
    ▼     ▼            │
    ×─────┘            │
    │                  │
    ▼                  │
Linear (d)             │
    │                  │
    ▼                  │
   Add ◄───────────────┘
    │
Output (b, l, d)
```

### 5. 핵심 혁신

| 특성 | Transformer | Mamba |
|------|-------------|-------|
| 시간 복잡도 | O(n²) | O(n) |
| 메모리 | O(n²) | O(1) |
| 긴 시퀀스 | 어려움 | 효율적 |
| 병렬화 | 좋음 | 보통 |
| 컨텍스트 | 전체 어텐션 | 선택적 |

## 📊 실습 예제

### 예제 1: SSM 수식 이해

```python
import numpy as np

def discrete_ssm_step(A, B, C, h_prev, x):
    """
    이산 SSM 한 스텝 계산
    
    h_k = A·h_{k-1} + B·x_k
    y_k = C·h_k
    """
    h = A @ h_prev + B @ x
    y = C @ h
    return h, y

# 예시 파라미터
d_state = 16
d_input = 1

np.random.seed(42)
A = np.eye(d_state) * 0.9  # 안정성을 위해 eigenvalue < 1
B = np.random.randn(d_state, d_input) * 0.1
C = np.random.randn(d_input, d_state) * 0.1

# 시퀀스 처리
sequence = [1.0, 0.5, -0.5, 1.0, 0.0]
h = np.zeros((d_state, 1))

outputs = []
for x in sequence:
    x_vec = np.array([[x]])
    h, y = discrete_ssm_step(A, B, C, h, x_vec)
    outputs.append(y[0, 0])

print(f"입력: {sequence}")
print(f"출력: {[f'{o:.4f}' for o in outputs]}")
```

### 예제 2: 선택적 SSM 개념

```python
import numpy as np

def selective_ssm_step(x, h_prev, W_delta, W_B, W_C, A_log):
    """
    Selective SSM 한 스텝 (Mamba 스타일)
    
    파라미터가 입력 x에 의존적
    """
    # 입력 의존적 파라미터 생성
    delta = np.maximum(0, W_delta @ x)  # softplus 근사
    B = W_B @ x
    C = W_C @ x
    
    # A 행렬 (로그 스케일에서 변환)
    A = np.exp(A_log)
    
    # 이산화 (간소화 버전)
    A_bar = A * delta
    B_bar = B * delta
    
    # SSM 스텝
    h = A_bar * h_prev + B_bar * x
    y = C @ h
    
    return h, y

# 이 방식으로 입력에 따라 정보를 선택적으로 저장/삭제
```

### 예제 3: Mamba Block 구조 (PyTorch 스타일)

```python
import torch
import torch.nn as nn

class SimpleMambaBlock(nn.Module):
    """
    간소화된 Mamba Block 구현
    
    실제 Mamba는 더 복잡한 최적화 포함
    """
    
    def __init__(self, d_model=64, d_inner=128, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        
        # 입력 투영
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # 1D Convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner
        )
        
        # SSM 파라미터 생성
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        
        # 출력 투영
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # A 행렬 (학습 가능)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        
        # 입력 투영
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # x, gate
        
        # Conv1D
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # 활성화
        x = torch.silu(x)
        
        # SSM (간소화)
        # 실제로는 더 효율적인 구현 필요
        y = self._ssm(x)
        
        # Gating
        y = y * torch.silu(z)
        
        # 출력 투영
        return self.out_proj(y)
    
    def _ssm(self, x):
        # 간소화된 SSM
        # 실제 구현은 scan 연산 사용
        return x  # 데모용 identity

# 사용 예시
model = SimpleMambaBlock(d_model=64)
x = torch.randn(2, 100, 64)  # (batch, seq_len, d_model)
y = model(x)
print(f"입력 shape: {x.shape}")
print(f"출력 shape: {y.shape}")
```

## 🎯 핵심 포인트

1. **선형 시간 복잡도**: Transformer의 O(n²) → O(n)
2. **선택적 메커니즘**: 입력에 따라 정보 저장/삭제 결정
3. **상태 공간**: RNN과 CNN의 장점 결합
4. **긴 시퀀스**: 100K+ 토큰도 효율적 처리

## ⚠️ 주의사항

- GPU 최적화 필요 (CUDA 커널)
- 병렬화가 Transformer보다 어려움
- 아직 발전 중인 아키텍처

## 📚 참고 자료

- 원본 코드: https://github.com/onlybooks/llm/tree/main/16장
- Mamba 논문: https://arxiv.org/abs/2312.00752
- mamba-minimal: https://github.com/johnma2006/mamba-minimal
