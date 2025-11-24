"""
Chapter 16: Mamba 아키텍처 실습 코드
====================================

이 파일은 Mamba (State Space Model)의 핵심 개념을 실습합니다:
1. State Space Model 기초
2. 이산화 (Discretization)
3. Selective SSM 개념
4. 간단한 Mamba Block

실행 방법:
    pip install numpy
    python chapter_16_practice.py

    # PyTorch 사용 시:
    pip install torch
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


# ============================================================
# Part 1: State Space Model 기초
# ============================================================

@dataclass
class SSMParams:
    """SSM 파라미터"""
    A: np.ndarray  # 상태 전이 행렬
    B: np.ndarray  # 입력 행렬
    C: np.ndarray  # 출력 행렬
    D: np.ndarray  # 피드스루 행렬


def continuous_to_discrete(A: np.ndarray, B: np.ndarray, 
                           delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    연속 시간 SSM을 이산 시간으로 변환
    
    Zero-Order Hold (ZOH) 방식:
    A_bar = exp(A * delta)
    B_bar = (A^-1) * (exp(A * delta) - I) * B
    
    간소화 버전 (Euler):
    A_bar = I + A * delta
    B_bar = B * delta
    """
    d_state = A.shape[0]
    I = np.eye(d_state)
    
    # Euler 방식 (간단)
    A_bar = I + A * delta
    B_bar = B * delta
    
    return A_bar, B_bar


def ssm_forward(params: SSMParams, x: np.ndarray, 
                delta: float = 0.1) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    SSM 순전파
    
    Args:
        params: SSM 파라미터 (A, B, C, D)
        x: (seq_len,) 입력 시퀀스
        delta: 이산화 간격
    
    Returns:
        y: (seq_len,) 출력 시퀀스
        states: 각 스텝의 은닉 상태
    """
    seq_len = len(x)
    d_state = params.A.shape[0]
    
    # 이산화
    A_bar, B_bar = continuous_to_discrete(params.A, params.B, delta)
    
    # 초기 상태
    h = np.zeros(d_state)
    
    outputs = []
    states = [h.copy()]
    
    for t in range(seq_len):
        # h_t = A_bar * h_{t-1} + B_bar * x_t
        h = A_bar @ h + B_bar * x[t]
        
        # y_t = C * h_t + D * x_t
        y = params.C @ h + params.D * x[t]
        
        outputs.append(y[0])
        states.append(h.copy())
    
    return np.array(outputs), states


def demo_basic_ssm():
    """기본 SSM 데모"""
    print("\n" + "="*60)
    print("📊 기본 SSM 데모")
    print("="*60)
    
    # 간단한 SSM 파라미터 설정
    d_state = 4
    
    # 안정적인 A 행렬 (eigenvalue < 1)
    np.random.seed(42)
    A = np.diag([-0.1, -0.2, -0.3, -0.4])  # 대각 행렬
    B = np.random.randn(d_state, 1) * 0.5
    C = np.random.randn(1, d_state) * 0.5
    D = np.array([[0.0]])
    
    params = SSMParams(A=A, B=B, C=C, D=D)
    
    # 입력 시퀀스 (임펄스 응답)
    seq_len = 20
    x_impulse = np.zeros(seq_len)
    x_impulse[0] = 1.0  # 임펄스
    
    # SSM 실행
    y, states = ssm_forward(params, x_impulse, delta=0.5)
    
    print(f"\n입력 (임펄스): {x_impulse[:10]}...")
    print(f"출력: {[f'{v:.4f}' for v in y[:10]]}...")
    
    # 상태 변화 시각화
    print("\n상태 변화 (처음 5 스텝):")
    for t in range(min(5, len(states))):
        state_str = [f"{v:.3f}" for v in states[t]]
        print(f"  t={t}: {state_str}")


# ============================================================
# Part 2: Selective SSM 개념
# ============================================================

def selective_ssm_forward(x: np.ndarray, 
                          d_state: int = 4,
                          d_inner: int = 8) -> np.ndarray:
    """
    Selective SSM 순전파 (Mamba 스타일)
    
    핵심 아이디어: A, B, C, delta를 입력 x에서 생성
    
    Args:
        x: (seq_len, d_inner) 입력 시퀀스
        d_state: 상태 차원
        d_inner: 입력 차원
    
    Returns:
        y: (seq_len, d_inner) 출력 시퀀스
    """
    seq_len = x.shape[0]
    
    # 투영 가중치 (실제로는 학습됨)
    np.random.seed(42)
    W_delta = np.random.randn(d_inner, d_inner) * 0.1
    W_B = np.random.randn(d_inner, d_state) * 0.1
    W_C = np.random.randn(d_inner, d_state) * 0.1
    
    # A 행렬 (로그 스케일, 음수)
    A_log = -np.abs(np.random.randn(d_inner, d_state))
    
    # 초기 상태
    h = np.zeros((d_inner, d_state))
    
    outputs = []
    
    for t in range(seq_len):
        x_t = x[t]  # (d_inner,)
        
        # 입력 의존적 파라미터 생성
        delta = np.maximum(0.01, W_delta @ x_t)  # softplus 근사, (d_inner,)
        B = x_t[:, np.newaxis] @ W_B[np.newaxis, :d_inner, :]  # (d_inner, d_state)
        B = B.mean(axis=0)  # 간소화
        C = W_C.T @ x_t  # (d_state,)
        
        # A 이산화
        A = np.exp(A_log)  # (d_inner, d_state)
        A_bar = A * delta[:, np.newaxis]
        
        # SSM 스텝
        # h: (d_inner, d_state)
        h = h * A_bar + np.outer(x_t, B.mean(axis=0))
        
        # 출력
        y_t = (h * C).sum(axis=1)  # (d_inner,)
        outputs.append(y_t)
    
    return np.array(outputs)


def demo_selective_ssm():
    """Selective SSM 데모"""
    print("\n" + "="*60)
    print("🎯 Selective SSM 데모")
    print("="*60)
    
    np.random.seed(42)
    seq_len = 10
    d_inner = 8
    
    # 입력 시퀀스
    x = np.random.randn(seq_len, d_inner)
    
    # Selective SSM 실행
    y = selective_ssm_forward(x, d_state=4, d_inner=d_inner)
    
    print(f"\n입력 shape: {x.shape}")
    print(f"출력 shape: {y.shape}")
    
    print("\n입력 (처음 3 스텝):")
    for t in range(3):
        print(f"  t={t}: [{x[t, :4].round(2)}...]")
    
    print("\n출력 (처음 3 스텝):")
    for t in range(3):
        print(f"  t={t}: [{y[t, :4].round(2)}...]")


# ============================================================
# Part 3: 시간 복잡도 비교
# ============================================================

def attention_complexity(seq_len: int) -> int:
    """Self-Attention 복잡도: O(n²)"""
    return seq_len * seq_len


def ssm_complexity(seq_len: int, d_state: int = 16) -> int:
    """SSM 복잡도: O(n)"""
    return seq_len * d_state


def demo_complexity():
    """시간 복잡도 비교 데모"""
    print("\n" + "="*60)
    print("⚡ 시간 복잡도 비교")
    print("="*60)
    
    seq_lengths = [100, 1000, 10000, 100000]
    
    print("\n시퀀스 길이별 연산 수:")
    print(f"{'시퀀스 길이':>12s} | {'Attention':>15s} | {'SSM':>15s} | {'비율':>10s}")
    print("-" * 60)
    
    for n in seq_lengths:
        attn = attention_complexity(n)
        ssm = ssm_complexity(n)
        ratio = attn / ssm
        
        print(f"{n:>12,d} | {attn:>15,d} | {ssm:>15,d} | {ratio:>10.0f}x")
    
    print("\n→ 시퀀스가 길어질수록 SSM의 효율성이 증가!")


# ============================================================
# Part 4: 간단한 Mamba Block (NumPy)
# ============================================================

class SimpleMambaBlockNumpy:
    """
    간소화된 Mamba Block (NumPy)
    
    구조:
    1. Linear projection (d_model → 2 * d_inner)
    2. Conv1D
    3. SSM
    4. Gating
    5. Linear projection (d_inner → d_model)
    """
    
    def __init__(self, d_model: int = 64, d_inner: int = 128, 
                 d_state: int = 16, d_conv: int = 4):
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        
        # 가중치 초기화 (Xavier)
        np.random.seed(42)
        scale = np.sqrt(2.0 / (d_model + d_inner))
        
        self.in_proj = np.random.randn(d_model, d_inner * 2) * scale
        self.conv_weight = np.random.randn(d_conv, d_inner) * 0.1
        self.out_proj = np.random.randn(d_inner, d_model) * scale
        
        # SSM 파라미터
        self.A_log = -np.abs(np.random.randn(d_inner, d_state))
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU (Swish) 활성화 함수"""
        return x * (1 / (1 + np.exp(-x)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        순전파
        
        Args:
            x: (seq_len, d_model) 입력
        
        Returns:
            y: (seq_len, d_model) 출력
        """
        seq_len = x.shape[0]
        
        # 1. 입력 투영
        xz = x @ self.in_proj  # (seq_len, 2 * d_inner)
        x_proj, z = np.split(xz, 2, axis=-1)  # 각각 (seq_len, d_inner)
        
        # 2. 1D Convolution (간소화)
        x_conv = self._conv1d(x_proj)
        x_conv = self.silu(x_conv)
        
        # 3. SSM
        y = self._ssm(x_conv)
        
        # 4. Gating
        y = y * self.silu(z)
        
        # 5. 출력 투영
        out = y @ self.out_proj  # (seq_len, d_model)
        
        return out
    
    def _conv1d(self, x: np.ndarray) -> np.ndarray:
        """간소화된 1D Convolution"""
        seq_len, d_inner = x.shape
        
        # Causal padding
        padded = np.vstack([
            np.zeros((self.d_conv - 1, d_inner)),
            x
        ])
        
        # Depthwise convolution
        out = np.zeros_like(x)
        for t in range(seq_len):
            window = padded[t:t + self.d_conv, :]
            out[t] = (window * self.conv_weight).sum(axis=0)
        
        return out
    
    def _ssm(self, x: np.ndarray) -> np.ndarray:
        """간소화된 SSM"""
        seq_len, d_inner = x.shape
        
        # 상태 초기화
        h = np.zeros((d_inner, self.d_state))
        
        A = np.exp(self.A_log)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]  # (d_inner,)
            
            # 간소화된 SSM 스텝
            delta = 0.1  # 고정 delta (실제로는 입력 의존적)
            A_bar = A * delta
            
            h = h * A_bar + x_t[:, np.newaxis] * delta
            y_t = h.sum(axis=1)
            outputs.append(y_t)
        
        return np.array(outputs)


def demo_mamba_block():
    """Mamba Block 데모"""
    print("\n" + "="*60)
    print("🧱 Mamba Block 데모")
    print("="*60)
    
    # 모델 생성
    mamba = SimpleMambaBlockNumpy(d_model=32, d_inner=64, d_state=8)
    
    # 입력
    np.random.seed(123)
    seq_len = 20
    x = np.random.randn(seq_len, 32)
    
    # 순전파
    y = mamba.forward(x)
    
    print(f"\n입력 shape: {x.shape}")
    print(f"출력 shape: {y.shape}")
    
    print(f"\n입력 (처음 3개, 처음 4차원):")
    for t in range(3):
        print(f"  t={t}: {x[t, :4].round(3)}")
    
    print(f"\n출력 (처음 3개, 처음 4차원):")
    for t in range(3):
        print(f"  t={t}: {y[t, :4].round(3)}")


# ============================================================
# Part 5: PyTorch Mamba (선택적)
# ============================================================

def demo_pytorch_mamba():
    """PyTorch Mamba 데모"""
    try:
        import torch
        import torch.nn as nn
        
        print("\n" + "="*60)
        print("🚀 PyTorch Mamba Block 데모")
        print("="*60)
        
        class MambaBlock(nn.Module):
            def __init__(self, d_model=64, d_inner=128, d_state=16):
                super().__init__()
                self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
                self.out_proj = nn.Linear(d_inner, d_model, bias=False)
                self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
            
            def forward(self, x):
                xz = self.in_proj(x)
                x, z = xz.chunk(2, dim=-1)
                # 간소화된 SSM
                y = torch.silu(x)
                y = y * torch.silu(z)
                return self.out_proj(y)
        
        model = MambaBlock()
        x = torch.randn(2, 100, 64)
        y = model(x)
        
        print(f"입력 shape: {x.shape}")
        print(f"출력 shape: {y.shape}")
        print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
    except ImportError:
        print("\n⚠️ torch가 설치되지 않았습니다.")
        print("설치: pip install torch")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 16: Mamba 아키텍처 실습")
    print("="*60)
    
    demo_basic_ssm()
    demo_selective_ssm()
    demo_complexity()
    demo_mamba_block()
    demo_pytorch_mamba()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
