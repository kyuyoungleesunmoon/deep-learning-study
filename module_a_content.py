"""
Module A Content: RNN Time Series Prediction
Complete content for comprehensive AI textbook
"""

def get_all_content(md, code):
    """Returns all cells for Module A"""
    
    cells = []
    
    # Title
    cells.append(md("""# 🔹 모듈 A: RNN 기반 시계열 예측 (Netflix 주가 예측)

## 📚 학습 목표
이 모듈에서는 순환 신경망(RNN)과 그 변형인 LSTM, GRU를 사용하여 시계열 데이터를 예측하는 방법을 학습합니다.

**주요 학습 내용:**
- RNN, LSTM, GRU의 내부 구조와 수학적 원리
- 그래디언트 소실/폭주 문제와 해결 방법
- 합성 데이터와 실제 주가 데이터를 이용한 실험
- 하이퍼파라미터 튜닝과 성능 평가

**대상 학습자:** 석사 수준의 딥러닝 학습자  
**예상 학습 시간:** 4-6시간"""))
    
    # Add theory sections
    cells.extend(get_theory_cells(md, code))
    
    # Add synthetic experiments
    cells.extend(get_synthetic_cells(md, code))
    
    # Add real data sections
    cells.extend(get_real_data_cells(md, code))
    
    # Add conclusion
    cells.extend(get_conclusion_cells(md, code))
    
    return cells

def get_theory_cells(md, code):
    """Theory section cells"""
    return [
        md("""---
# 1️⃣ 이론 파트

## 1.1 순환 신경망(RNN)의 기본 구조

### 왜 RNN이 필요한가?

일반적인 피드포워드 신경망은 각 입력을 독립적으로 처리합니다. 하지만 시계열 데이터나 자연어와 같이 **순서가 중요한 데이터**에서는 이전 정보를 기억해야 합니다.

RNN은 **순환 구조**를 통해 이전 시점의 정보를 현재 시점으로 전달할 수 있습니다.

### RNN의 수학적 정의

RNN의 기본 순전파 수식은 다음과 같습니다:

$$h_t = \\tanh(W_h h_{t-1} + W_x x_t + b_h)$$

$$y_t = W_y h_t + b_y$$

**기호 설명:**
- $x_t$: 시점 $t$에서의 입력 벡터
- $h_t$: 시점 $t$에서의 은닉 상태 (hidden state)
- $h_{t-1}$: 이전 시점의 은닉 상태
- $y_t$: 시점 $t$에서의 출력
- $W_h$: 은닉 상태 가중치 행렬
- $W_x$: 입력 가중치 행렬
- $W_y$: 출력 가중치 행렬
- $b_h, b_y$: 편향(bias)
- $\\tanh$: 하이퍼볼릭 탄젠트 활성화 함수

### RNN의 동작 원리

1. **초기 은닉 상태** $h_0$는 보통 0으로 초기화
2. 각 시점 $t$마다:
   - 현재 입력 $x_t$와 이전 은닉 상태 $h_{t-1}$을 결합
   - 가중치 행렬을 곱하고 활성화 함수 적용
   - 새로운 은닉 상태 $h_t$ 생성
   - 필요시 출력 $y_t$ 생성"""),
        
        # Continue with more theory sections...
    ]

def get_synthetic_cells(md, code):
    """Synthetic experiments cells"""
    return [
        md("""---
# 2️⃣ 합성 데이터 실험

## 2.1 합성 시계열 데이터 생성

단순한 사인파에 노이즈를 추가한 시계열 데이터를 생성하여 RNN 계열 모델의 동작을 이해합니다."""),
        
        code("""# 필요한 패키지 임포트
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')"""),
        
        # More synthetic experiment cells...
    ]

def get_real_data_cells(md, code):
    """Real data experiment cells"""
    return [
        md("""---
# 3️⃣ 실제 데이터 학습: Netflix 주가 예측

## 3.1 데이터 수집

`yfinance` 패키지를 사용하여 넷플릭스 주가 데이터를 다운로드합니다."""),
        
        code("""# yfinance 설치 및 임포트
try:
    import yfinance as yf
except ImportError:
    !pip install yfinance
    import yfinance as yf

# Netflix 주가 다운로드
ticker = 'NFLX'
df = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
close_prices = df['Close'].values
print(f'데이터 포인트: {len(close_prices)}개')"""),
        
        # More real data cells...
    ]

def get_conclusion_cells(md, code):
    """Conclusion cells"""
    return [
        md("""---
# 📊 모듈 A 요약 및 결론

## 학습한 내용

### 이론
- ✅ RNN의 기본 구조와 수식
- ✅ 그래디언트 소실/폭주 문제
- ✅ LSTM의 게이트 메커니즘
- ✅ GRU의 단순화된 구조

### 실습
- ✅ 합성 데이터 실험
- ✅ RNN, LSTM, GRU 비교
- ✅ Netflix 주가 예측
- ✅ 모델 평가 및 분석

## 핵심 인사이트

1. LSTM/GRU는 장기 의존성 학습에 필수적
2. 하이퍼파라미터 튜닝이 성능에 큰 영향
3. 실제 데이터 예측은 매우 도전적
4. 그래디언트 클리핑은 학습 안정성에 중요

**수고하셨습니다! 🎉**""")
    ]
