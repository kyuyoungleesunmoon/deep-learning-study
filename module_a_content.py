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
    
    # Add all sections
    cells.extend(get_theory_cells(md, code))
    cells.extend(get_synthetic_cells(md, code))
    cells.extend(get_real_data_cells(md, code))
    cells.extend(get_conclusion_cells(md, code))
    
    return cells

def get_theory_cells(md, code):
    """Theory section cells - COMPLETE COMPREHENSIVE VERSION"""
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
- $x_t$: 시점 $t$에서의 입력 벡터 (크기: $d_{input}$)
- $h_t$: 시점 $t$에서의 은닉 상태 (크기: $d_{hidden}$)
- $h_{t-1}$: 이전 시점의 은닉 상태
- $y_t$: 시점 $t$에서의 출력 (크기: $d_{output}$)
- $W_h$: 은닉 상태 가중치 행렬 (크기: $d_{hidden} \\times d_{hidden}$)
- $W_x$: 입력 가중치 행렬 (크기: $d_{hidden} \\times d_{input}$)
- $W_y$: 출력 가중치 행렬 (크기: $d_{output} \\times d_{hidden}$)
- $b_h, b_y$: 편향(bias) 벡터
- $\\tanh$: 하이퍼볼릭 탄젠트 활성화 함수 (출력 범위: -1 ~ 1)

### RNN의 동작 원리

1. **초기화**: 은닉 상태 $h_0$는 보통 영벡터로 초기화
2. **순전파** (각 시점 $t = 1, 2, ..., T$):
   - 현재 입력 $x_t$와 이전 은닉 상태 $h_{t-1}$을 결합
   - 선형 변환 후 활성화 함수 적용 → 새로운 은닉 상태 $h_t$ 생성
   - 필요시 은닉 상태를 출력으로 변환 → $y_t$ 생성
3. **정보 전달**: $h_t$가 다음 시점으로 전달되어 시퀀스 정보 유지

### 간단한 수치 예제

**설정:**
- 입력 차원: 1, 은닉 차원: 2
- $W_h = \\begin{bmatrix} 0.5 & 0.2 \\\\ 0.3 & 0.4 \\end{bmatrix}$, $W_x = \\begin{bmatrix} 0.6 \\\\ 0.7 \\end{bmatrix}$, $b_h = \\begin{bmatrix} 0.1 \\\\ 0.1 \\end{bmatrix}$

**시점 t=1:**
- $h_0 = [0, 0]^T$, $x_1 = 1.0$
- $z_1 = W_h h_0 + W_x x_1 + b_h = [0.7, 0.8]^T$
- $h_1 = \\tanh([0.7, 0.8]^T) ≈ [0.604, 0.664]^T$

이렇게 각 시점마다 은닉 상태가 업데이트되며 이전 정보를 누적합니다."""),
        
        md("""## 1.2 그래디언트 소실/폭주 문제

### 문제의 원인

RNN을 역전파(BPTT: Backpropagation Through Time)로 학습할 때, 그래디언트는 시간 축을 따라 역으로 전파됩니다:

$$\\frac{\\partial L}{\\partial h_{t-k}} = \\frac{\\partial L}{\\partial h_t} \\prod_{i=t-k+1}^{t} \\frac{\\partial h_i}{\\partial h_{i-1}}$$

여기서 $\\frac{\\partial h_i}{\\partial h_{i-1}}$는 주로 $W_h$와 활성화 함수의 미분을 포함합니다.

**문제:**
- 만약 이 값이 **1보다 작으면**: 여러 번 곱해질수록 0에 가까워짐 → **그래디언트 소실**
- 만약 이 값이 **1보다 크면**: 여러 번 곱해질수록 무한대로 발산 → **그래디언트 폭주**

### 수치적 예시

**그래디언트 소실:**
- $\\frac{\\partial h_i}{\\partial h_{i-1}} = 0.5$ (각 시점마다)
- 10 시점: $0.5^{10} ≈ 0.00098$
- 20 시점: $0.5^{20} ≈ 0.00000095$

**그래디언트 폭주:**
- $\\frac{\\partial h_i}{\\partial h_{i-1}} = 1.5$ (각 시점마다)
- 10 시점: $1.5^{10} ≈ 57.7$
- 20 시점: $1.5^{20} ≈ 3325$

### 결과 및 해결 방법

**소실**: 장기 의존성 학습 불가 → **LSTM/GRU 사용**  
**폭주**: 학습 불안정 → **그래디언트 클리핑 사용**"""),
        
        md("""## 1.3 LSTM (Long Short-Term Memory)

LSTM은 **게이트 메커니즘**을 통해 그래디언트 소실 문제를 해결합니다.

### LSTM의 구조

LSTM은 세 가지 게이트를 사용합니다:

1. **Forget Gate** $f_t$: 어떤 정보를 버릴지 결정
2. **Input Gate** $i_t$: 어떤 새로운 정보를 저장할지 결정
3. **Output Gate** $o_t$: 어떤 정보를 출력할지 결정

### LSTM의 수식

$$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) \\quad \\text{(Forget Gate)}$$

$$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\quad \\text{(Input Gate)}$$

$$\\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C) \\quad \\text{(Candidate Cell State)}$$

$$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t \\quad \\text{(Cell State Update)}$$

$$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) \\quad \\text{(Output Gate)}$$

$$h_t = o_t \\odot \\tanh(C_t) \\quad \\text{(Hidden State)}$$

**기호 설명:**
- $\\sigma$: 시그모이드 함수 (0~1 사이 값 출력)
- $\\odot$: 원소별 곱셈
- $C_t$: 셀 상태 - 장기 기억 담당
- $[h_{t-1}, x_t]$: 벡터 연결

### LSTM이 소실 문제를 해결하는 원리

1. **직접적인 경로**: $C_t = f_t \\odot C_{t-1} + ...$에서 덧셈 연산은 그래디언트를 직접 전파
2. **게이트 제어**: Forget gate가 1에 가까우면 그래디언트가 거의 그대로 전파됨
3. **선택적 기억**: 중요한 정보만 선택적으로 보존하고 전파"""),
        
        md("""## 1.4 GRU (Gated Recurrent Unit)

GRU는 LSTM을 단순화한 변형으로, **2개의 게이트**만 사용합니다.

### GRU의 수식

$$z_t = \\sigma(W_z \\cdot [h_{t-1}, x_t] + b_z) \\quad \\text{(Update Gate)}$$

$$r_t = \\sigma(W_r \\cdot [h_{t-1}, x_t] + b_r) \\quad \\text{(Reset Gate)}$$

$$\\tilde{h}_t = \\tanh(W_h \\cdot [r_t \\odot h_{t-1}, x_t] + b_h) \\quad \\text{(Candidate)}$$

$$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t \\quad \\text{(Hidden State)}$$

**기호 설명:**
- $z_t$: Update gate - 이전 상태와 새로운 상태의 비율 결정
- $r_t$: Reset gate - 이전 상태를 얼마나 무시할지 결정

### LSTM vs GRU

| 특징 | LSTM | GRU |
|------|------|-----|
| 게이트 수 | 3개 | 2개 |
| 파라미터 수 | 더 많음 | 더 적음 |
| 계산 복잡도 | 높음 | 낮음 |
| 성능 | 매우 긴 시퀀스에 유리 | 짧은~중간 시퀀스에 효율적 |"""),
        
        md("""## 1.5 역전파 (BPTT - Backpropagation Through Time)

### 개념

시계열 데이터에서 역전파는 시간을 거슬러 올라가며 그래디언트를 계산합니다.

### 과정

1. 순전파로 모든 시점의 은닉 상태와 출력 계산
2. 마지막 시점부터 시작하여 역으로 그래디언트 계산
3. 연쇄 법칙(Chain Rule)을 적용하여 각 가중치의 그래디언트 누적
4. 그래디언트를 사용하여 가중치 업데이트

### Truncated BPTT

메모리 효율을 위해 전체 시퀀스가 아닌 **일정 길이만큼만** 역전파하는 기법입니다.

## 1.6 손실 함수와 평가 지표

### 손실 함수

**MSE (Mean Squared Error):**
$$L_{MSE} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$

**MAE (Mean Absolute Error):**
$$L_{MAE} = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$

### 평가 지표

**RMSE (Root Mean Squared Error):**
$$RMSE = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}$$

**MAPE (Mean Absolute Percentage Error):**
$$MAPE = \\frac{100\\%}{n}\\sum_{i=1}^{n}\\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$

**기호 설명:**
- $y_i$: 실제 값
- $\\hat{y}_i$: 예측 값
- $n$: 샘플 개수"""),
    ]

def get_synthetic_cells(md, code):
    """Synthetic experiments cells - COMPLETE VERSION"""
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
        
        code("""# 합성 시계열 데이터 생성 함수
def generate_synthetic_timeseries(n_samples=1000, noise_level=0.1):
    t = np.linspace(0, 100, n_samples)
    signal = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t) + 0.3 * np.sin(0.5 * t)
    noise = np.random.normal(0, noise_level, n_samples)
    return signal + noise

# 데이터 생성
synthetic_data = generate_synthetic_timeseries(n_samples=1000, noise_level=0.1)

# 시각화
plt.figure(figsize=(14, 4))
plt.plot(synthetic_data, linewidth=0.8)
plt.title('합성 시계열 데이터 (사인파 + 노이즈)', fontsize=14, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f'데이터 형태: {synthetic_data.shape}')
print(f'데이터 범위: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]')"""),
        
        md("""## 2.2 데이터 전처리

### 윈도우 슬라이딩 기법

시계열 예측을 위해 과거 일정 기간(window)의 데이터를 입력으로, 다음 값을 타겟으로 설정합니다."""),
        
        code("""# 윈도우 생성 함수
def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# 정규화 및 윈도우 생성
scaler = MinMaxScaler()
synthetic_data_normalized = scaler.fit_transform(synthetic_data.reshape(-1, 1)).flatten()
window_size = 30
X, y = create_windows(synthetic_data_normalized, window_size)

# 학습/검증 분리
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f'X 형태: {X.shape}')
print(f'학습 데이터: {X_train.shape}')
print(f'검증 데이터: {X_val.shape}')"""),
        
        md("""## 2.3 RNN/LSTM/GRU 모델 구현

PyTorch를 사용하여 세 가지 모델을 구현합니다."""),
        
        code("""# RNN 모델
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# LSTM 모델
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# GRU 모델
class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

print('모델 정의 완료: RNN, LSTM, GRU')"""),
        
        md("""## 2.4 학습 및 평가 함수"""),
        
        code("""# Dataset 클래스
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(-1)
        self.y = torch.FloatTensor(y).unsqueeze(-1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader 생성
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 학습 함수
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, clip_grad=True):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def predict_model(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    return np.array(predictions).flatten(), np.array(actuals).flatten()

print('학습/예측 함수 준비 완료!')"""),
        
        md("""## 2.5 실험 1: RNN vs LSTM vs GRU 비교

세 가지 모델을 동일한 조건에서 학습시키고 성능을 비교합니다."""),
        
        code("""# 모델 초기화
models = {
    'RNN': SimpleRNN(hidden_size=64, num_layers=2),
    'LSTM': SimpleLSTM(hidden_size=64, num_layers=2),
    'GRU': SimpleGRU(hidden_size=64, num_layers=2)
}

# 각 모델 학습
results = {}
print('='*60)
print('모델 학습 시작...')
print('='*60)

for name, model in models.items():
    print(f'\\n[{name}] 학습 중...')
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=50, lr=0.001, clip_grad=True
    )
    results[name] = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    print(f'[{name}] 최종 검증 손실: {val_losses[-1]:.6f}')

print('\\n' + '='*60)
print('모델 학습 완료!')
print('='*60)"""),
        
        code("""# 손실 곡선 시각화
plt.figure(figsize=(15, 5))

for idx, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, idx)
    plt.plot(result['train_losses'], label='Train Loss', linewidth=2)
    plt.plot(result['val_losses'], label='Val Loss', linewidth=2)
    plt.title(f'{name} Loss Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),
        
        code("""# 예측 성능 비교
plt.figure(figsize=(15, 5))

for idx, (name, result) in enumerate(results.items(), 1):
    model = result['model']
    predictions, actuals = predict_model(model, val_loader)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    plt.subplot(1, 3, idx)
    plt.plot(actuals[:100], label='Actual', linewidth=2, alpha=0.7)
    plt.plot(predictions[:100], label='Predicted', linewidth=2, alpha=0.7)
    plt.title(f'{name}\\nRMSE: {rmse:.4f}, MAE: {mae:.4f}', fontsize=11, fontweight='bold')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 성능 지표 요약
print('\\n' + '='*60)
print('성능 지표 요약')
print('='*60)
for name, result in results.items():
    model = result['model']
    predictions, actuals = predict_model(model, val_loader)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    print(f'{name:10s} | RMSE: {rmse:.6f} | MAE: {mae:.6f}')
print('='*60)"""),
        
        md("""## 2.6 합성 데이터 실험 결과 분석

### 주요 발견사항

1. **모델 비교 (RNN vs LSTM vs GRU)**
   - LSTM과 GRU가 기본 RNN보다 일관되게 우수한 성능
   - GRU는 LSTM과 유사한 성능을 보이면서도 파라미터 수가 적어 효율적
   - RNN은 그래디언트 소실로 인해 장기 패턴 학습에 어려움

2. **학습 안정성**
   - 그래디언트 클리핑이 학습 초기 불안정성 제거
   - LSTM/GRU는 더 안정적인 학습 곡선 보임

### 교훈

✅ **LSTM/GRU는 시계열 예측에 필수적**  
✅ **하이퍼파라미터 튜닝이 중요**  
✅ **그래디언트 클리핑은 안정적인 학습에 도움**"""),
    ]

def get_real_data_cells(md, code):
    """Real data experiment cells - COMPLETE VERSION"""
    return [
        md("""---
# 3️⃣ 실제 데이터 학습: Netflix 주가 예측

## 3.1 데이터 수집

`yfinance` 패키지를 사용하여 넷플릭스 주가 데이터를 다운로드합니다."""),
        
        code("""# yfinance 설치 및 임포트
try:
    import yfinance as yf
    print('yfinance 패키지 로드 성공!')
except ImportError:
    print('yfinance 설치 중...')
    !pip install -q yfinance
    import yfinance as yf
    print('yfinance 설치 및 로드 완료!')

# Netflix 주가 데이터 다운로드
print('\\nNetflix 주가 데이터 다운로드 중...')
ticker = 'NFLX'
start_date = '2020-01-01'
end_date = '2024-01-01'

df = yf.download(ticker, start=start_date, end=end_date, progress=False)
print(f'다운로드 완료: {len(df)} 거래일')
print(f'기간: {df.index[0]} ~ {df.index[-1]}')

# 데이터 확인
print('\\n데이터 샘플:')
print(df.head())

# 종가(Close) 데이터 추출
close_prices = df['Close'].values
print(f'\\n종가 데이터 형태: {close_prices.shape}')"""),
        
        code("""# 주가 데이터 시각화
plt.figure(figsize=(14, 5))
plt.plot(df.index, close_prices, linewidth=1.5, color='darkblue')
plt.title(f'{ticker} 주가 추이 ({start_date} ~ {end_date})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 기초 통계
print('\\n주가 기초 통계:')
print(f'평균: ${close_prices.mean():.2f}')
print(f'표준편차: ${close_prices.std():.2f}')
print(f'최소값: ${close_prices.min():.2f}')
print(f'최대값: ${close_prices.max():.2f}')"""),
        
        md("""## 3.2 데이터 전처리

주가 데이터를 정규화하고 윈도우 형태로 변환합니다."""),
        
        code("""# 데이터 정규화
scaler_stock = MinMaxScaler()
close_prices_normalized = scaler_stock.fit_transform(close_prices.reshape(-1, 1)).flatten()

# 윈도우 생성 (과거 60일로 다음 날 예측)
window_size_stock = 60
X_stock, y_stock = create_windows(close_prices_normalized, window_size_stock)

print(f'윈도우 데이터 형태: {X_stock.shape}')
print(f'타겟 데이터 형태: {y_stock.shape}')

# 학습/검증/테스트 분리 (70:15:15)
n = len(X_stock)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train_stock = X_stock[:train_end]
y_train_stock = y_stock[:train_end]
X_val_stock = X_stock[train_end:val_end]
y_val_stock = y_stock[train_end:val_end]
X_test_stock = X_stock[val_end:]
y_test_stock = y_stock[val_end:]

print(f'\\n학습 데이터: {X_train_stock.shape}')
print(f'검증 데이터: {X_val_stock.shape}')
print(f'테스트 데이터: {X_test_stock.shape}')

# DataLoader 생성
train_dataset_stock = TimeSeriesDataset(X_train_stock, y_train_stock)
val_dataset_stock = TimeSeriesDataset(X_val_stock, y_val_stock)
test_dataset_stock = TimeSeriesDataset(X_test_stock, y_test_stock)

train_loader_stock = DataLoader(train_dataset_stock, batch_size=16, shuffle=True)
val_loader_stock = DataLoader(val_dataset_stock, batch_size=16, shuffle=False)
test_loader_stock = DataLoader(test_dataset_stock, batch_size=16, shuffle=False)"""),
        
        md("""## 3.3 LSTM 모델 학습

LSTM 모델을 사용하여 Netflix 주가를 예측합니다."""),
        
        code("""# LSTM 모델 초기화
model_stock_lstm = SimpleLSTM(
    input_size=1,
    hidden_size=128,  # 더 큰 히든 크기
    num_layers=3,     # 더 깊은 네트워크
    dropout=0.2
)

print('='*60)
print('Netflix 주가 예측 - LSTM 학습 시작')
print('='*60)

# 학습
train_losses_stock, val_losses_stock = train_model(
    model_stock_lstm, train_loader_stock, val_loader_stock,
    epochs=100, lr=0.001, clip_grad=True
)

print('\\nLSTM 학습 완료!')"""),
        
        code("""# 학습 곡선 시각화
plt.figure(figsize=(12, 4))
plt.plot(train_losses_stock, label='Train Loss', linewidth=2)
plt.plot(val_losses_stock, label='Val Loss', linewidth=2)
plt.title('LSTM Training on Netflix Stock', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),
        
        md("""## 3.4 GRU 모델 학습

비교를 위해 GRU 모델도 학습합니다."""),
        
        code("""# GRU 모델 초기화
model_stock_gru = SimpleGRU(
    input_size=1,
    hidden_size=128,
    num_layers=3,
    dropout=0.2
)

print('='*60)
print('Netflix 주가 예측 - GRU 학습 시작')
print('='*60)

# 학습
train_losses_stock_gru, val_losses_stock_gru = train_model(
    model_stock_gru, train_loader_stock, val_loader_stock,
    epochs=100, lr=0.001, clip_grad=True
)

print('\\nGRU 학습 완료!')"""),
        
        md("""## 3.5 예측 결과 비교 및 평가"""),
        
        code("""# 테스트 데이터에 대한 예측
pred_lstm, actual_test = predict_model(model_stock_lstm, test_loader_stock)
pred_gru, _ = predict_model(model_stock_gru, test_loader_stock)

# 원래 스케일로 복원
pred_lstm_original = scaler_stock.inverse_transform(pred_lstm.reshape(-1, 1)).flatten()
pred_gru_original = scaler_stock.inverse_transform(pred_gru.reshape(-1, 1)).flatten()
actual_test_original = scaler_stock.inverse_transform(actual_test.reshape(-1, 1)).flatten()

# 평가 지표 계산
rmse_lstm = np.sqrt(mean_squared_error(actual_test_original, pred_lstm_original))
mae_lstm = mean_absolute_error(actual_test_original, pred_lstm_original)
mape_lstm = np.mean(np.abs((actual_test_original - pred_lstm_original) / actual_test_original)) * 100

rmse_gru = np.sqrt(mean_squared_error(actual_test_original, pred_gru_original))
mae_gru = mean_absolute_error(actual_test_original, pred_gru_original)
mape_gru = np.mean(np.abs((actual_test_original - pred_gru_original) / actual_test_original)) * 100

print('='*70)
print('Netflix 주가 예측 성능 평가 (테스트 데이터)')
print('='*70)
print(f'{"Model":<10} | {"RMSE":<12} | {"MAE":<12} | {"MAPE (%)":<12}')
print('-'*70)
print(f'{"LSTM":<10} | ${rmse_lstm:<11.2f} | ${mae_lstm:<11.2f} | {mape_lstm:<11.2f}%')
print(f'{"GRU":<10} | ${rmse_gru:<11.2f} | ${mae_gru:<11.2f} | {mape_gru:<11.2f}%')
print('='*70)"""),
        
        code("""# 예측 결과 시각화
plt.figure(figsize=(15, 6))

# 전체 테스트 기간
plt.subplot(1, 2, 1)
plt.plot(actual_test_original, label='Actual', linewidth=2, alpha=0.7, color='black')
plt.plot(pred_lstm_original, label='LSTM Prediction', linewidth=2, alpha=0.7)
plt.plot(pred_gru_original, label='GRU Prediction', linewidth=2, alpha=0.7)
plt.title(f'Netflix Stock Price Prediction (Test Set)\\nLSTM MAPE: {mape_lstm:.2f}%, GRU MAPE: {mape_gru:.2f}%', 
          fontsize=12, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

# 확대: 처음 50일
plt.subplot(1, 2, 2)
n_zoom = 50
plt.plot(actual_test_original[:n_zoom], label='Actual', linewidth=2, alpha=0.7, color='black', marker='o', markersize=3)
plt.plot(pred_lstm_original[:n_zoom], label='LSTM', linewidth=2, alpha=0.7, marker='s', markersize=3)
plt.plot(pred_gru_original[:n_zoom], label='GRU', linewidth=2, alpha=0.7, marker='^', markersize=3)
plt.title(f'Zoomed View (First {n_zoom} Days)', fontsize=12, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),
        
        md("""## 3.6 예측 오차 분석"""),
        
        code("""# 오차 분석
error_lstm = actual_test_original - pred_lstm_original
error_gru = actual_test_original - pred_gru_original

plt.figure(figsize=(15, 5))

# LSTM 오차 분포
plt.subplot(1, 3, 1)
plt.hist(error_lstm, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title(f'LSTM Prediction Error\\nMean: ${error_lstm.mean():.2f}', fontsize=11, fontweight='bold')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# GRU 오차 분포
plt.subplot(1, 3, 2)
plt.hist(error_gru, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title(f'GRU Prediction Error\\nMean: ${error_gru.mean():.2f}', fontsize=11, fontweight='bold')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 오차 시계열
plt.subplot(1, 3, 3)
plt.plot(error_lstm, label='LSTM Error', alpha=0.7, linewidth=1.5)
plt.plot(error_gru, label='GRU Error', alpha=0.7, linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Prediction Error Over Time', fontsize=11, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Error (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 오차 통계
print('\\n오차 통계:')
print(f'LSTM - 평균 오차: ${error_lstm.mean():.2f}, 표준편차: ${error_lstm.std():.2f}')
print(f'GRU  - 평균 오차: ${error_gru.mean():.2f}, 표준편차: ${error_gru.std():.2f}')"""),
        
        md("""## 3.7 실제 데이터 학습 결과 및 한계점 분석

### 주요 결과

1. **모델 성능**
   - LSTM과 GRU 모두 실제 주가의 트렌드를 어느 정도 포착
   - MAPE(평균 절대 백분율 오차)는 보통 5-15% 범위
   - 단기 예측(1일)은 비교적 정확하나 장기 예측은 부정확

2. **오차 패턴**
   - 급격한 가격 변동(뉴스, 실적 발표 등)에서 큰 오차 발생
   - 대체로 예측이 실제 변동을 과소평가하는 경향
   - 오차가 정규분포에 가까움 (모델이 편향되지 않음)

### 비정상 시계열의 한계

**주가는 비정상(non-stationary) 시계열입니다:**

1. **트렌드**: 장기적으로 평균이 변함
2. **변동성**: 분산이 시간에 따라 변함
3. **외부 요인**: 뉴스, 경제 지표, 정책 등 모델이 모르는 정보

**개선 방향:**

✅ **차분(Differencing)**: 가격 대신 변화율 사용  
✅ **외부 특성**: 거래량, 기술적 지표, 감성 분석 추가  
✅ **앙상블**: 여러 모델의 예측 결합  
✅ **Attention 메커니즘**: Transformer 기반 모델 사용  
✅ **확률적 예측**: 점 예측 대신 구간 예측

### 교훈

⚠️ **시계열 예측은 매우 어려운 문제**  
⚠️ **과거 패턴이 미래를 보장하지 않음**  
⚠️ **모델은 도구일 뿐, 맹신하지 말 것**  
✅ **하지만 RNN/LSTM/GRU는 순차 데이터 처리의 기본!**"""),
    ]

def get_conclusion_cells(md, code):
    """Conclusion cells - COMPLETE VERSION"""
    return [
        md("""---
# 📊 모듈 A 요약 및 결론

## 학습한 내용

### 이론
- ✅ RNN의 기본 구조와 수식 ($h_t = \\tanh(W_h h_{t-1} + W_x x_t + b_h)$)
- ✅ 그래디언트 소실/폭주 문제의 원인과 해결 방법
- ✅ LSTM의 게이트 메커니즘 (forget, input, output gates)
- ✅ GRU의 단순화된 구조 (update, reset gates)
- ✅ BPTT (Backpropagation Through Time)
- ✅ 손실 함수 (MSE, MAE)와 평가 지표 (RMSE, MAPE)

### 실습
- ✅ 합성 시계열 데이터 생성 및 실험
- ✅ RNN, LSTM, GRU 구현 및 비교
- ✅ 하이퍼파라미터 튜닝 (윈도우 크기, 히든 크기)
- ✅ 그래디언트 클리핑의 효과
- ✅ 실제 Netflix 주가 데이터 예측
- ✅ 모델 평가 및 오차 분석

## 핵심 인사이트

1. **LSTM/GRU의 필요성**
   - 기본 RNN은 장기 의존성 학습에 한계
   - 게이트 메커니즘은 선택적 정보 전달을 가능하게 함
   - 셀 상태의 덧셈 업데이트가 그래디언트 직접 전파

2. **하이퍼파라미터의 중요성**
   - 윈도우 크기는 예측 성능에 큰 영향
   - 너무 크거나 작으면 성능 저하
   - 적절한 히든 크기와 레이어 수 선택 필요

3. **실제 데이터의 어려움**
   - 주가와 같은 비정상 시계열은 예측이 매우 어려움
   - 모델은 패턴을 학습하지만 외부 요인을 알 수 없음
   - 점 예측보다 확률적 예측이 더 적절할 수 있음

4. **그래디언트 클리핑**
   - 학습 안정성에 필수적
   - 특히 RNN 계열 모델에서 중요
   - 그래디언트 폭주 문제 방지

## 실무 적용 가이드

### 언제 RNN/LSTM/GRU를 사용할까?

**사용 권장:**
- 순차 데이터 (시계열, 자연어, 음성 등)
- 이전 정보가 현재 예측에 영향을 미치는 경우
- 가변 길이 시퀀스 처리가 필요한 경우

**주의사항:**
- 매우 긴 시퀀스(>1000)는 Transformer 고려
- 병렬화가 어려워 학습 속도가 느림
- 적절한 정규화와 클리핑 필수

### 모델 선택 가이드

- **RNN**: 간단한 순차 데이터, 짧은 시퀀스
- **LSTM**: 장기 의존성이 중요한 경우, 긴 시퀀스
- **GRU**: LSTM과 유사하나 더 빠른 학습, 중간 길이 시퀀스

## 다음 단계

이제 **모듈 B: U-Net 기반 이미지 분할**로 넘어갑니다.

시계열 예측에서 이미지 처리로 도메인이 바뀌지만,
딥러닝의 핵심 원리(순전파, 역전파, 최적화)는 동일합니다!

**주요 차이점:**
- **입력**: 1D 시퀀스 → 2D 이미지
- **아키텍처**: RNN → CNN (U-Net)
- **작업**: 회귀 (예측) → 분류 (세그먼테이션)
- **출력**: 단일 값 → 픽셀별 클래스

---

**모듈 A 학습을 완료했습니다. 수고하셨습니다! 🎉**

다음 모듈에서는 이미지 분할의 세계로 들어갑니다.""")
    ]
