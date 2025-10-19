# PyTorch 딥러닝 문법 가이드

이 문서는 PyTorch를 사용한 딥러닝의 기본 문법을 단계별로 설명합니다.

## 목차
1. [PyTorch 기초](#1-pytorch-기초)
2. [텐서(Tensor) 기본 연산](#2-텐서tensor-기본-연산)
3. [자동 미분(Autograd)](#3-자동-미분autograd)
4. [신경망 구축](#4-신경망-구축)
5. [데이터 로딩](#5-데이터-로딩)
6. [학습 루프](#6-학습-루프)
7. [모델 저장 및 로드](#7-모델-저장-및-로드)

---

## 1. PyTorch 기초

### 1.1 PyTorch란?
PyTorch는 Facebook AI Research에서 개발한 오픈소스 머신러닝 라이브러리입니다. 
동적 계산 그래프를 사용하여 유연한 모델 구축이 가능합니다.

### 1.2 설치
```bash
pip install torch torchvision
```

### 1.3 기본 import
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

---

## 2. 텐서(Tensor) 기본 연산

### 2.1 텐서 생성
텐서는 PyTorch의 기본 데이터 구조입니다. NumPy의 ndarray와 유사하지만 GPU에서 연산할 수 있습니다.

```python
import torch

# 빈 텐서 생성
x = torch.empty(3, 4)
print("빈 텐서:\n", x)

# 랜덤 텐서 생성
x = torch.rand(3, 4)
print("\n랜덤 텐서:\n", x)

# 0으로 채워진 텐서
x = torch.zeros(3, 4, dtype=torch.long)
print("\n0 텐서:\n", x)

# 1로 채워진 텐서
x = torch.ones(3, 4)
print("\n1 텐서:\n", x)

# 직접 데이터로 텐서 생성
x = torch.tensor([5.5, 3, 2.1])
print("\n직접 생성한 텐서:\n", x)

# 기존 텐서를 기반으로 새 텐서 생성
x = torch.ones(3, 4)
y = torch.randn_like(x, dtype=torch.float)
print("\n기존 텐서 기반 새 텐서:\n", y)
```

### 2.2 텐서 연산
```python
import torch

# 덧셈
x = torch.rand(3, 4)
y = torch.rand(3, 4)

# 방법 1
z = x + y
print("덧셈 결과:\n", z)

# 방법 2
z = torch.add(x, y)

# 방법 3 (in-place)
y.add_(x)  # y에 x를 더하고 y를 변경

# 곱셈
z = x * y  # 요소별 곱셈
print("\n요소별 곱셈:\n", z)

# 행렬 곱셈
x = torch.rand(3, 4)
y = torch.rand(4, 5)
z = torch.mm(x, y)  # 또는 x @ y
print("\n행렬 곱셈:\n", z)
print("크기:", z.size())

# 텐서 슬라이싱
x = torch.rand(4, 4)
print("\n원본 텐서:\n", x)
print("첫 번째 행:", x[0, :])
print("첫 번째 열:", x[:, 0])
print("특정 요소:", x[1, 1].item())

# 텐서 크기 변경 (reshape)
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1은 자동으로 계산
print("\n크기 변경:")
print("원본:", x.size())
print("view(16):", y.size())
print("view(-1, 8):", z.size())
```

### 2.3 NumPy 변환
```python
import torch
import numpy as np

# Tensor to NumPy
a = torch.ones(5)
b = a.numpy()
print("Tensor:", a)
print("NumPy:", b)

# NumPy to Tensor
a = np.ones(5)
b = torch.from_numpy(a)
print("\nNumPy:", a)
print("Tensor:", b)
```

### 2.4 GPU 사용
```python
import torch

# CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)  # 직접 GPU에 생성
    y = torch.ones(5)
    y = y.to(device)  # CPU에서 GPU로 이동
    z = x + y
    print("GPU 연산 결과:", z)
    print("다시 CPU로:", z.to("cpu", torch.double))
else:
    print("CUDA를 사용할 수 없습니다.")
    device = torch.device("cpu")
```

---

## 3. 자동 미분(Autograd)

PyTorch의 `autograd` 패키지는 자동 미분을 제공하여 역전파를 쉽게 구현할 수 있습니다.

### 3.1 기본 개념
```python
import torch

# requires_grad=True로 설정하면 연산을 추적
x = torch.ones(2, 2, requires_grad=True)
print("x:\n", x)

# 연산 수행
y = x + 2
print("\ny:\n", y)
print("y의 grad_fn:", y.grad_fn)

z = y * y * 3
out = z.mean()
print("\nz:\n", z)
print("out:", out)

# 역전파
out.backward()

# 기울기(gradient) 출력
print("\ndout/dx:\n", x.grad)
```

### 3.2 그래디언트 제어
```python
import torch

x = torch.randn(3, requires_grad=True)
print("x:", x)

y = x * 2
i = 0
while y.data.norm() < 1000:
    y = y * 2
    i += 1

print("y:", y)
print("반복 횟수:", i)

# 벡터에 대한 역전파 (Jacobian 필요)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print("x.grad:", x.grad)

# gradient 계산 중지
print("\nrequires_grad:", x.requires_grad)
print("requires_grad:", (x ** 2).requires_grad)

with torch.no_grad():
    print("no_grad 내부:", (x ** 2).requires_grad)

# 또는 .detach() 사용
y = x.detach()
print("detach 후:", y.requires_grad)
```

---

## 4. 신경망 구축

### 4.1 nn.Module 기본
`nn.Module`은 모든 신경망의 기본 클래스입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 레이어 정의
        self.fc1 = nn.Linear(784, 128)  # 입력: 784, 출력: 128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Forward pass 정의
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 생성
model = SimpleNet()
print(model)

# 파라미터 확인
print("\n모델 파라미터:")
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")
```

### 4.2 컨볼루션 신경망 (CNN)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 컨볼루션 레이어
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 입력 채널: 1, 출력 채널: 32, 커널 크기: 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output

# 모델 생성 및 테스트
model = CNN()
input_tensor = torch.randn(1, 1, 28, 28)  # 배치 크기 1, 채널 1, 28x28 이미지
output = model(input_tensor)
print("출력 크기:", output.size())
```

### 4.3 순차적 모델 (Sequential)
```python
import torch
import torch.nn as nn

# Sequential을 사용한 간단한 모델
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

print(model)

# 입력 테스트
x = torch.randn(32, 784)  # 배치 크기 32
output = model(x)
print("출력 크기:", output.size())
```

---

## 5. 데이터 로딩

### 5.1 Dataset과 DataLoader
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 커스텀 Dataset 클래스
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# 샘플 데이터 생성
data = np.random.randn(1000, 20).astype(np.float32)
labels = np.random.randint(0, 2, 1000).astype(np.int64)

# Dataset 및 DataLoader 생성
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 데이터 로딩 예제
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape = {data.shape}, target shape = {target.shape}")
    if batch_idx == 2:  # 처음 3개 배치만 출력
        break
```

### 5.2 이미지 데이터 변환
```python
from torchvision import datasets, transforms

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST 데이터셋 예제 (실제로 다운로드하지 않음)
# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print("데이터 변환 파이프라인이 준비되었습니다.")
```

---

## 6. 학습 루프

### 6.1 기본 학습 루프
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 간단한 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 샘플 데이터 생성
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 2, (1000,))

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델, 손실 함수, 옵티마이저 초기화
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # 학습 모드
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        
        # 파라미터 업데이트
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    # 에폭 통계 출력
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
```

### 6.2 검증 루프
```python
def validate(model, val_loader, criterion):
    model.eval()  # 평가 모드
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for data, target in val_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

# 검증 데이터 생성
X_val = torch.randn(200, 20)
y_val = torch.randint(0, 2, (200,))
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 검증 실행
val_loss, val_acc = validate(model, val_loader, criterion)
```

### 6.3 학습률 스케줄러
```python
import torch.optim as optim

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습률 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 학습 루프에서 사용
for epoch in range(num_epochs):
    # ... 학습 코드 ...
    
    # 학습률 업데이트
    scheduler.step()
    
    # 현재 학습률 출력
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr:.6f}')
```

---

## 7. 모델 저장 및 로드

### 7.1 전체 모델 저장
```python
import torch

# 모델 저장
torch.save(model, 'model_complete.pth')

# 모델 로드
loaded_model = torch.load('model_complete.pth')
loaded_model.eval()
```

### 7.2 state_dict 저장 (권장)
```python
import torch

# state_dict 저장
torch.save(model.state_dict(), 'model_weights.pth')

# state_dict 로드
model = Net()  # 먼저 모델 구조를 정의
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

### 7.3 체크포인트 저장
```python
# 학습 중간 상태 저장
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 체크포인트에서 재개
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

## 추가 리소스

### 유용한 PyTorch 함수들
```python
# 모델 요약
from torchsummary import summary
summary(model, input_size=(1, 28, 28))  # 입력 크기 지정

# 그래디언트 클리핑
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 가중치 초기화
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# 모델을 특정 디바이스로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 배치 데이터도 같은 디바이스로
data, target = data.to(device), target.to(device)
```

---

## 마치며

이 문서는 PyTorch의 기본적인 문법과 사용법을 다룹니다. 실제 프로젝트에서는:
- 적절한 모델 아키텍처 선택
- 하이퍼파라미터 튜닝
- 데이터 전처리 및 증강
- 모델 평가 및 검증
- 과적합 방지 기법

등을 추가로 고려해야 합니다.

더 자세한 내용은 [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)를 참고하세요.
