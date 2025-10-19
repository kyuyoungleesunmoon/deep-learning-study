"""
예제 4: 학습 루프
완전한 학습 파이프라인을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 간단한 신경망 정의
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_dataset(n_samples=1000, n_features=20, n_classes=3):
    """합성 데이터셋 생성"""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y

def train_epoch(model, train_loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
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
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return epoch_loss, accuracy

def validate(model, val_loader, criterion, device):
    """모델 검증"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def main():
    print("=" * 50)
    print("1. 데이터셋 준비")
    print("=" * 50)
    
    # 하이퍼파라미터
    input_size = 20
    hidden_size = 128
    num_classes = 3
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n사용 디바이스: {device}")
    
    # 데이터 생성
    X_train, y_train = create_dataset(1000, input_size, num_classes)
    X_val, y_val = create_dataset(200, input_size, num_classes)
    
    print(f"\n학습 데이터: {X_train.size()}")
    print(f"검증 데이터: {X_val.size()}")
    
    # Dataset과 DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"배치 개수 - 학습: {len(train_loader)}, 검증: {len(val_loader)}")
    
    print("\n" + "=" * 50)
    print("2. 모델 초기화")
    print("=" * 50)
    
    # 모델 생성
    model = Net(input_size, hidden_size, num_classes).to(device)
    print("\n모델 구조:")
    print(model)
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    print(f"\n손실 함수: {criterion}")
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"옵티마이저: {optimizer}")
    
    print("\n" + "=" * 50)
    print("3. 학습")
    print("=" * 50)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc")
    print("-" * 55)
    
    for epoch in range(num_epochs):
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 기록
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 출력
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | {val_loss:8.4f} | {val_acc:7.2f}%")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print("\n최고 검증 정확도: {:.2f}%".format(best_val_acc))
    
    print("\n" + "=" * 50)
    print("4. 다양한 옵티마이저")
    print("=" * 50)
    
    model = Net(input_size, hidden_size, num_classes).to(device)
    
    # SGD
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("\nSGD with momentum:")
    print(optimizer_sgd)
    
    # Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    print("\nAdam:")
    print(optimizer_adam)
    
    # RMSprop
    optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01)
    print("\nRMSprop:")
    print(optimizer_rmsprop)
    
    # AdamW
    optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    print("\nAdamW (with weight decay):")
    print(optimizer_adamw)
    
    print("\n" + "=" * 50)
    print("5. 학습률 스케줄러")
    print("=" * 50)
    
    model = Net(input_size, hidden_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # StepLR: 일정 스텝마다 학습률 감소
    scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print("\nStepLR (step_size=5, gamma=0.5):")
    
    for epoch in range(15):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}: lr = {current_lr:.6f}")
        scheduler_step.step()
    
    # ExponentialLR
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print("\nExponentialLR (gamma=0.9):")
    
    for epoch in range(10):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}: lr = {current_lr:.6f}")
        scheduler_exp.step()
    
    # ReduceLROnPlateau
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    print("\nReduceLROnPlateau (검증 손실이 개선되지 않으면 학습률 감소):")
    print("(실제 사용 시: scheduler.step(val_loss))")
    
    print("\n" + "=" * 50)
    print("6. 그래디언트 클리핑")
    print("=" * 50)
    
    model = Net(input_size, hidden_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 샘플 배치로 그래디언트 클리핑 테스트
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # 클리핑 전 그래디언트 norm
    total_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5
    
    print(f"\n클리핑 전 그래디언트 norm: {total_norm_before:.4f}")
    
    # 그래디언트 클리핑
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # 클리핑 후 그래디언트 norm
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5
    
    print(f"클리핑 후 그래디언트 norm: {total_norm_after:.4f}")
    print(f"최대 norm: {max_norm}")
    
    print("\n" + "=" * 50)
    print("7. Early Stopping")
    print("=" * 50)
    
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            
        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0
    
    # Early Stopping 사용 예제
    model = Net(input_size, hidden_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=3)
    
    print("\nEarly Stopping (patience=3) 시뮬레이션:")
    
    # 가상의 검증 손실
    val_losses = [0.5, 0.4, 0.35, 0.34, 0.36, 0.37, 0.38, 0.39, 0.40]
    
    for epoch, val_loss in enumerate(val_losses):
        print(f"Epoch {epoch+1}: val_loss = {val_loss:.2f}")
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            print(f"Early Stopping at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 50)
    print("8. 모델 저장 및 로드")
    print("=" * 50)
    
    model = Net(input_size, hidden_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 체크포인트 저장
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.5,
    }
    
    torch.save(checkpoint, '/tmp/checkpoint.pth')
    print("\n체크포인트 저장됨: /tmp/checkpoint.pth")
    
    # 체크포인트 로드
    loaded_checkpoint = torch.load('/tmp/checkpoint.pth')
    
    # 새 모델 생성 및 로드
    new_model = Net(input_size, hidden_size, num_classes).to(device)
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']
    loss = loaded_checkpoint['loss']
    
    print(f"체크포인트 로드됨: epoch={epoch}, loss={loss}")
    
    # 상태 검증
    new_model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, input_size).to(device)
        output = new_model(test_input)
        print(f"테스트 출력: {output.size()}")

if __name__ == "__main__":
    main()
