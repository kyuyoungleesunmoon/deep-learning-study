"""
예제 5: 완전한 예제
MNIST 스타일의 이미지 분류 작업을 수행하는 완전한 예제입니다.
(실제 MNIST 데이터 대신 합성 데이터 사용)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ========== 1. 데이터셋 정의 ==========
class SyntheticMNIST(Dataset):
    """합성 MNIST 스타일 데이터셋"""
    def __init__(self, num_samples=1000, train=True):
        self.num_samples = num_samples
        self.train = train
        
        # 합성 이미지와 레이블 생성
        np.random.seed(42 if train else 123)
        self.images = torch.randn(num_samples, 1, 28, 28)
        self.labels = torch.randint(0, 10, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ========== 2. CNN 모델 정의 ==========
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 컨볼루션 레이어
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 배치 정규화
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 풀링
        self.pool = nn.MaxPool2d(2, 2)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ========== 3. 학습 함수 ==========
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

# ========== 4. 테스트 함수 ==========
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

# ========== 5. 메인 함수 ==========
def main():
    print("=" * 60)
    print("CNN을 사용한 이미지 분류 - 완전한 예제")
    print("=" * 60)
    
    # ========== 설정 ==========
    print("\n1. 하이퍼파라미터 설정")
    print("-" * 60)
    
    # 하이퍼파라미터
    batch_size = 64
    test_batch_size = 100
    epochs = 10
    learning_rate = 0.001
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"배치 크기: {batch_size}")
    print(f"테스트 배치 크기: {test_batch_size}")
    print(f"에폭 수: {epochs}")
    print(f"학습률: {learning_rate}")
    print(f"디바이스: {device}")
    
    # ========== 데이터 로딩 ==========
    print("\n2. 데이터 로딩")
    print("-" * 60)
    
    # 데이터셋 생성
    train_dataset = SyntheticMNIST(num_samples=5000, train=True)
    test_dataset = SyntheticMNIST(num_samples=1000, train=False)
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"학습 샘플 수: {len(train_dataset)}")
    print(f"테스트 샘플 수: {len(test_dataset)}")
    print(f"학습 배치 수: {len(train_loader)}")
    print(f"테스트 배치 수: {len(test_loader)}")
    
    # ========== 모델 생성 ==========
    print("\n3. 모델 생성")
    print("-" * 60)
    
    model = CNN().to(device)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"전체 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    print("\n모델 구조:")
    print(model)
    
    # ========== 손실 함수와 옵티마이저 ==========
    print("\n4. 손실 함수와 옵티마이저")
    print("-" * 60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    print(f"손실 함수: {criterion}")
    print(f"옵티마이저: Adam")
    print(f"학습률 스케줄러: StepLR (step_size=3, gamma=0.7)")
    
    # ========== 학습 ==========
    print("\n5. 학습 시작")
    print("-" * 60)
    
    best_test_acc = 0
    
    print("\n{:^5} | {:^12} | {:^10} | {:^12} | {:^10} | {:^8}".format(
        "Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc", "LR"
    ))
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        # 학습
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        
        # 테스트
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        # 학습률 업데이트
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 결과 출력
        print("{:5d} | {:12.4f} | {:9.2f}% | {:12.4f} | {:9.2f}% | {:.6f}".format(
            epoch, train_loss, train_acc, test_loss, test_acc, current_lr
        ))
        
        # 최고 성능 모델 기록
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    print("-" * 75)
    print(f"최고 테스트 정확도: {best_test_acc:.2f}%")
    
    # ========== 모델 저장 ==========
    print("\n6. 모델 저장")
    print("-" * 60)
    
    # 모델 저장
    model_path = '/tmp/cnn_model.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_acc': best_test_acc,
    }, model_path)
    
    print(f"모델이 저장되었습니다: {model_path}")
    
    # ========== 모델 로드 및 평가 ==========
    print("\n7. 저장된 모델 로드 및 평가")
    print("-" * 60)
    
    # 새 모델 생성
    loaded_model = CNN().to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"모델 로드 완료 (저장된 에폭: {checkpoint['epoch']})")
    print(f"저장된 최고 정확도: {checkpoint['best_test_acc']:.2f}%")
    
    # 로드된 모델로 테스트
    test_loss, test_acc = test(loaded_model, device, test_loader, criterion)
    print(f"로드된 모델 테스트 정확도: {test_acc:.2f}%")
    
    # ========== 추론 예제 ==========
    print("\n8. 추론 예제")
    print("-" * 60)
    
    loaded_model.eval()
    
    # 테스트 데이터에서 몇 개 샘플 가져오기
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 첫 5개 샘플 추론
    images_sample = images[:5].to(device)
    labels_sample = labels[:5]
    
    with torch.no_grad():
        outputs = loaded_model(images_sample)
        _, predicted = torch.max(outputs, 1)
    
    print("\n샘플 추론 결과:")
    for i in range(5):
        print(f"샘플 {i+1}: 실제 레이블 = {labels_sample[i].item()}, "
              f"예측 레이블 = {predicted[i].item()}, "
              f"일치 = {'O' if labels_sample[i].item() == predicted[i].item() else 'X'}")
    
    # 확률 출력
    print("\n첫 번째 샘플의 클래스별 확률:")
    probs = F.softmax(outputs[0], dim=0)
    for i, prob in enumerate(probs):
        print(f"클래스 {i}: {prob.item():.4f} ({prob.item() * 100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("학습 및 평가 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
