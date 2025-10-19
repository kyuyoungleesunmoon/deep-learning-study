"""
예제 3: 신경망 구축
nn.Module을 사용한 신경망 구축 방법을 다룹니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    print("=" * 50)
    print("1. 간단한 신경망 (Fully Connected)")
    print("=" * 50)
    
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, output_size)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 모델 생성
    model = SimpleNet(784, 128, 10)
    print("\n모델 구조:")
    print(model)
    
    # 파라미터 개수 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n전체 파라미터 개수: {total_params:,}")
    print(f"학습 가능한 파라미터 개수: {trainable_params:,}")
    
    # 각 레이어의 파라미터 확인
    print("\n레이어별 파라미터:")
    for name, param in model.named_parameters():
        print(f"{name:15s}: {str(param.size()):20s} - {param.numel():,} 개")
    
    # 테스트 입력
    x = torch.randn(32, 784)  # 배치 크기 32
    output = model(x)
    print(f"\n입력 크기: {x.size()}")
    print(f"출력 크기: {output.size()}")
    
    print("\n" + "=" * 50)
    print("2. Sequential을 사용한 모델")
    print("=" * 50)
    
    # Sequential 모델
    model_seq = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    print("\nSequential 모델:")
    print(model_seq)
    
    # 테스트
    x = torch.randn(32, 784)
    output = model_seq(x)
    print(f"\n입력 크기: {x.size()}")
    print(f"출력 크기: {output.size()}")
    
    print("\n" + "=" * 50)
    print("3. 컨볼루션 신경망 (CNN)")
    print("=" * 50)
    
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # 컨볼루션 레이어
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 완전 연결 레이어
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            
            # Dropout
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            # 컨볼루션 블록 1
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            
            # 컨볼루션 블록 2
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            
            # Flatten
            x = x.view(-1, 64 * 7 * 7)
            
            # 완전 연결 레이어
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    # CNN 모델 생성
    cnn_model = CNN()
    print("\nCNN 모델:")
    print(cnn_model)
    
    # 테스트 입력 (배치 크기 4, 채널 1, 28x28 이미지)
    x = torch.randn(4, 1, 28, 28)
    output = cnn_model(x)
    print(f"\n입력 크기: {x.size()}")
    print(f"출력 크기: {output.size()}")
    
    print("\n" + "=" * 50)
    print("4. 다양한 활성화 함수")
    print("=" * 50)
    
    x = torch.linspace(-3, 3, 20)
    
    print("\n입력:", x[:5].tolist(), "...")
    
    # ReLU
    relu_output = F.relu(x)
    print("\nReLU 출력:", relu_output[:5].tolist(), "...")
    
    # Sigmoid
    sigmoid_output = torch.sigmoid(x)
    print("Sigmoid 출력:", sigmoid_output[:5].tolist(), "...")
    
    # Tanh
    tanh_output = torch.tanh(x)
    print("Tanh 출력:", tanh_output[:5].tolist(), "...")
    
    # LeakyReLU
    leaky_relu_output = F.leaky_relu(x, negative_slope=0.1)
    print("LeakyReLU 출력:", leaky_relu_output[:5].tolist(), "...")
    
    print("\n" + "=" * 50)
    print("5. 배치 정규화 (Batch Normalization)")
    print("=" * 50)
    
    class BNNet(nn.Module):
        def __init__(self):
            super(BNNet, self).__init__()
            self.fc1 = nn.Linear(100, 50)
            self.bn1 = nn.BatchNorm1d(50)
            self.fc2 = nn.Linear(50, 10)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
    
    bn_model = BNNet()
    print("\n배치 정규화 모델:")
    print(bn_model)
    
    # 테스트
    x = torch.randn(32, 100)
    bn_model.train()  # 학습 모드
    output_train = bn_model(x)
    
    bn_model.eval()  # 평가 모드
    output_eval = bn_model(x)
    
    print(f"\n학습 모드 출력 크기: {output_train.size()}")
    print(f"평가 모드 출력 크기: {output_eval.size()}")
    print("\n(배치 정규화는 학습/평가 모드에서 다르게 동작)")
    
    print("\n" + "=" * 50)
    print("6. 잔차 연결 (Residual Connection)")
    print("=" * 50)
    
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            
        def forward(self, x):
            residual = x
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            
            out += residual  # 잔차 연결
            out = F.relu(out)
            
            return out
    
    # 잔차 블록 생성
    res_block = ResidualBlock(64)
    print("\n잔차 블록:")
    print(res_block)
    
    # 테스트
    x = torch.randn(4, 64, 32, 32)
    output = res_block(x)
    print(f"\n입력 크기: {x.size()}")
    print(f"출력 크기: {output.size()}")
    print("(잔차 연결로 입출력 크기가 동일)")
    
    print("\n" + "=" * 50)
    print("7. 가중치 초기화")
    print("=" * 50)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # 모델에 초기화 적용
    model = SimpleNet(784, 128, 10)
    print("\n초기화 전 첫 번째 레이어 가중치 샘플:")
    print(model.fc1.weight[0, :5])
    
    model.apply(init_weights)
    print("\n초기화 후 첫 번째 레이어 가중치 샘플:")
    print(model.fc1.weight[0, :5])
    
    print("\n" + "=" * 50)
    print("8. 모델 요약")
    print("=" * 50)
    
    model = CNN()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n총 학습 가능한 파라미터 개수: {count_parameters(model):,}")
    
    # 레이어별 정보
    print("\n레이어별 상세 정보:")
    for name, module in model.named_children():
        if hasattr(module, 'weight'):
            print(f"{name:10s}: {str(type(module).__name__):20s} - Weight: {module.weight.size()}")
    
    print("\n" + "=" * 50)
    print("9. 모델 모드 (train vs eval)")
    print("=" * 50)
    
    model = BNNet()
    
    print("\n학습 모드:")
    model.train()
    print(f"model.training = {model.training}")
    
    print("\n평가 모드:")
    model.eval()
    print(f"model.training = {model.training}")
    
    print("\n(Dropout, BatchNorm 등은 학습/평가 모드에서 다르게 동작)")

if __name__ == "__main__":
    main()
