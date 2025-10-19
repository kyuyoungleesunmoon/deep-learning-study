"""
예제 1: 텐서 기초
PyTorch 텐서의 생성과 기본 연산을 다룹니다.
"""

import torch
import numpy as np

def main():
    print("=" * 50)
    print("1. 텐서 생성")
    print("=" * 50)
    
    # 빈 텐서
    x = torch.empty(3, 4)
    print("\n빈 텐서 (초기화되지 않음):")
    print(x)
    
    # 랜덤 텐서
    x = torch.rand(3, 4)
    print("\n랜덤 텐서 (0~1 균등 분포):")
    print(x)
    
    # 정규분포 랜덤 텐서
    x = torch.randn(3, 4)
    print("\n정규분포 랜덤 텐서:")
    print(x)
    
    # 0으로 채워진 텐서
    x = torch.zeros(3, 4, dtype=torch.long)
    print("\n0으로 채워진 텐서:")
    print(x)
    
    # 1로 채워진 텐서
    x = torch.ones(3, 4)
    print("\n1로 채워진 텐서:")
    print(x)
    
    # 직접 데이터로 생성
    x = torch.tensor([5.5, 3, 2.1, 7.8])
    print("\n직접 생성한 텐서:")
    print(x)
    
    # 특정 값으로 채워진 텐서
    x = torch.full((3, 4), 3.14)
    print("\n3.14로 채워진 텐서:")
    print(x)
    
    # 범위 텐서
    x = torch.arange(0, 10, 2)
    print("\n범위 텐서 (0부터 10까지 2씩 증가):")
    print(x)
    
    print("\n" + "=" * 50)
    print("2. 텐서 연산")
    print("=" * 50)
    
    # 덧셈
    x = torch.rand(3, 4)
    y = torch.rand(3, 4)
    
    print("\nx 텐서:")
    print(x)
    print("\ny 텐서:")
    print(y)
    
    # 여러 방법으로 덧셈
    print("\n덧셈 (x + y):")
    print(x + y)
    
    print("\n덧셈 (torch.add):")
    print(torch.add(x, y))
    
    # in-place 연산
    y_copy = y.clone()
    y_copy.add_(x)
    print("\nin-place 덧셈 (y.add_(x)):")
    print(y_copy)
    
    # 요소별 곱셈
    print("\n요소별 곱셈 (x * y):")
    print(x * y)
    
    # 행렬 곱셈
    x = torch.rand(3, 4)
    y = torch.rand(4, 5)
    z = torch.mm(x, y)
    print("\n행렬 곱셈 (3x4 @ 4x5 = 3x5):")
    print(z)
    print("결과 크기:", z.size())
    
    # @ 연산자 사용
    z2 = x @ y
    print("\n@ 연산자로 행렬 곱셈:")
    print(z2)
    
    print("\n" + "=" * 50)
    print("3. 텐서 인덱싱과 슬라이싱")
    print("=" * 50)
    
    x = torch.arange(16).reshape(4, 4)
    print("\n원본 텐서 (4x4):")
    print(x)
    
    print("\n첫 번째 행:")
    print(x[0, :])
    
    print("\n첫 번째 열:")
    print(x[:, 0])
    
    print("\n특정 요소 [1, 1]:")
    print(x[1, 1])
    print("스칼라 값으로:", x[1, 1].item())
    
    print("\n처음 2행, 처음 3열:")
    print(x[:2, :3])
    
    print("\n" + "=" * 50)
    print("4. 텐서 크기 변경")
    print("=" * 50)
    
    x = torch.randn(4, 4)
    print("\n원본 텐서 크기:", x.size())
    
    # view를 사용한 크기 변경
    y = x.view(16)
    print("view(16):", y.size())
    
    z = x.view(-1, 8)  # -1은 자동으로 계산
    print("view(-1, 8):", z.size())
    
    # reshape (더 유연함)
    w = x.reshape(2, 8)
    print("reshape(2, 8):", w.size())
    
    # flatten
    flat = x.flatten()
    print("flatten():", flat.size())
    
    print("\n" + "=" * 50)
    print("5. NumPy 변환")
    print("=" * 50)
    
    # Tensor to NumPy
    a = torch.ones(5)
    b = a.numpy()
    print("\nPyTorch 텐서:", a)
    print("NumPy 배열:", b)
    print("타입:", type(b))
    
    # NumPy to Tensor
    c = np.ones(5)
    d = torch.from_numpy(c)
    print("\nNumPy 배열:", c)
    print("PyTorch 텐서:", d)
    print("타입:", type(d))
    
    # 메모리 공유 확인
    a.add_(1)
    print("\n텐서 변경 후 (a.add_(1)):")
    print("텐서:", a)
    print("NumPy 배열:", b)
    print("(메모리를 공유하므로 둘 다 변경됨)")
    
    print("\n" + "=" * 50)
    print("6. GPU 사용")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print("\nCUDA 사용 가능!")
        device = torch.device("cuda")
        
        # GPU에 직접 생성
        x = torch.ones(5, device=device)
        print("GPU에 생성된 텐서:", x)
        
        # CPU에서 GPU로 이동
        y = torch.ones(5)
        y = y.to(device)
        print("CPU에서 GPU로 이동:", y)
        
        # GPU에서 연산
        z = x + y
        print("GPU 연산 결과:", z)
        
        # GPU에서 CPU로 이동
        z_cpu = z.to("cpu")
        print("다시 CPU로:", z_cpu)
    else:
        print("\nCUDA를 사용할 수 없습니다. CPU만 사용합니다.")
        device = torch.device("cpu")
        x = torch.ones(5, device=device)
        print("CPU에 생성된 텐서:", x)
    
    print("\n" + "=" * 50)
    print("7. 집계 함수")
    print("=" * 50)
    
    x = torch.randn(3, 4)
    print("\n텐서:")
    print(x)
    
    print("\n합계:", x.sum().item())
    print("평균:", x.mean().item())
    print("최대값:", x.max().item())
    print("최소값:", x.min().item())
    print("표준편차:", x.std().item())
    
    print("\n행별 합계:")
    print(x.sum(dim=1))
    
    print("\n열별 평균:")
    print(x.mean(dim=0))
    
    # argmax, argmin
    print("\n최대값의 인덱스:", x.argmax().item())
    print("행별 최대값의 인덱스:")
    print(x.argmax(dim=1))

if __name__ == "__main__":
    main()
