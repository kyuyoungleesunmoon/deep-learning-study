"""
예제 2: 자동 미분 (Autograd)
PyTorch의 자동 미분 기능을 다룹니다.
"""

import torch

def main():
    print("=" * 50)
    print("1. 기본 자동 미분")
    print("=" * 50)
    
    # requires_grad=True로 연산 추적
    x = torch.ones(2, 2, requires_grad=True)
    print("\nx (requires_grad=True):")
    print(x)
    
    # 연산 수행
    y = x + 2
    print("\ny = x + 2:")
    print(y)
    print("y의 grad_fn:", y.grad_fn)
    
    # 더 복잡한 연산
    z = y * y * 3
    print("\nz = y * y * 3:")
    print(z)
    print("z의 grad_fn:", z.grad_fn)
    
    out = z.mean()
    print("\nout = z.mean():")
    print(out)
    print("out의 grad_fn:", out.grad_fn)
    
    # 역전파
    out.backward()
    
    # 그래디언트 출력
    print("\n역전파 후 x.grad:")
    print(x.grad)
    
    # 수동 계산과 비교
    print("\n수동 계산:")
    print("out = mean(3 * (x + 2)^2)")
    print("dout/dx = d/dx[3/4 * (x + 2)^2] = 3/2 * (x + 2)")
    print("x = [[1, 1], [1, 1]]일 때, dout/dx = [[4.5, 4.5], [4.5, 4.5]]")
    
    print("\n" + "=" * 50)
    print("2. 그래디언트 누적")
    print("=" * 50)
    
    x = torch.ones(2, 2, requires_grad=True)
    
    # 첫 번째 연산
    y = x + 2
    z = y * y
    z.sum().backward()
    print("\n첫 번째 backward 후 x.grad:")
    print(x.grad)
    
    # 그래디언트를 초기화하지 않고 다시 backward
    y = x + 3
    z = y * y
    z.sum().backward()
    print("\n두 번째 backward 후 x.grad (누적됨):")
    print(x.grad)
    
    # 그래디언트 초기화
    x.grad.zero_()
    y = x * 2
    z = y * y
    z.sum().backward()
    print("\n그래디언트 초기화 후 backward:")
    print(x.grad)
    
    print("\n" + "=" * 50)
    print("3. 그래디언트 흐름 제어")
    print("=" * 50)
    
    x = torch.randn(3, requires_grad=True)
    print("\nx:", x)
    
    # 연산
    y = x * 2
    print("\ny = x * 2, requires_grad:", y.requires_grad)
    
    # with torch.no_grad(): 그래디언트 추적 중지
    with torch.no_grad():
        y = x * 2
        print("\nwith torch.no_grad():")
        print("y = x * 2, requires_grad:", y.requires_grad)
    
    # detach(): 그래디언트 추적에서 분리
    y = x * 2
    z = y.detach()
    print("\ndetach() 사용:")
    print("y.requires_grad:", y.requires_grad)
    print("z = y.detach(), z.requires_grad:", z.requires_grad)
    
    print("\n" + "=" * 50)
    print("4. 벡터-Jacobian 곱")
    print("=" * 50)
    
    # 벡터 출력에 대한 역전파
    x = torch.randn(3, requires_grad=True)
    print("\nx:", x)
    
    y = x * 2
    print("\ny = x * 2:", y)
    
    # 벡터에 대한 역전파 (gradient 인자 필요)
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)
    
    print("\nv:", v)
    print("x.grad:", x.grad)
    print("(각 요소에 v가 곱해진 그래디언트)")
    
    print("\n" + "=" * 50)
    print("5. 실제 예제: 선형 회귀")
    print("=" * 50)
    
    # 데이터 생성 (y = 3x + 2 + noise)
    torch.manual_seed(42)
    x_data = torch.randn(100, 1)
    y_data = 3 * x_data + 2 + torch.randn(100, 1) * 0.5
    
    # 파라미터 초기화
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    print(f"\n초기 파라미터: w = {w.item():.4f}, b = {b.item():.4f}")
    
    # 학습
    learning_rate = 0.01
    epochs = 100
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = x_data @ w + b
        
        # 손실 계산 (MSE)
        loss = ((y_pred - y_data) ** 2).mean()
        
        # Backward pass
        loss.backward()
        
        # 파라미터 업데이트 (그래디언트 추적 없이)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        # 그래디언트 초기화
        w.grad.zero_()
        b.grad.zero_()
        
        # 진행 상황 출력
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
    
    print(f"\n최종 파라미터: w = {w.item():.4f}, b = {b.item():.4f}")
    print("실제 값: w = 3.0000, b = 2.0000")
    
    print("\n" + "=" * 50)
    print("6. 그래디언트 체크포인팅")
    print("=" * 50)
    
    # 계산 그래프 확인
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2
    z = y ** 3
    
    print("\nx =", x.item())
    print("y = x^2 =", y.item())
    print("z = y^3 =", z.item())
    
    z.backward()
    
    print("\ndz/dx =", x.grad.item())
    print("수동 계산: dz/dx = dz/dy * dy/dx = 3y^2 * 2x = 3(x^2)^2 * 2x = 6x^5")
    print(f"= 6 * {x.item()}^5 = {6 * (x.item() ** 5)}")
    
    print("\n" + "=" * 50)
    print("7. requires_grad 동적 변경")
    print("=" * 50)
    
    x = torch.ones(2, 2)
    print("\n초기 상태:")
    print("x.requires_grad:", x.requires_grad)
    
    # requires_grad 활성화
    x.requires_grad_()
    print("\nx.requires_grad_() 후:")
    print("x.requires_grad:", x.requires_grad)
    
    y = x * 2
    print("\ny = x * 2:")
    print("y.requires_grad:", y.requires_grad)
    print("y.grad_fn:", y.grad_fn)
    
    # 일부 연산에서만 그래디언트 추적
    z = (x ** 2).sum()
    z.backward()
    print("\n역전파 후 x.grad:")
    print(x.grad)

if __name__ == "__main__":
    main()
