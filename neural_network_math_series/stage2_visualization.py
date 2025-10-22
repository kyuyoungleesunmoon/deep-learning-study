"""
Stage 2 시각화: 퍼셉트론과 가중합
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== 1. 퍼셉트론 구조 다이어그램 ==========
ax1 = axes[0, 0]
ax1.axis('off')

# 입력 노드
input_x = 0.2
input_y_positions = [0.7, 0.5, 0.3]
input_labels = ['x₁', 'x₂', 'x₃']

for i, (y, label) in enumerate(zip(input_y_positions, input_labels)):
    circle = plt.Circle((input_x, y), 0.04, color='lightblue', ec='black', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(input_x - 0.12, y, label, fontsize=14, ha='right', fontweight='bold')

# 가중합 노드
sum_x = 0.5
sum_y = 0.5
circle = plt.Circle((sum_x, sum_y), 0.06, color='yellow', ec='black', linewidth=2)
ax1.add_patch(circle)
ax1.text(sum_x, sum_y, 'Σ', fontsize=18, ha='center', va='center', fontweight='bold')

# 활성화 함수 노드
act_x = 0.7
act_y = 0.5
circle = plt.Circle((act_x, act_y), 0.05, color='lightcoral', ec='black', linewidth=2)
ax1.add_patch(circle)
ax1.text(act_x, act_y, 'f', fontsize=14, ha='center', va='center', fontweight='bold')

# 출력 노드
output_x = 0.9
output_y = 0.5
circle = plt.Circle((output_x, output_y), 0.04, color='lightgreen', ec='black', linewidth=2)
ax1.add_patch(circle)
ax1.text(output_x + 0.08, output_y, 'y', fontsize=14, ha='left', fontweight='bold')

# 연결선 (가중치)
weights = ['w₁', 'w₂', 'w₃']
for i, (y, w_label) in enumerate(zip(input_y_positions, weights)):
    ax1.plot([input_x + 0.04, sum_x - 0.06], [y, sum_y], 'gray', linewidth=2)
    mid_x = (input_x + sum_x) / 2
    mid_y = (y + sum_y) / 2
    ax1.text(mid_x, mid_y + 0.02, w_label, fontsize=10, ha='center', color='red', fontweight='bold')

# 편향
ax1.plot([sum_x, sum_x], [sum_y - 0.15, sum_y - 0.06], 'gray', linewidth=2)
ax1.text(sum_x, sum_y - 0.18, 'b', fontsize=12, ha='center', color='red', fontweight='bold')

# 연결선 (활성화)
ax1.arrow(sum_x + 0.06, sum_y, act_x - sum_x - 0.11, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')
ax1.text((sum_x + act_x) / 2, sum_y + 0.05, 'z', fontsize=12, ha='center', color='blue', fontweight='bold')

# 연결선 (출력)
ax1.arrow(act_x + 0.05, act_y, output_x - act_x - 0.09, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('Perceptron Structure', fontsize=16, fontweight='bold', pad=20)

# 수식 표시
ax1.text(0.5, 0.1, r'z = w₁x₁ + w₂x₂ + w₃x₃ + b', fontsize=13, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.text(0.5, 0.02, r'y = f(z)', fontsize=13, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ========== 2. 결정 경계 시각화 (AND 게이트) ==========
ax2 = axes[0, 1]

# AND 게이트 파라미터
w1, w2 = 0.5, 0.5
b = -0.7

# 데이터 포인트
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# 결정 경계: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
x1_line = np.linspace(-0.5, 1.5, 100)
x2_line = -(w1 * x1_line + b) / w2

# 배경 색칠
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
Z = w1 * xx + w2 * yy + b
ax2.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['lightcoral', 'lightblue'], alpha=0.3)

# 결정 경계선
ax2.plot(x1_line, x2_line, 'k-', linewidth=3, label='Decision Boundary')

# 데이터 포인트
for i, (x, y) in enumerate(X_and):
    color = 'blue' if y_and[i] == 1 else 'red'
    marker = 'o' if y_and[i] == 1 else 'x'
    label_text = f'Class {y_and[i]}' if (i == 0 or i == 3) else None
    ax2.scatter(x, y, c=color, marker=marker, s=200, edgecolors='black', linewidth=2, label=label_text)

ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel('x₁', fontsize=14)
ax2.set_ylabel('x₂', fontsize=14)
ax2.set_title('AND Gate Decision Boundary', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 1.3, f'w₁={w1}, w₂={w2}, b={b}', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ========== 3. 가중치 변화에 따른 결정 경계 ==========
ax3 = axes[1, 0]

# 여러 가중치 설정
weights_list = [(1, 1, -0.5), (1, 0.5, -0.3), (0.5, 1, -0.3)]
colors = ['red', 'blue', 'green']
labels = ['w=[1,1], b=-0.5', 'w=[1,0.5], b=-0.3', 'w=[0.5,1], b=-0.3']

for (w1, w2, b), color, label in zip(weights_list, colors, labels):
    x1_line = np.linspace(-0.5, 2, 100)
    x2_line = -(w1 * x1_line + b) / w2
    ax3.plot(x1_line, x2_line, linewidth=2, color=color, label=label)

ax3.set_xlim(-0.5, 2)
ax3.set_ylim(-0.5, 2)
ax3.set_xlabel('x₁', fontsize=14)
ax3.set_ylabel('x₂', fontsize=14)
ax3.set_title('Effect of Different Weights on Decision Boundary', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ========== 4. 편향 변화에 따른 결정 경계 ==========
ax4 = axes[1, 1]

# 고정된 가중치, 다양한 편향
w1, w2 = 1, 1
biases = [-1, -0.5, 0, 0.5]
colors = ['red', 'orange', 'blue', 'green']

for b, color in zip(biases, colors):
    x1_line = np.linspace(-0.5, 2, 100)
    x2_line = -(w1 * x1_line + b) / w2
    ax4.plot(x1_line, x2_line, linewidth=2, color=color, label=f'b={b}')

ax4.set_xlim(-0.5, 2)
ax4.set_ylim(-0.5, 2)
ax4.set_xlabel('x₁', fontsize=14)
ax4.set_ylabel('x₂', fontsize=14)
ax4.set_title('Effect of Bias on Decision Boundary', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.text(1, 1.7, 'w₁=1, w₂=1 (fixed)', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 화살표로 편향 증가 방향 표시
ax4.annotate('Bias increasing →', xy=(0.5, 0.3), xytext=(0.2, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'),
            fontsize=11, color='purple', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage2_perceptron_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Perceptron visualization saved!")
plt.close()

# ========== 추가: 3D 가중합 시각화 ==========
fig = plt.figure(figsize=(14, 6))

# ========== 1. 3D 가중합 평면 ==========
ax1 = fig.add_subplot(121, projection='3d')

# 파라미터
w1, w2, b = 0.8, 0.6, -1.0

# 그리드 생성
x1 = np.linspace(-2, 2, 30)
x2 = np.linspace(-2, 2, 30)
X1, X2 = np.meshgrid(x1, x2)
Z = w1 * X1 + w2 * X2 + b

# 평면 그리기
surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7, edgecolor='none')

# z=0 평면 (결정 경계)
ax1.plot_surface(X1, X2, np.zeros_like(Z), alpha=0.3, color='red')

ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_zlabel('z = w₁x₁ + w₂x₂ + b', fontsize=12)
ax1.set_title('3D Weighted Sum Surface', fontsize=14, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# ========== 2. 등고선 플롯 ==========
ax2 = fig.add_subplot(122)

# 등고선
contour = ax2.contour(X1, X2, Z, levels=15, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)

# z=0 결정 경계 강조
contour_zero = ax2.contour(X1, X2, Z, levels=[0], colors='red', linewidths=3)
ax2.clabel(contour_zero, inline=True, fontsize=10)

ax2.set_xlabel('x₁', fontsize=12)
ax2.set_ylabel('x₂', fontsize=12)
ax2.set_title('Contour Plot of Weighted Sum', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0, 1.7, f'w₁={w1}, w₂={w2}, b={b}', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage2_weighted_sum_3d.png', dpi=300, bbox_inches='tight')
print("✅ 3D weighted sum visualization saved!")
plt.close()

print("\n🎉 All Stage 2 visualizations completed successfully!")
