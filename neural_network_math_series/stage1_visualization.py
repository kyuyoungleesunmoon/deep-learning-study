"""
Stage 1 시각화: 벡터와 선형 변환
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'

# 그림 크기 설정
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== 1. 벡터 표현 ==========
ax1 = axes[0, 0]
# 원점
origin = [0, 0]
# 벡터 정의
v1 = [3, 4]
v2 = [2, -1]

# 벡터 그리기
ax1.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='blue', width=0.006, label='v1 = [3, 4]')
ax1.quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, label='v2 = [2, -1]')

# 격자 및 축 설정
ax1.set_xlim(-1, 5)
ax1.set_ylim(-2, 5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Vectors in 2D Space', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)

# 벡터 크기 표시
magnitude_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
ax1.text(1.5, 2.5, f'||v1|| = {magnitude_v1:.2f}', fontsize=10, color='blue')

# ========== 2. 벡터 덧셈 ==========
ax2 = axes[0, 1]
a = [2, 3]
b = [1, 4]
c = [a[0] + b[0], a[1] + b[1]]  # 벡터 합

ax2.quiver(*origin, *a, angles='xy', scale_units='xy', scale=1, color='blue', width=0.006, label='a = [2, 3]')
ax2.quiver(*a, *b, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, label='b = [1, 4]')
ax2.quiver(*origin, *c, angles='xy', scale_units='xy', scale=1, color='green', width=0.008, label='a + b = [3, 7]')

ax2.set_xlim(-1, 5)
ax2.set_ylim(-1, 8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Vector Addition', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)

# ========== 3. 내적 시각화 ==========
ax3 = axes[1, 0]
v_a = np.array([4, 2])
v_b = np.array([3, 3])
dot_product = np.dot(v_a, v_b)

ax3.quiver(*origin, *v_a, angles='xy', scale_units='xy', scale=1, color='blue', width=0.006, label=f'a = {v_a}')
ax3.quiver(*origin, *v_b, angles='xy', scale_units='xy', scale=1, color='red', width=0.006, label=f'b = {v_b}')

# 각도 계산
cos_angle = dot_product / (np.linalg.norm(v_a) * np.linalg.norm(v_b))
angle = np.arccos(cos_angle) * 180 / np.pi

ax3.set_xlim(-1, 5)
ax3.set_ylim(-1, 5)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Dot Product', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.text(2, 4, f'a · b = {dot_product}', fontsize=12, color='purple', fontweight='bold')
ax3.text(2, 3.5, f'angle = {angle:.1f}°', fontsize=10, color='purple')

# ========== 4. 선형 변환 (회전) ==========
ax4 = axes[1, 1]
# 45도 회전 행렬
theta = np.pi / 4  # 45도
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# 원본 벡터들
original_vectors = np.array([[1, 0], [0, 1], [1, 1]]).T
# 변환된 벡터들
transformed_vectors = R @ original_vectors

# 원본 벡터 그리기
for i in range(original_vectors.shape[1]):
    ax4.quiver(*origin, *original_vectors[:, i], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.005, alpha=0.5)

# 변환된 벡터 그리기
for i in range(transformed_vectors.shape[1]):
    ax4.quiver(*origin, *transformed_vectors[:, i], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.006)

ax4.set_xlim(-0.5, 1.5)
ax4.set_ylim(-0.5, 1.5)
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('Linear Transformation (45° Rotation)', fontsize=14, fontweight='bold')
ax4.text(0.1, 1.3, 'Blue: Original', fontsize=10, color='blue')
ax4.text(0.1, 1.2, 'Red: Rotated', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage1_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to stage1_visualization.png")
plt.close()

# ========== 행렬 연산 시각화 ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ========== 1. 행렬-벡터 곱 시각화 ==========
ax1 = axes[0]
A = np.array([[2, 3], [4, 5]])
x = np.array([1, 2])
y = A @ x

# 입력 벡터
ax1.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.008, label=f'x = {x}')
# 출력 벡터
ax1.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.008, label=f'Ax = {y}')

ax1.set_xlim(-1, 10)
ax1.set_ylim(-1, 15)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Matrix-Vector Multiplication', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.text(4, 12, f'A = {A.tolist()}', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========== 2. 행렬 곱셈 결과 히트맵 ==========
ax2 = axes[1]
A_mat = np.array([[1, 2], [3, 4]])
B_mat = np.array([[5, 6], [7, 8]])
C_mat = A_mat @ B_mat

im = ax2.imshow(C_mat, cmap='YlOrRd', aspect='auto')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['col 0', 'col 1'])
ax2.set_yticklabels(['row 0', 'row 1'])

# 값 표시
for i in range(2):
    for j in range(2):
        text = ax2.text(j, i, f'{C_mat[i, j]}',
                       ha="center", va="center", color="black", fontsize=14, fontweight='bold')

ax2.set_title('Matrix Multiplication Result C = A × B', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2)

# 수식 표시
ax2.text(0.5, -0.3, f'A = {A_mat.tolist()}, B = {B_mat.tolist()}', 
         ha='center', transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ========== 3. 신경망 선형 변환 ==========
ax3 = axes[2]
# 3개 입력 -> 2개 출력
W = np.array([[0.5, 0.3, 0.2], [0.7, 0.4, 0.6]])
x_input = np.array([1.0, 2.0, 3.0])
b = np.array([0.1, 0.2])
z_output = W @ x_input + b

# 신경망 구조 시각화
input_layer_x = 0.2
hidden_layer_x = 0.8

# 입력 노드
for i in range(3):
    y = 0.2 + i * 0.3
    circle = plt.Circle((input_layer_x, y), 0.05, color='lightblue', ec='black', linewidth=2)
    ax3.add_patch(circle)
    ax3.text(input_layer_x - 0.15, y, f'x{i+1}={x_input[i]}', fontsize=10, ha='right')

# 출력 노드
for i in range(2):
    y = 0.35 + i * 0.3
    circle = plt.Circle((hidden_layer_x, y), 0.05, color='lightcoral', ec='black', linewidth=2)
    ax3.add_patch(circle)
    ax3.text(hidden_layer_x + 0.15, y, f'z{i+1}={z_output[i]:.2f}', fontsize=10, ha='left')

# 연결선 (가중치)
for i in range(3):
    for j in range(2):
        y_in = 0.2 + i * 0.3
        y_out = 0.35 + j * 0.3
        ax3.plot([input_layer_x + 0.05, hidden_layer_x - 0.05], [y_in, y_out], 
                'gray', alpha=0.3, linewidth=1)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('Neural Network Linear Transformation', fontsize=14, fontweight='bold')
ax3.text(0.5, 0.05, 'z = Wx + b', fontsize=12, ha='center', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage1_matrix_operations.png', dpi=300, bbox_inches='tight')
print("✅ Matrix operations visualization saved to stage1_matrix_operations.png")
plt.close()

print("\n🎉 All Stage 1 visualizations completed successfully!")
