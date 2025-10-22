"""
Stage 3 시각화: 활성화 함수
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정
rcParams['font.family'] = 'DejaVu Sans'

# z 값 범위
z = np.linspace(-5, 5, 400)

# ========== 활성화 함수 정의 ==========
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Prevent overflow

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

# ========== 그림 1: 주요 활성화 함수 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Sigmoid
ax1 = axes[0, 0]
ax1.plot(z, sigmoid(z), 'b-', linewidth=2.5, label='Sigmoid')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('z', fontsize=13)
ax1.set_ylabel('σ(z)', fontsize=13)
ax1.set_title('Sigmoid Activation Function', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.text(2, 0.2, r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.text(2, 0.05, 'Range: (0, 1)', fontsize=11, color='red')

# ReLU
ax2 = axes[0, 1]
ax2.plot(z, relu(z), 'r-', linewidth=2.5, label='ReLU')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('z', fontsize=13)
ax2.set_ylabel('ReLU(z)', fontsize=13)
ax2.set_title('ReLU Activation Function', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.text(2, 1, r'$ReLU(z) = \max(0, z)$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
ax2.text(2, 0.3, r'Range: $[0, \infty)$', fontsize=11, color='red')

# Tanh
ax3 = axes[1, 0]
ax3.plot(z, tanh(z), 'g-', linewidth=2.5, label='Tanh')
ax3.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('z', fontsize=13)
ax3.set_ylabel('tanh(z)', fontsize=13)
ax3.set_title('Tanh Activation Function', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.text(2, -0.5, r'$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax3.text(2, -0.8, 'Range: (-1, 1)', fontsize=11, color='red')

# Comparison
ax4 = axes[1, 1]
ax4.plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid', alpha=0.8)
ax4.plot(z, relu(z), 'r-', linewidth=2, label='ReLU', alpha=0.8)
ax4.plot(z, tanh(z), 'g-', linewidth=2, label='Tanh', alpha=0.8)
ax4.plot(z, leaky_relu(z), 'm--', linewidth=2, label='Leaky ReLU', alpha=0.8)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('z', fontsize=13)
ax4.set_ylabel('f(z)', fontsize=13)
ax4.set_title('Comparison of Activation Functions', fontsize=15, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylim(-1.5, 5)

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage3_activation_functions.png', dpi=300, bbox_inches='tight')
print("✅ Activation functions visualization saved!")
plt.close()

# ========== 그림 2: 미분 시각화 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Sigmoid derivative
ax1 = axes[0, 0]
ax1.plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid', alpha=0.5)
ax1_twin = ax1.twinx()
ax1_twin.plot(z, sigmoid_derivative(z), 'r--', linewidth=2.5, label="Sigmoid'")
ax1.set_xlabel('z', fontsize=13)
ax1.set_ylabel('σ(z)', fontsize=13, color='b')
ax1_twin.set_ylabel("σ'(z)", fontsize=13, color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1.grid(True, alpha=0.3)
ax1.set_title("Sigmoid and Its Derivative", fontsize=15, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1_twin.legend(loc='upper right', fontsize=10)
ax1.text(0, 0.7, r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$", fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# ReLU derivative
ax2 = axes[0, 1]
ax2.plot(z, relu(z), 'r-', linewidth=2, label='ReLU', alpha=0.5)
ax2_twin = ax2.twinx()
ax2_twin.plot(z, relu_derivative(z), 'b--', linewidth=2.5, label="ReLU'")
ax2.set_xlabel('z', fontsize=13)
ax2.set_ylabel('ReLU(z)', fontsize=13, color='r')
ax2_twin.set_ylabel("ReLU'(z)", fontsize=13, color='b')
ax2.tick_params(axis='y', labelcolor='r')
ax2_twin.tick_params(axis='y', labelcolor='b')
ax2.grid(True, alpha=0.3)
ax2.set_title("ReLU and Its Derivative", fontsize=15, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2_twin.legend(loc='lower right', fontsize=10)
ax2_twin.set_ylim(-0.2, 1.5)

# Tanh derivative
ax3 = axes[1, 0]
ax3.plot(z, tanh(z), 'g-', linewidth=2, label='Tanh', alpha=0.5)
ax3_twin = ax3.twinx()
ax3_twin.plot(z, tanh_derivative(z), 'purple', linestyle='--', linewidth=2.5, label="Tanh'")
ax3.set_xlabel('z', fontsize=13)
ax3.set_ylabel('tanh(z)', fontsize=13, color='g')
ax3_twin.set_ylabel("tanh'(z)", fontsize=13, color='purple')
ax3.tick_params(axis='y', labelcolor='g')
ax3_twin.tick_params(axis='y', labelcolor='purple')
ax3.grid(True, alpha=0.3)
ax3.set_title("Tanh and Its Derivative", fontsize=15, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3_twin.legend(loc='upper right', fontsize=10)
ax3.text(0, 0.2, r"$\tanh'(z) = 1 - \tanh^2(z)$", fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Derivative comparison
ax4 = axes[1, 1]
ax4.plot(z, sigmoid_derivative(z), 'b-', linewidth=2, label="Sigmoid'", alpha=0.8)
ax4.plot(z, relu_derivative(z), 'r-', linewidth=2, label="ReLU'", alpha=0.8)
ax4.plot(z, tanh_derivative(z), 'g-', linewidth=2, label="Tanh'", alpha=0.8)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('z', fontsize=13)
ax4.set_ylabel("f'(z)", fontsize=13)
ax4.set_title('Comparison of Derivatives', fontsize=15, fontweight='bold')
ax4.legend(fontsize=11)
ax4.text(2, 0.5, 'ReLU: constant gradient\nwhen z > 0', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/runner/work/deep-learning-study/deep-learning-study/neural_network_math_series/stage3_derivatives.png', dpi=300, bbox_inches='tight')
print("✅ Derivatives visualization saved!")
plt.close()

print("\n🎉 All Stage 3 visualizations completed successfully!")
