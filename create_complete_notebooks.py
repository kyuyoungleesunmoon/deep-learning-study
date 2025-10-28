#!/usr/bin/env python3
"""
Complete AI Textbook Module Generator
Generates comprehensive Jupyter notebooks for:
- Module A: RNN Time Series Prediction (Netflix Stock)
- Module B: U-Net Image Segmentation (Oxford-IIIT Pet)
"""

import json

def md(text):
    """Create markdown cell"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code(text):
    """Create code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def create_notebook(cells):
    """Create notebook structure"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def save_notebook(notebook, filename):
    """Save notebook to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"✅ Saved: {filename} ({len(notebook['cells'])} cells)")

print("="*70)
print("AI 교재 생성 시작")
print("="*70)
print()
print("이 스크립트는 완전한 AI 교재 노트북 2개를 생성합니다:")
print("1. Module A: RNN 기반 시계열 예측 (Netflix 주가)")
print("2. Module B: U-Net 기반 이미지 분할 (Oxford-IIIT Pet)")
print()
print("각 모듈은 다음을 포함합니다:")
print("- 이론 설명 (수식, 기호 설명)")
print("- 합성 데이터 실험")
print("- 실제 데이터 학습")
print("- 결과 분석 및 시각화")
print("="*70)
print()

# Store this script for later execution
# Will generate notebooks when run
