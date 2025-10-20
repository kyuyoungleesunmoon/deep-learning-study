#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 평가와 하이퍼파라미터 튜닝 - 실습 예제

이 파일은 model_evaluation_theory.md의 이론을 실제 코드로 구현합니다.
각 섹션은 독립적으로 실행 가능하며, 단계별로 학습할 수 있습니다.

실행 방법:
    python 06_model_evaluation.py

학습 시간: 약 60-90분
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    learning_curve, validation_curve, GridSearchCV,
    RandomizedSearchCV, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, auc,
    precision_recall_curve, ConfusionMatrixDisplay
)
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')


def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


# ============================================================================
# 섹션 1: 위스콘신 유방암 데이터셋 로드 및 탐색
# ============================================================================

def section1_load_data():
    """
    위스콘신 유방암 데이터셋을 로드하고 기본 정보를 확인합니다.
    
    학습 목표:
    - 실제 데이터셋의 구조 이해
    - 특성과 타겟의 관계 파악
    - 클래스 분포 확인
    """
    print_section("섹션 1: 위스콘신 유방암 데이터셋")
    
    # 데이터 로드
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print("📊 데이터셋 기본 정보")
    print(f"샘플 수: {X.shape[0]}")
    print(f"특성 수: {X.shape[1]}")
    print(f"클래스: {data.target_names}")
    print(f"\n특성 이름 (처음 10개):")
    for i, name in enumerate(data.feature_names[:10]):
        print(f"  {i+1}. {name}")
    
    # 클래스 분포
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📈 클래스 분포:")
    for label, count in zip(unique, counts):
        class_name = data.target_names[label]
        percentage = count / len(y) * 100
        print(f"  {class_name}: {count}개 ({percentage:.1f}%)")
    
    # 특성 통계
    print(f"\n📐 특성 통계 (처음 5개 특성):")
    for i in range(5):
        print(f"\n  {data.feature_names[i]}:")
        print(f"    평균: {X[:, i].mean():.2f}")
        print(f"    표준편차: {X[:, i].std():.2f}")
        print(f"    최소값: {X[:, i].min():.2f}")
        print(f"    최대값: {X[:, i].max():.2f}")
    
    return X, y, data


# ============================================================================
# 섹션 2: 파이프라인 기본 사용법
# ============================================================================

def section2_basic_pipeline(X, y):
    """
    파이프라인의 기본 사용법을 배웁니다.
    
    학습 목표:
    - Pipeline 객체 생성
    - 표준화 + 모델 학습을 한 번에 수행
    - 파이프라인의 장점 이해
    """
    print_section("섹션 2: 파이프라인 기본 사용법")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("🔧 방법 1: 수동으로 각 단계 실행 (파이프라인 없이)")
    print("-" * 60)
    
    # 수동 방식
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    manual_score = lr.score(X_test_scaled, y_test)
    
    print(f"학습 세트 크기: {X_train.shape}")
    print(f"테스트 세트 크기: {X_test.shape}")
    print(f"테스트 정확도: {manual_score:.4f}")
    
    print("\n🚀 방법 2: 파이프라인 사용")
    print("-" * 60)
    
    # 파이프라인 방식
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipe.fit(X_train, y_train)
    pipeline_score = pipe.score(X_test, y_test)
    
    print("파이프라인 구조:")
    for name, step in pipe.steps:
        print(f"  {name}: {step.__class__.__name__}")
    
    print(f"\n테스트 정확도: {pipeline_score:.4f}")
    print(f"수동 방식과 동일한 결과: {np.isclose(manual_score, pipeline_score)}")
    
    print("\n✅ 파이프라인의 장점:")
    print("  1. 코드가 간결하고 읽기 쉬움")
    print("  2. 전처리 단계를 잊어버릴 위험이 없음")
    print("  3. 교차 검증과 쉽게 결합 가능")
    print("  4. 하이퍼파라미터 튜닝이 용이")
    
    return pipe


# ============================================================================
# 섹션 3: 교차 검증 비교 (홀드아웃 vs k-겹)
# ============================================================================

def section3_cross_validation(X, y):
    """
    홀드아웃 방법과 k-겹 교차 검증을 비교합니다.
    
    학습 목표:
    - 홀드아웃의 한계 이해
    - k-겹 교차 검증의 장점
    - 층화 k-겹의 중요성
    """
    print_section("섹션 3: 교차 검증 비교")
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("📊 방법 1: 홀드아웃 방법 (단일 분할)")
    print("-" * 60)
    
    # 여러 번 반복하여 불안정성 확인
    holdout_scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        holdout_scores.append(score)
        if i < 3:  # 처음 3개만 출력
            print(f"  시도 {i+1}: {score:.4f}")
    
    print(f"  ...")
    print(f"\n홀드아웃 평균: {np.mean(holdout_scores):.4f} (±{np.std(holdout_scores):.4f})")
    print(f"최소값: {np.min(holdout_scores):.4f}")
    print(f"최대값: {np.max(holdout_scores):.4f}")
    print(f"범위: {np.max(holdout_scores) - np.min(holdout_scores):.4f}")
    
    print("\n📊 방법 2: k-겹 교차 검증")
    print("-" * 60)
    
    # 5-겹 교차 검증
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    
    print("각 Fold의 점수:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"\n교차 검증 평균: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    print("\n📊 방법 3: 층화 k-겹 교차 검증")
    print("-" * 60)
    
    # 층화 k-겹
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    
    print("각 Fold의 점수:")
    for i, score in enumerate(stratified_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"\n층화 교차 검증 평균: {stratified_scores.mean():.4f} (±{stratified_scores.std():.4f})")
    
    # 각 fold의 클래스 분포 확인
    print("\n각 Fold의 클래스 분포:")
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        y_fold = y[test_idx]
        unique, counts = np.unique(y_fold, return_counts=True)
        ratio = counts[1] / len(y_fold) * 100
        print(f"  Fold {i}: 악성 {ratio:.1f}%")
    
    print("\n✅ 결론:")
    print(f"  - 홀드아웃: 분할에 따라 성능 차이가 큼 (범위: ±{np.std(holdout_scores):.4f})")
    print(f"  - k-겹: 더 안정적인 추정 (표준편차: {cv_scores.std():.4f})")
    print(f"  - 층화 k-겹: 클래스 비율 유지로 더욱 신뢰성 있음")


# ============================================================================
# 섹션 4: 학습 곡선으로 편향-분산 분석
# ============================================================================

def section4_learning_curves(X, y):
    """
    학습 곡선을 그려 모델의 편향과 분산을 분석합니다.
    
    학습 목표:
    - 학습 곡선 생성
    - 과소적합/과대적합 판단
    - 데이터 추가 필요성 판단
    """
    print_section("섹션 4: 학습 곡선 분석")
    
    print("📈 다양한 모델의 학습 곡선 비교")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = [
        ('과소적합 예시\n(단순 모델)', 
         Pipeline([('scaler', StandardScaler()), 
                  ('clf', LogisticRegression(C=0.01, max_iter=1000))])),
        ('적절한 적합\n(기본 모델)', 
         Pipeline([('scaler', StandardScaler()), 
                  ('clf', LogisticRegression(C=1.0, max_iter=1000))])),
        ('약간 과대적합\n(복잡 모델)', 
         Pipeline([('scaler', StandardScaler()), 
                  ('clf', SVC(kernel='rbf', gamma='auto'))])),
        ('과대적합 예시\n(매우 복잡)', 
         Pipeline([('scaler', StandardScaler()), 
                  ('clf', DecisionTreeClassifier(max_depth=None))]))
    ]
    
    for idx, (title, model) in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        
        # 학습 곡선 계산
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy',
            n_jobs=-1
        )
        
        # 평균과 표준편차
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        # 그래프 그리기
        ax.plot(train_sizes, train_mean, label='학습 점수', marker='o', color='blue')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.15, color='blue')
        
        ax.plot(train_sizes, test_mean, label='검증 점수', marker='s', color='red')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                        alpha=0.15, color='red')
        
        ax.set_xlabel('학습 샘플 수', fontsize=10)
        ax.set_ylabel('정확도', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.05])
        
        # 최종 성능 출력
        final_train = train_mean[-1]
        final_test = test_mean[-1]
        gap = final_train - final_test
        
        print(f"\n{title.replace(chr(10), ' ')}:")
        print(f"  최종 학습 점수: {final_train:.4f}")
        print(f"  최종 검증 점수: {final_test:.4f}")
        print(f"  점수 차이: {gap:.4f}")
        
        if gap < 0.05 and final_test < 0.85:
            print(f"  진단: 과소적합 (높은 편향)")
            print(f"  해결책: 더 복잡한 모델 사용")
        elif gap > 0.15:
            print(f"  진단: 과대적합 (높은 분산)")
            print(f"  해결책: 더 많은 데이터 수집 또는 정규화 증가")
        else:
            print(f"  진단: 적절한 적합")
    
    plt.tight_layout()
    plt.savefig('/tmp/learning_curves.png', dpi=100, bbox_inches='tight')
    print(f"\n💾 그래프 저장: /tmp/learning_curves.png")
    plt.close()
    
    print("\n✅ 학습 곡선 해석 가이드:")
    print("  1. 과소적합: 두 곡선이 낮은 점수에서 수렴")
    print("  2. 과대적합: 학습 점수는 높지만 검증 점수는 낮음")
    print("  3. 적절: 두 곡선이 높은 점수에서 가깝게 수렴")
    print("  4. 더 많은 데이터 필요: 검증 곡선이 계속 상승 중")


# ============================================================================
# 섹션 5: 검증 곡선으로 하이퍼파라미터 분석
# ============================================================================

def section5_validation_curves(X, y):
    """
    검증 곡선을 사용하여 하이퍼파라미터의 영향을 분석합니다.
    
    학습 목표:
    - 검증 곡선 생성
    - 최적 하이퍼파라미터 범위 파악
    - 과대적합/과소적합 구간 확인
    """
    print_section("섹션 5: 검증 곡선 분석")
    
    print("📊 하이퍼파라미터에 따른 성능 변화")
    print("-" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Logistic Regression의 C 파라미터
    print("\n1️⃣ Logistic Regression의 정규화 파라미터 C")
    print("   (C가 작을수록 강한 정규화)")
    
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    param_range = np.logspace(-4, 4, 9)  # 0.0001 ~ 10000
    
    train_scores, test_scores = validation_curve(
        pipe_lr, X, y,
        param_name='clf__C',
        param_range=param_range,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    axes[0].semilogx(param_range, train_mean, label='학습 점수', marker='o', color='blue')
    axes[0].fill_between(param_range, train_mean - train_std, train_mean + train_std,
                         alpha=0.15, color='blue')
    axes[0].semilogx(param_range, test_mean, label='검증 점수', marker='s', color='red')
    axes[0].fill_between(param_range, test_mean - test_std, test_mean + test_std,
                         alpha=0.15, color='red')
    axes[0].set_xlabel('C (정규화 파라미터)', fontsize=11)
    axes[0].set_ylabel('정확도', fontsize=11)
    axes[0].set_title('Logistic Regression: C 파라미터 영향', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 최적값 찾기
    best_idx = np.argmax(test_mean)
    best_c = param_range[best_idx]
    best_score = test_mean[best_idx]
    
    print(f"\n   최적 C: {best_c:.4f}")
    print(f"   최고 검증 점수: {best_score:.4f}")
    
    # 구간별 분석
    print("\n   구간별 분석:")
    print(f"   C < 0.1: 과소적합 (강한 정규화)")
    print(f"   0.1 ≤ C ≤ 10: 적절한 범위")
    print(f"   C > 100: 과대적합 가능성 (약한 정규화)")
    
    # 2. Decision Tree의 max_depth 파라미터
    print("\n2️⃣ Decision Tree의 최대 깊이 (max_depth)")
    
    pipe_dt = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    
    param_range_dt = range(1, 21)
    
    train_scores_dt, test_scores_dt = validation_curve(
        pipe_dt, X, y,
        param_name='clf__max_depth',
        param_range=param_range_dt,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean_dt = train_scores_dt.mean(axis=1)
    train_std_dt = train_scores_dt.std(axis=1)
    test_mean_dt = test_scores_dt.mean(axis=1)
    test_std_dt = test_scores_dt.std(axis=1)
    
    axes[1].plot(param_range_dt, train_mean_dt, label='학습 점수', marker='o', color='blue')
    axes[1].fill_between(param_range_dt, train_mean_dt - train_std_dt, 
                         train_mean_dt + train_std_dt, alpha=0.15, color='blue')
    axes[1].plot(param_range_dt, test_mean_dt, label='검증 점수', marker='s', color='red')
    axes[1].fill_between(param_range_dt, test_mean_dt - test_std_dt, 
                         test_mean_dt + test_std_dt, alpha=0.15, color='red')
    axes[1].set_xlabel('max_depth (최대 깊이)', fontsize=11)
    axes[1].set_ylabel('정확도', fontsize=11)
    axes[1].set_title('Decision Tree: max_depth 파라미터 영향', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # 최적값 찾기
    best_idx_dt = np.argmax(test_mean_dt)
    best_depth = param_range_dt[best_idx_dt]
    best_score_dt = test_mean_dt[best_idx_dt]
    
    print(f"\n   최적 max_depth: {best_depth}")
    print(f"   최고 검증 점수: {best_score_dt:.4f}")
    
    # 과대적합 확인
    overfitting_gap = train_mean_dt[-1] - test_mean_dt[-1]
    print(f"\n   깊이 20에서의 과대적합 정도: {overfitting_gap:.4f}")
    print(f"   (학습 점수 - 검증 점수)")
    
    plt.tight_layout()
    plt.savefig('/tmp/validation_curves.png', dpi=100, bbox_inches='tight')
    print(f"\n💾 그래프 저장: /tmp/validation_curves.png")
    plt.close()
    
    print("\n✅ 검증 곡선 해석:")
    print("  - C가 너무 작으면: 과소적합 (모델이 너무 단순)")
    print("  - C가 적절하면: 최고 성능")
    print("  - C가 너무 크면: 과대적합 (학습 데이터에 과도하게 맞춤)")


# ============================================================================
# 섹션 6: 그리드 서치로 하이퍼파라미터 튜닝
# ============================================================================

def section6_grid_search(X, y):
    """
    그리드 서치를 사용하여 최적의 하이퍼파라미터를 찾습니다.
    
    학습 목표:
    - GridSearchCV 사용법
    - 파이프라인과 결합
    - 최적 파라미터 해석
    """
    print_section("섹션 6: 그리드 서치")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("🔍 그리드 서치: 체계적인 하이퍼파라미터 탐색")
    print("-" * 60)
    
    # 파이프라인 구성
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', SVC())
    ])
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'pca__n_components': [5, 10, 15, 20],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }
    
    print("탐색할 하이퍼파라미터:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = 4 * 4 * 2 * 2
    print(f"\n총 조합 수: {total_combinations}")
    print(f"5-겹 교차 검증 사용 시 학습 횟수: {total_combinations * 5}")
    
    # 그리드 서치 실행
    print("\n⏳ 그리드 서치 실행 중...")
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print("✅ 완료!")
    print(f"\n최적 하이퍼파라미터:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n최고 교차 검증 점수: {grid_search.best_score_:.4f}")
    
    # 테스트 세트 평가
    test_score = grid_search.score(X_test, y_test)
    print(f"테스트 세트 점수: {test_score:.4f}")
    
    # 상위 5개 조합
    print("\n📊 상위 5개 하이퍼파라미터 조합:")
    results = grid_search.cv_results_
    indices = np.argsort(results['mean_test_score'])[::-1][:5]
    
    for i, idx in enumerate(indices, 1):
        print(f"\n{i}위:")
        print(f"  점수: {results['mean_test_score'][idx]:.4f} (±{results['std_test_score'][idx]:.4f})")
        print(f"  파라미터: {results['params'][idx]}")
    
    # PCA 성분 수의 영향 분석
    print("\n📈 PCA 성분 수에 따른 평균 성능:")
    for n_comp in [5, 10, 15, 20]:
        mask = [p['pca__n_components'] == n_comp for p in results['params']]
        avg_score = results['mean_test_score'][mask].mean()
        print(f"  n_components={n_comp}: {avg_score:.4f}")
    
    return grid_search


# ============================================================================
# 섹션 7: 랜덤 서치로 효율적인 탐색
# ============================================================================

def section7_random_search(X, y):
    """
    랜덤 서치를 사용하여 더 넓은 범위를 효율적으로 탐색합니다.
    
    학습 목표:
    - RandomizedSearchCV 사용
    - 연속 분포에서 샘플링
    - 그리드 서치와 비교
    """
    print_section("섹션 7: 랜덤 서치")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("🎲 랜덤 서치: 넓은 범위를 효율적으로 탐색")
    print("-" * 60)
    
    # 파이프라인
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # 랜덤 서치용 파라미터 분포
    param_distributions = {
        'classifier__n_estimators': randint(50, 500),  # 50~500 사이의 정수
        'classifier__max_depth': randint(5, 50),       # 5~50 사이의 정수
        'classifier__min_samples_split': randint(2, 20),
        'classifier__min_samples_leaf': randint(1, 10),
        'classifier__max_features': uniform(0.1, 0.9)  # 0.1~1.0 사이의 실수
    }
    
    print("탐색할 하이퍼파라미터 분포:")
    for param, dist in param_distributions.items():
        print(f"  {param}: {dist}")
    
    # 랜덤 서치 실행
    n_iter = 50  # 50개 조합만 시도
    print(f"\n시도할 조합 수: {n_iter}")
    print(f"5-겹 교차 검증 사용 시 학습 횟수: {n_iter * 5}")
    
    print("\n⏳ 랜덤 서치 실행 중...")
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0,
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    
    print("✅ 완료!")
    print(f"\n최적 하이퍼파라미터:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n최고 교차 검증 점수: {random_search.best_score_:.4f}")
    
    test_score = random_search.score(X_test, y_test)
    print(f"테스트 세트 점수: {test_score:.4f}")
    
    # 탐색 과정 시각화
    results = random_search.cv_results_
    
    print("\n📊 탐색한 하이퍼파라미터 범위:")
    for param in param_distributions.keys():
        param_short = param.split('__')[1]
        values = [p[param] for p in results['params']]
        print(f"  {param_short}:")
        print(f"    최소: {min(values)}")
        print(f"    최대: {max(values)}")
        print(f"    최적: {random_search.best_params_[param]}")
    
    # 상위 5개 조합
    print("\n🏆 상위 5개 조합:")
    indices = np.argsort(results['mean_test_score'])[::-1][:5]
    for i, idx in enumerate(indices, 1):
        print(f"\n{i}위:")
        print(f"  점수: {results['mean_test_score'][idx]:.4f}")
        params = results['params'][idx]
        for k, v in params.items():
            param_short = k.split('__')[1]
            if isinstance(v, float):
                print(f"  {param_short}: {v:.3f}")
            else:
                print(f"  {param_short}: {v}")
    
    print("\n✅ 랜덤 서치의 장점:")
    print("  - 넓은 범위를 효율적으로 탐색")
    print("  - 중요한 파라미터에 더 많은 값 시도")
    print("  - 시간 제약이 있을 때 유용")
    print("  - 연속 분포 사용 가능")
    
    return random_search


# ============================================================================
# 섹션 8: 성능 평가 지표 - 오차 행렬과 분류 리포트
# ============================================================================

def section8_classification_metrics(X, y):
    """
    다양한 분류 성능 지표를 계산하고 해석합니다.
    
    학습 목표:
    - 오차 행렬 이해
    - 정밀도, 재현율, F1 점수
    - 분류 리포트 해석
    """
    print_section("섹션 8: 분류 성능 지표")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 모델 학습
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    
    # 예측
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    
    print("📊 오차 행렬 (Confusion Matrix)")
    print("-" * 60)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n오차 행렬:")
    print("               예측")
    print("           악성   양성")
    print(f"실제 악성   {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"    양성   {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    # 용어 설명
    tn, fp, fn, tp = cm.ravel()
    print(f"\n용어 정리:")
    print(f"  TN (True Negative):  {tn} - 양성을 양성으로 정확히 예측")
    print(f"  FP (False Positive): {fp} - 양성을 악성으로 잘못 예측")
    print(f"  FN (False Negative): {fn} - 악성을 양성으로 잘못 예측")
    print(f"  TP (True Positive):  {tp} - 악성을 악성으로 정확히 예측")
    
    print("\n📈 성능 지표")
    print("-" * 60)
    
    # 정확도
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"\n1️⃣ 정확도 (Accuracy)")
    print(f"   공식: (TP + TN) / (TP + TN + FP + FN)")
    print(f"   계산: ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})")
    print(f"   결과: {accuracy:.4f}")
    print(f"   의미: 전체 예측 중 {accuracy*100:.2f}%가 정확")
    
    # 정밀도
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"\n2️⃣ 정밀도 (Precision)")
    print(f"   공식: TP / (TP + FP)")
    print(f"   계산: {tp} / ({tp} + {fp})")
    print(f"   결과: {precision:.4f}")
    print(f"   의미: 악성으로 예측한 것 중 {precision*100:.2f}%가 실제 악성")
    
    # 재현율
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n3️⃣ 재현율 (Recall) / 민감도 (Sensitivity)")
    print(f"   공식: TP / (TP + FN)")
    print(f"   계산: {tp} / ({tp} + {fn})")
    print(f"   결과: {recall:.4f}")
    print(f"   의미: 실제 악성 중 {recall*100:.2f}%를 찾아냄")
    
    # F1 점수
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n4️⃣ F1 점수 (F1-Score)")
    print(f"   공식: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   계산: 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"   결과: {f1:.4f}")
    print(f"   의미: 정밀도와 재현율의 조화 평균")
    
    # 특이도
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n5️⃣ 특이도 (Specificity)")
    print(f"   공식: TN / (TN + FP)")
    print(f"   계산: {tn} / ({tn} + {fp})")
    print(f"   결과: {specificity:.4f}")
    print(f"   의미: 실제 양성 중 {specificity*100:.2f}%를 정확히 판별")
    
    # 분류 리포트
    print("\n📋 종합 분류 리포트")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['악성', '양성']))
    
    print("\n✅ 지표 선택 가이드:")
    print("  - 정확도: 클래스가 균형잡혔을 때")
    print("  - 정밀도: 거짓 양성이 치명적일 때 (예: 스팸 필터)")
    print("  - 재현율: 거짓 음성이 치명적일 때 (예: 암 진단)")
    print("  - F1 점수: 정밀도와 재현율의 균형이 필요할 때")
    
    # 오차 행렬 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['악성', '양성'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('오차 행렬 (Confusion Matrix)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/tmp/confusion_matrix.png', dpi=100, bbox_inches='tight')
    print(f"\n💾 오차 행렬 그래프 저장: /tmp/confusion_matrix.png")
    plt.close()


# ============================================================================
# 섹션 9: ROC 곡선과 AUC
# ============================================================================

def section9_roc_curve(X, y):
    """
    ROC 곡선을 그리고 AUC를 계산합니다.
    
    학습 목표:
    - ROC 곡선 이해
    - AUC 해석
    - 임계값 조정
    """
    print_section("섹션 9: ROC 곡선과 AUC")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("📈 ROC 곡선 (Receiver Operating Characteristic)")
    print("-" * 60)
    
    # 여러 모델 비교
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=42))
        ])
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # 예측 확률
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC 곡선 계산
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # ROC 곡선 그리기
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        print(f"\n{name}:")
        print(f"  AUC: {roc_auc:.4f}")
        
        # 최적 임계값 찾기 (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        print(f"  최적 임계값: {optimal_threshold:.3f}")
        print(f"  최적 임계값에서 TPR: {optimal_tpr:.3f}")
        print(f"  최적 임계값에서 FPR: {optimal_fpr:.3f}")
    
    # 무작위 분류기 (대각선)
    plt.plot([0, 1], [0, 1], 'k--', label='무작위 분류기 (AUC = 0.5)', linewidth=1)
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC 곡선 비교', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/roc_curves.png', dpi=100, bbox_inches='tight')
    print(f"\n💾 ROC 곡선 저장: /tmp/roc_curves.png")
    plt.close()
    
    print("\n✅ AUC 해석:")
    print("  - AUC = 1.0: 완벽한 분류기")
    print("  - AUC = 0.9-1.0: 탁월")
    print("  - AUC = 0.8-0.9: 우수")
    print("  - AUC = 0.7-0.8: 양호")
    print("  - AUC = 0.6-0.7: 보통")
    print("  - AUC = 0.5: 무작위 수준")
    print("  - AUC < 0.5: 무작위보다 못함")
    
    # Precision-Recall 곡선도 그리기
    print("\n📊 Precision-Recall 곡선")
    print("-" * 60)
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # PR 곡선 계산
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})', linewidth=2)
        
        print(f"{name} PR-AUC: {pr_auc:.4f}")
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall 곡선', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/pr_curves.png', dpi=100, bbox_inches='tight')
    print(f"\n💾 PR 곡선 저장: /tmp/pr_curves.png")
    plt.close()


# ============================================================================
# 섹션 10: 불균형 클래스 처리
# ============================================================================

def section10_imbalanced_classes():
    """
    불균형한 클래스를 다루는 여러 기법을 배웁니다.
    
    학습 목표:
    - 클래스 불균형 문제 이해
    - 클래스 가중치 사용
    - 리샘플링 기법
    - 적절한 평가 지표 선택
    """
    print_section("섹션 10: 불균형한 클래스 처리")
    
    # 불균형 데이터 생성 (1:9 비율)
    print("🔧 불균형 데이터 생성 (악성:양성 = 1:9)")
    print("-" * 60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 악성(0)을 소수 클래스로 만들기
    malignant_indices = np.where(y == 0)[0]
    benign_indices = np.where(y == 1)[0]
    
    # 악성은 일부만, 양성은 더 많이 사용
    # 악성의 30%만 사용, 양성은 전체 사용하여 1:9 비율 만들기
    n_malignant = int(len(malignant_indices) * 0.3)
    selected_malignant = np.random.choice(malignant_indices, size=n_malignant, replace=False)
    selected_benign = benign_indices  # 전체 사용
    
    selected_indices = np.concatenate([selected_malignant, selected_benign])
    np.random.shuffle(selected_indices)
    
    X_imb = X[selected_indices]
    y_imb = y[selected_indices]
    
    # 클래스 분포 확인
    unique, counts = np.unique(y_imb, return_counts=True)
    print(f"\n클래스 분포:")
    for label, count in zip(unique, counts):
        percentage = count / len(y_imb) * 100
        print(f"  클래스 {label}: {count}개 ({percentage:.1f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
    )
    
    print("\n📊 방법 1: 기본 모델 (불균형 무시)")
    print("-" * 60)
    
    pipe_basic = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipe_basic.fit(X_train, y_train)
    
    y_pred_basic = pipe_basic.predict(X_test)
    
    print("\n오차 행렬:")
    cm_basic = confusion_matrix(y_test, y_pred_basic)
    print(cm_basic)
    
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred_basic, target_names=['악성', '양성']))
    
    print("\n⚠️  문제점:")
    print(f"  - 소수 클래스(악성) 재현율이 낮음")
    print(f"  - 정확도는 높아 보이지만 실제로는 쓸모없음")
    
    print("\n📊 방법 2: 클래스 가중치 사용")
    print("-" * 60)
    
    pipe_weighted = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    pipe_weighted.fit(X_train, y_train)
    
    y_pred_weighted = pipe_weighted.predict(X_test)
    
    print("\n오차 행렬:")
    cm_weighted = confusion_matrix(y_test, y_pred_weighted)
    print(cm_weighted)
    
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred_weighted, target_names=['악성', '양성']))
    
    # 가중치 계산 설명
    n_samples = len(y_train)
    n_classes = 2
    class_counts = np.bincount(y_train)
    
    print("\n가중치 계산:")
    for i in range(n_classes):
        weight = n_samples / (n_classes * class_counts[i])
        print(f"  클래스 {i}: {weight:.3f}")
        print(f"    계산: {n_samples} / ({n_classes} × {class_counts[i]})")
    
    print("\n✅ 개선점:")
    print(f"  - 소수 클래스의 재현율 향상")
    print(f"  - 클래스 간 성능 균형")
    
    print("\n📊 방법 3: Random Forest (균형 모드)")
    print("-" * 60)
    
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)
    
    y_pred_rf = pipe_rf.predict(X_test)
    
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred_rf, target_names=['악성', '양성']))
    
    print("\n📊 방법 비교 (AUC 기준)")
    print("-" * 60)
    
    models_comparison = {
        '기본 모델': pipe_basic,
        '가중치 적용': pipe_weighted,
        'Random Forest': pipe_rf
    }
    
    for name, model in models_comparison.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        y_pred = model.predict(X_test)
        f1_malignant = f1_score(y_test, y_pred, pos_label=0)
        f1_benign = f1_score(y_test, y_pred, pos_label=1)
        
        print(f"\n{name}:")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  F1 (악성): {f1_malignant:.4f}")
        print(f"  F1 (양성): {f1_benign:.4f}")
    
    print("\n✅ 불균형 클래스 처리 요약:")
    print("  1. 클래스 가중치: 간단하고 효과적")
    print("  2. 리샘플링: 데이터 크기 조정 가능")
    print("  3. 앙상블: Random Forest 등 효과적")
    print("  4. 평가 지표: AUC, F1 점수 사용 필수")
    print("  5. 정확도는 의미 없음!")


# ============================================================================
# 메인 실행 함수
# ============================================================================

def main():
    """
    모든 섹션을 순서대로 실행합니다.
    """
    print("\n" + "="*80)
    print("  모델 평가와 하이퍼파라미터 튜닝 - 실습")
    print("  Model Evaluation and Hyperparameter Tuning")
    print("="*80)
    print("\n이 튜토리얼에서 배울 내용:")
    print("  1. 위스콘신 유방암 데이터셋 탐색")
    print("  2. 파이프라인으로 효율적인 워크플로 구성")
    print("  3. k-겹 교차 검증으로 신뢰성 있는 평가")
    print("  4. 학습/검증 곡선으로 모델 진단")
    print("  5. 그리드/랜덤 서치로 하이퍼파라미터 튜닝")
    print("  6. 다양한 성능 평가 지표 이해")
    print("  7. 불균형 클래스 처리 기법")
    print("\n총 예상 시간: 60-90분")
    print("="*80)
    
    try:
        # 섹션 1: 데이터 로드
        X, y, data = section1_load_data()
        
        # 섹션 2: 파이프라인 기본
        pipe = section2_basic_pipeline(X, y)
        
        # 섹션 3: 교차 검증
        section3_cross_validation(X, y)
        
        # 섹션 4: 학습 곡선
        section4_learning_curves(X, y)
        
        # 섹션 5: 검증 곡선
        section5_validation_curves(X, y)
        
        # 섹션 6: 그리드 서치
        grid_search = section6_grid_search(X, y)
        
        # 섹션 7: 랜덤 서치
        random_search = section7_random_search(X, y)
        
        # 섹션 8: 분류 지표
        section8_classification_metrics(X, y)
        
        # 섹션 9: ROC 곡선
        section9_roc_curve(X, y)
        
        # 섹션 10: 불균형 클래스
        section10_imbalanced_classes()
        
        print("\n" + "="*80)
        print("  🎉 모든 섹션 완료!")
        print("="*80)
        print("\n생성된 파일:")
        print("  - /tmp/learning_curves.png")
        print("  - /tmp/validation_curves.png")
        print("  - /tmp/confusion_matrix.png")
        print("  - /tmp/roc_curves.png")
        print("  - /tmp/pr_curves.png")
        print("\n다음 단계:")
        print("  1. 생성된 그래프를 확인하세요")
        print("  2. 코드를 수정하며 실험해보세요")
        print("  3. 자신의 데이터셋에 적용해보세요")
        print("  4. model_evaluation_theory.md에서 이론을 복습하세요")
        print("\n행운을 빕니다! 🚀")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
