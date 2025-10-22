#!/usr/bin/env python3
"""
앙상블 학습 (Ensemble Learning) - 실습 예제

이 파일은 다양한 앙상블 학습 알고리즘을 실습하는 완전한 예제입니다.

섹션:
1. 데이터 준비 및 탐색
2. 단일 모델 vs 앙상블 비교
3. Voting Classifier (하드/소프트 보팅)
4. Bagging과 Random Forest
5. AdaBoost
6. Gradient Boosting과 XGBoost
7. 모델 성능 비교 및 시각화
8. 하이퍼파라미터 튜닝

실행 시간: ~3-5분
학습 시간: 60-90분
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (선택사항)
rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("ENSEMBLE LEARNING - COMPREHENSIVE TUTORIAL")
print("=" * 80)
print()

# ============================================================================
# 섹션 1: 데이터 준비 및 탐색
# ============================================================================
print("\n" + "=" * 80)
print("Section 1: Data Preparation and Exploration")
print("=" * 80)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 유방암 데이터셋 로드
data = load_breast_cancer()
X, y = data.data, data.target

print(f"\nDataset Information:")
print(f"  Number of samples: {X.shape[0]}")
print(f"  Number of features: {X.shape[1]}")
print(f"  Number of classes: {len(np.unique(y))}")
print(f"  Class distribution: {np.bincount(y)}")
print(f"  Feature names: {data.feature_names[:5]}... (showing first 5)")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# 섹션 2: 단일 모델 vs 앙상블 비교
# ============================================================================
print("\n" + "=" * 80)
print("Section 2: Single Model vs Ensemble Comparison")
print("=" * 80)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 단일 결정 트리
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train_scaled, y_train)

# 랜덤 포레스트 (앙상블)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_scaled, y_train)

print(f"\nSingle Decision Tree:")
print(f"  Train Accuracy: {single_tree.score(X_train_scaled, y_train):.4f}")
print(f"  Test Accuracy:  {single_tree.score(X_test_scaled, y_test):.4f}")

print(f"\nRandom Forest (100 trees):")
print(f"  Train Accuracy: {random_forest.score(X_train_scaled, y_train):.4f}")
print(f"  Test Accuracy:  {random_forest.score(X_test_scaled, y_test):.4f}")

improvement = (random_forest.score(X_test_scaled, y_test) - 
               single_tree.score(X_test_scaled, y_test)) * 100
print(f"\nImprovement: {improvement:.2f}%")

# ============================================================================
# 섹션 3: Voting Classifier (하드/소프트 보팅)
# ============================================================================
print("\n" + "=" * 80)
print("Section 3: Voting Classifier (Hard and Soft Voting)")
print("=" * 80)

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 개별 분류기
lr = LogisticRegression(random_state=42, max_iter=1000)
svm = SVC(kernel='rbf', probability=True, random_state=42)
nb = GaussianNB()
dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# 하드 보팅
voting_hard = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('nb', nb), ('dt', dt)],
    voting='hard'
)

# 소프트 보팅
voting_soft = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('nb', nb), ('dt', dt)],
    voting='soft'
)

# 학습 및 평가
print("\nTraining individual classifiers and voting ensembles...")
for name, clf in [('LR', lr), ('SVM', svm), ('NB', nb), ('DT', dt),
                  ('Hard Voting', voting_hard), ('Soft Voting', voting_soft)]:
    clf.fit(X_train_scaled, y_train)
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    print(f"{name:15s} - Train: {train_score:.4f}, Test: {test_score:.4f}")

# ============================================================================
# 섹션 4: Bagging과 Random Forest
# ============================================================================
print("\n" + "=" * 80)
print("Section 4: Bagging and Random Forest")
print("=" * 80)

from sklearn.ensemble import BaggingClassifier

# Bagging with Decision Trees
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

print("\nTraining Bagging and Random Forest...")
bagging.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

print(f"\nBagging Classifier:")
print(f"  Train Accuracy: {bagging.score(X_train_scaled, y_train):.4f}")
print(f"  Test Accuracy:  {bagging.score(X_test_scaled, y_test):.4f}")

print(f"\nRandom Forest:")
print(f"  Train Accuracy: {rf.score(X_train_scaled, y_train):.4f}")
print(f"  Test Accuracy:  {rf.score(X_test_scaled, y_test):.4f}")

# 트리 개수에 따른 성능 변화
print("\nPerformance vs Number of Trees:")
for n in [10, 50, 100, 200]:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train_scaled, y_train)
    print(f"  n_estimators={n:3d}: Test Acc = {rf_temp.score(X_test_scaled, y_test):.4f}")

# ============================================================================
# 섹션 5: AdaBoost
# ============================================================================
print("\n" + "=" * 80)
print("Section 5: AdaBoost (Adaptive Boosting)")
print("=" * 80)

from sklearn.ensemble import AdaBoostClassifier

# AdaBoost with Decision Stumps
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

print("\nTraining AdaBoost...")
ada.fit(X_train_scaled, y_train)

print(f"\nAdaBoost:")
print(f"  Train Accuracy: {ada.score(X_train_scaled, y_train):.4f}")
print(f"  Test Accuracy:  {ada.score(X_test_scaled, y_test):.4f}")

# 학습 과정 분석
print(f"\nFirst 10 Estimator Errors:")
for i, error in enumerate(ada.estimator_errors_[:10], 1):
    print(f"  Round {i:2d}: Error = {error:.4f}")

print(f"\nFirst 10 Estimator Weights:")
for i, weight in enumerate(ada.estimator_weights_[:10], 1):
    print(f"  Round {i:2d}: Weight = {weight:.4f}")

# ============================================================================
# 섹션 6: Gradient Boosting과 XGBoost
# ============================================================================
print("\n" + "=" * 80)
print("Section 6: Gradient Boosting and XGBoost")
print("=" * 80)

from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

print("\nTraining Gradient Boosting...")
gb.fit(X_train_scaled, y_train)

print(f"\nGradient Boosting:")
print(f"  Train Accuracy: {gb.score(X_train_scaled, y_train):.4f}")
print(f"  Test Accuracy:  {gb.score(X_test_scaled, y_test):.4f}")

# XGBoost (if available)
try:
    from xgboost import XGBClassifier
    
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("\nTraining XGBoost...")
    xgb.fit(X_train_scaled, y_train)
    
    print(f"\nXGBoost:")
    print(f"  Train Accuracy: {xgb.score(X_train_scaled, y_train):.4f}")
    print(f"  Test Accuracy:  {xgb.score(X_test_scaled, y_test):.4f}")
    
    xgboost_available = True
except ImportError:
    print("\nXGBoost not installed. Skipping XGBoost examples.")
    print("Install with: pip install xgboost")
    xgboost_available = False

# 학습률 효과 분석
print("\nLearning Rate Effect on Gradient Boosting:")
for lr in [0.01, 0.05, 0.1, 0.5, 1.0]:
    gb_temp = GradientBoostingClassifier(
        n_estimators=100, learning_rate=lr, max_depth=3, random_state=42
    )
    gb_temp.fit(X_train_scaled, y_train)
    print(f"  LR = {lr:4.2f}: Test Acc = {gb_temp.score(X_test_scaled, y_test):.4f}")

# ============================================================================
# 섹션 7: 모델 성능 비교 및 시각화
# ============================================================================
print("\n" + "=" * 80)
print("Section 7: Model Performance Comparison and Visualization")
print("=" * 80)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# 모든 모델 수집
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Voting (Soft)': voting_soft,
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), 
                                 n_estimators=100, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

if xgboost_available:
    models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42,
                                     use_label_encoder=False, eval_metric='logloss')

# 성능 평가
results = []
print("\nEvaluating all models...")

for name, model in models.items():
    print(f"  Training {name}...")
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_scaled)
    
    results.append({
        'Model': name,
        'Train Acc': accuracy_score(y_train, model.predict(X_train_scaled)),
        'Test Acc': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'Train Time': train_time
    })

# 결과 표시
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Test Acc', ascending=False)

print("\n" + "=" * 100)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 100)
print(df_results.to_string(index=False))
print("=" * 100)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Test Accuracy 비교
ax1 = axes[0, 0]
sorted_df = df_results.sort_values('Test Acc')
bars = ax1.barh(sorted_df['Model'], sorted_df['Test Acc'], color='skyblue', edgecolor='black')
for bar, value in zip(bars, sorted_df['Test Acc']):
    ax1.text(value, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
            ha='left', va='center', fontsize=9, weight='bold')
ax1.set_xlabel('Test Accuracy', fontsize=11, weight='bold')
ax1.set_title('Test Accuracy Comparison', fontsize=12, weight='bold')
ax1.set_xlim([0.9, 1.0])
ax1.grid(axis='x', alpha=0.3)

# 2. F1 Score 비교
ax2 = axes[0, 1]
sorted_df = df_results.sort_values('F1')
bars = ax2.barh(sorted_df['Model'], sorted_df['F1'], color='coral', edgecolor='black')
for bar, value in zip(bars, sorted_df['F1']):
    ax2.text(value, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
            ha='left', va='center', fontsize=9, weight='bold')
ax2.set_xlabel('F1 Score', fontsize=11, weight='bold')
ax2.set_title('F1 Score Comparison', fontsize=12, weight='bold')
ax2.set_xlim([0.9, 1.0])
ax2.grid(axis='x', alpha=0.3)

# 3. Training Time 비교
ax3 = axes[1, 0]
sorted_df = df_results.sort_values('Train Time')
bars = ax3.barh(sorted_df['Model'], sorted_df['Train Time'], color='lightgreen', edgecolor='black')
for bar, value in zip(bars, sorted_df['Train Time']):
    ax3.text(value, bar.get_y() + bar.get_height()/2, f'{value:.2f}s',
            ha='left', va='center', fontsize=9, weight='bold')
ax3.set_xlabel('Training Time (seconds)', fontsize=11, weight='bold')
ax3.set_title('Training Time Comparison', fontsize=12, weight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Precision vs Recall
ax4 = axes[1, 1]
scatter = ax4.scatter(df_results['Recall'], df_results['Precision'], 
                     s=200, c=df_results['Test Acc'], cmap='viridis',
                     edgecolor='black', linewidth=2, alpha=0.7)
for idx, row in df_results.iterrows():
    ax4.annotate(row['Model'], (row['Recall'], row['Precision']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.colorbar(scatter, ax=ax4, label='Test Accuracy')
ax4.set_xlabel('Recall', fontsize=11, weight='bold')
ax4.set_ylabel('Precision', fontsize=11, weight='bold')
ax4.set_title('Precision vs Recall', fontsize=12, weight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'ensemble_comparison.png'")
plt.show()

# ============================================================================
# 섹션 8: 하이퍼파라미터 튜닝
# ============================================================================
print("\n" + "=" * 80)
print("Section 8: Hyperparameter Tuning with GridSearchCV")
print("=" * 80)

from sklearn.model_selection import GridSearchCV

# Random Forest 튜닝
print("\nTuning Random Forest...")
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_rf.fit(X_train_scaled, y_train)

print(f"\nRandom Forest - Best Parameters: {grid_rf.best_params_}")
print(f"Random Forest - Best CV Score: {grid_rf.best_score_:.4f}")
print(f"Random Forest - Test Score: {grid_rf.score(X_test_scaled, y_test):.4f}")

# Gradient Boosting 튜닝
print("\nTuning Gradient Boosting...")
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_gb.fit(X_train_scaled, y_train)

print(f"\nGradient Boosting - Best Parameters: {grid_gb.best_params_}")
print(f"Gradient Boosting - Best CV Score: {grid_gb.best_score_:.4f}")
print(f"Gradient Boosting - Test Score: {grid_gb.score(X_test_scaled, y_test):.4f}")

# 최종 비교
print("\n" + "=" * 80)
print("FINAL COMPARISON: Default vs Tuned Models")
print("=" * 80)

print(f"\nRandom Forest:")
print(f"  Default:  {RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train).score(X_test_scaled, y_test):.4f}")
print(f"  Tuned:    {grid_rf.score(X_test_scaled, y_test):.4f}")

print(f"\nGradient Boosting:")
print(f"  Default:  {GradientBoostingClassifier(random_state=42).fit(X_train_scaled, y_train).score(X_test_scaled, y_test):.4f}")
print(f"  Tuned:    {grid_gb.score(X_test_scaled, y_test):.4f}")

# ============================================================================
# 마무리
# ============================================================================
print("\n" + "=" * 80)
print("TUTORIAL COMPLETED!")
print("=" * 80)
print("\nKey Takeaways:")
print("  1. Ensemble methods generally outperform single models")
print("  2. Different ensemble methods have different strengths:")
print("     - Bagging/RF: Reduce variance, good for high-variance models")
print("     - Boosting: Reduce bias, sequential improvement")
print("     - Voting: Combine diverse models")
print("  3. Hyperparameter tuning can significantly improve performance")
print("  4. Consider trade-offs: accuracy, training time, interpretability")
print("\nNext Steps:")
print("  - Try on your own datasets")
print("  - Experiment with different base estimators")
print("  - Explore stacking and other advanced ensemble methods")
print("  - Participate in Kaggle competitions!")
print("\n" + "=" * 80)
