# 6교시: 머신러닝 기초 - 예측 모델 만들기

> **학습 시간**: 1시간  
> **난이도**: ⭐⭐⭐⭐  
> **목표**: Scikit-learn을 사용하여 실무 예측 모델을 구축하고 평가합니다.

---

## 📚 학습 내용

1. 머신러닝 개념
2. 데이터 전처리
3. 회귀 모델 (매출 예측)
4. 분류 모델 (고객 등급 예측)
5. 모델 평가

---

## 1. 머신러닝 개념

### 1.1 머신러닝이란?

**머신러닝**은 데이터에서 패턴을 학습하여 예측하는 기술입니다.

**실무 활용 예시:**
- 📊 매출 예측
- 👥 고객 이탈 예측
- 💰 신용 평가
- 🎯 추천 시스템

### 1.2 머신러닝 워크플로우

```
1. 문제 정의
   ↓
2. 데이터 수집
   ↓
3. 데이터 전처리
   ↓
4. 모델 선택
   ↓
5. 모델 학습
   ↓
6. 모델 평가
   ↓
7. 예측 및 활용
```

---

## 2. 데이터 준비 및 전처리

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df = pd.read_csv('../data/sales_data.csv')
print(f"데이터 크기: {df.shape}")

# 결측치 처리
df['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
df['region'].fillna('Unknown', inplace=True)

# 날짜 처리
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.month
df['day_of_week'] = df['order_date'].dt.dayofweek
df['quarter'] = df['order_date'].dt.quarter

print("\n=== 전처리 완료 ===")
print(df.info())
```

### 2.1 범주형 변수 인코딩

```python
# Label Encoding
le_category = LabelEncoder()
df['category_encoded'] = le_category.fit_transform(df['product_category'])

le_region = LabelEncoder()
df['region_encoded'] = le_region.fit_transform(df['region'])

le_payment = LabelEncoder()
df['payment_encoded'] = le_payment.fit_transform(df['payment_method'])

le_gender = LabelEncoder()
df['gender_encoded'] = le_gender.fit_transform(df['customer_gender'])

print("=== 인코딩 완료 ===")
print(f"카테고리 종류: {le_category.classes_}")
```

---

## 3. 회귀 모델: 매출 예측

### 3.1 특성(Feature) 선택

```python
# 독립 변수 (X): 예측에 사용할 특성
feature_cols = [
    'quantity', 'unit_price', 'customer_age',
    'category_encoded', 'region_encoded', 'payment_encoded',
    'gender_encoded', 'month', 'day_of_week', 'quarter'
]

X = df[feature_cols]
y = df['final_amount']  # 종속 변수 (Target): 예측할 값

print(f"특성(X) 크기: {X.shape}")
print(f"타겟(y) 크기: {y.shape}")
```

### 3.2 Train/Test 분리

```python
# 8:2 비율로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
```

### 3.3 스케일링

```python
# 표준화 (평균 0, 표준편차 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== 스케일링 완료 ===")
print(f"학습 데이터 평균: {X_train_scaled.mean():.4f}")
print(f"학습 데이터 표준편차: {X_train_scaled.std():.4f}")
```

### 3.4 모델 학습 및 평가

#### 선형 회귀

```python
# 모델 생성 및 학습
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# 예측
y_pred_train = lr_model.predict(X_train_scaled)
y_pred_test = lr_model.predict(X_test_scaled)

# 평가
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("=== 선형 회귀 결과 ===")
print(f"학습 MSE: {train_mse:,.0f}")
print(f"테스트 MSE: {test_mse:,.0f}")
print(f"학습 R²: {train_r2:.4f}")
print(f"테스트 R²: {test_r2:.4f}")
```

#### 랜덤 포레스트

```python
# 랜덤 포레스트 모델
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred_rf = rf_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print("\n=== 랜덤 포레스트 결과 ===")
print(f"테스트 MSE: {rf_mse:,.0f}")
print(f"테스트 R²: {rf_r2:.4f}")
print(f"테스트 MAE: {rf_mae:,.0f}원")
```

### 3.5 특성 중요도 분석

```python
# 특성 중요도
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== 특성 중요도 TOP 5 ===")
print(feature_importance.head())

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('../output/figures/feature_importance.png', dpi=300)
plt.show()
```

### 3.6 예측 결과 시각화

```python
# 실제값 vs 예측값
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales Amount (KRW)')
plt.ylabel('Predicted Sales Amount (KRW)')
plt.title(f'Actual vs Predicted (R² = {rf_r2:.4f})')
plt.tight_layout()
plt.savefig('../output/figures/prediction_scatter.png', dpi=300)
plt.show()
```

---

## 4. 분류 모델: 고객 등급 예측

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 고객 등급 생성 (매출 기준)
def classify_customer(amount):
    if amount >= 1000000:
        return 'VIP'
    elif amount >= 500000:
        return 'Gold'
    else:
        return 'Silver'

df['customer_grade'] = df['final_amount'].apply(classify_customer)

# Label Encoding
le_grade = LabelEncoder()
df['grade_encoded'] = le_grade.fit_transform(df['customer_grade'])

# 특성과 타겟
X_class = df[feature_cols]
y_class = df['grade_encoded']

# Train/Test 분리
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 스케일링
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# 모델 학습
clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_model.fit(X_train_c_scaled, y_train_c)

# 예측
y_pred_class = clf_model.predict(X_test_c_scaled)

# 평가
accuracy = accuracy_score(y_test_c, y_pred_class)
print(f"\n=== 분류 모델 정확도: {accuracy:.4f} ===")

print("\n=== 분류 리포트 ===")
print(classification_report(y_test_c, y_pred_class, target_names=le_grade.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_grade.classes_, yticklabels=le_grade.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('../output/figures/confusion_matrix.png', dpi=300)
plt.show()
```

---

## 5. 모델 저장 및 로드

```python
import joblib

# 모델 저장
joblib.dump(rf_model, '../output/models/sales_prediction_model.pkl')
joblib.dump(scaler, '../output/models/scaler.pkl')
print("✅ 모델 저장 완료")

# 모델 로드
loaded_model = joblib.load('../output/models/sales_prediction_model.pkl')
loaded_scaler = joblib.load('../output/models/scaler.pkl')

# 새 데이터 예측
new_data = [[5, 150000, 35, 0, 1, 2, 0, 6, 3, 2]]  # 예시 데이터
new_data_scaled = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_data_scaled)
print(f"\n예측 매출액: {prediction[0]:,.0f}원")
```

---

## 💪 실습 문제

### 문제 1: 모델 개선

다른 특성을 추가하여 모델 성능을 개선해보세요:
- 할인율 (discount_rate)
- 총 구매액 (total_amount)

```python
# TODO: 코드 작성
```

### 문제 2: 하이퍼파라미터 튜닝

RandomForestRegressor의 하이퍼파라미터를 조정하여 성능을 개선하세요:
- n_estimators
- max_depth
- min_samples_split

```python
# TODO: 코드 작성
```

---

## 📝 정리

✅ **머신러닝 개념**: 지도학습, 회귀, 분류  
✅ **데이터 전처리**: 인코딩, 스케일링  
✅ **모델 학습**: LinearRegression, RandomForest  
✅ **모델 평가**: MSE, R², Accuracy  
✅ **특성 중요도**: 예측에 영향을 주는 변수 분석  
✅ **모델 저장**: joblib로 모델 저장 및 로드

---

**수고하셨습니다! 🎉**
