# 3교시: Pandas 고급 - 데이터 정제 & 변환

> **학습 시간**: 1시간  
> **난이도**: ⭐⭐⭐  
> **목표**: 실무 데이터의 문제를 해결하고 분석 가능한 형태로 변환합니다.

---

## 📚 학습 내용

1. 결측치 (Missing Values) 처리
2. 중복 데이터 제거
3. 데이터 타입 변환
4. 문자열 처리
5. 날짜/시간 처리
6. apply 함수

---

## 1. 결측치 처리

### 1.1 결측치 확인

```python
import pandas as pd
import numpy as np

# 데이터 불러오기
df = pd.read_csv('../data/sales_data.csv')

# 결측치 개수 확인
print("=== 컬럼별 결측치 개수 ===")
print(df.isnull().sum())

# 결측치 비율
print("\n=== 결측치 비율 ===")
missing_ratio = (df.isnull().sum() / len(df)) * 100
print(missing_ratio[missing_ratio > 0])

# 결측치가 있는 행만 보기
print(f"\n결측치가 있는 행: {df.isnull().any(axis=1).sum()}개")
```

### 1.2 결측치 제거

```python
# 결측치가 하나라도 있는 행 제거
df_dropped = df.dropna()
print(f"원본: {len(df)}행 → 제거 후: {len(df_dropped)}행")

# 특정 컬럼의 결측치만 제거
df_dropped2 = df.dropna(subset=['customer_age'])
print(f"나이 결측치만 제거: {len(df_dropped2)}행")

# 모든 값이 결측치인 행만 제거
df_dropped3 = df.dropna(how='all')
```

### 1.3 결측치 채우기

```python
# 특정 값으로 채우기
df_filled = df.copy()
df_filled['customer_age'] = df_filled['customer_age'].fillna(0)

# 평균값으로 채우기
mean_age = df['customer_age'].mean()
df_filled['customer_age'] = df['customer_age'].fillna(mean_age)
print(f"평균 나이: {mean_age:.1f}세")

# 중앙값으로 채우기
median_age = df['customer_age'].median()
df_filled['customer_age'] = df['customer_age'].fillna(median_age)

# 최빈값으로 채우기 (범주형 데이터)
mode_region = df['region'].mode()[0]
df_filled['region'] = df['region'].fillna(mode_region)
print(f"최빈 지역: {mode_region}")

# 앞/뒤 값으로 채우기 (시계열 데이터)
df_filled['region'] = df['region'].fillna(method='ffill')  # forward fill
# df_filled['region'] = df['region'].fillna(method='bfill')  # backward fill
```

### 1.4 보간법 (Interpolation)

```python
# 선형 보간
df_filled['customer_age'] = df['customer_age'].interpolate(method='linear')

# 결측치 처리 후 확인
print("=== 결측치 처리 후 ===")
print(df_filled.isnull().sum())
```

---

## 2. 중복 데이터 제거

### 2.1 중복 확인

```python
# 완전히 동일한 행 찾기
duplicates = df.duplicated()
print(f"중복 행: {duplicates.sum()}개")

# 중복 행 보기
print(df[duplicates])

# 특정 컬럼 기준 중복 확인
duplicates_order = df.duplicated(subset=['order_id'])
print(f"중복 주문 ID: {duplicates_order.sum()}개")
```

### 2.2 중복 제거

```python
# 첫 번째 행만 남기고 중복 제거
df_unique = df.drop_duplicates()
print(f"원본: {len(df)}행 → 제거 후: {len(df_unique)}행")

# 특정 컬럼 기준 중복 제거
df_unique2 = df.drop_duplicates(subset=['order_id'], keep='first')
# keep='first': 첫 번째 유지
# keep='last': 마지막 유지
# keep=False: 모두 제거
```

---

## 3. 데이터 타입 변환

### 3.1 타입 확인 및 변환

```python
# 현재 타입 확인
print("=== 데이터 타입 ===")
print(df.dtypes)

# 문자열을 숫자로
df['quantity'] = df['quantity'].astype(int)
df['unit_price'] = df['unit_price'].astype(float)

# 문자열을 카테고리로 (메모리 절약)
df['product_category'] = df['product_category'].astype('category')
df['region'] = df['region'].astype('category')

# 날짜 문자열을 datetime으로
df['order_date'] = pd.to_datetime(df['order_date'])
print(f"\n변환 후 타입: {df['order_date'].dtype}")
```

### 3.2 날짜에서 정보 추출

```python
# 날짜 컬럼에서 년, 월, 일 추출
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['day'] = df['order_date'].dt.day
df['weekday'] = df['order_date'].dt.day_name()  # 요일 이름
df['quarter'] = df['order_date'].dt.quarter  # 분기

print("=== 날짜 정보 추출 ===")
print(df[['order_date', 'year', 'month', 'weekday']].head())

# 월별 매출 집계
monthly_sales = df.groupby(['year', 'month'])['final_amount'].sum()
print("\n=== 월별 매출 ===")
print(monthly_sales)
```

---

## 4. 문자열 처리

### 4.1 문자열 메서드

```python
# 문자열 데이터 생성 예시
df['product_name_upper'] = df['product_name'].str.upper()  # 대문자
df['product_name_lower'] = df['product_name'].str.lower()  # 소문자
df['product_name_title'] = df['product_name'].str.title()  # 첫 글자만 대문자

# 공백 제거
df['product_name'] = df['product_name'].str.strip()  # 양쪽 공백
df['product_name'] = df['product_name'].str.lstrip()  # 왼쪽 공백
df['product_name'] = df['product_name'].str.rstrip()  # 오른쪽 공백

# 문자열 치환
df['product_category'] = df['product_category'].str.replace('전자제품', 'Electronics')

# 문자열 포함 여부
mask = df['product_name'].str.contains('상품', na=False)
print(f"'상품'이 포함된 항목: {mask.sum()}개")

# 문자열 분리
# 예: 'CUST0001' -> '0001'
df['customer_number'] = df['customer_id'].str.split('CUST').str[1]

# 문자열 길이
df['name_length'] = df['product_name'].str.len()
```

### 4.2 정규표현식 활용

```python
# 숫자만 추출
df['order_number'] = df['order_id'].str.extract('(\d+)')

# 패턴 검사
pattern = 'ORD\d{6}'  # ORD 뒤 6자리 숫자
valid_orders = df['order_id'].str.match(pattern)
print(f"유효한 주문 ID: {valid_orders.sum()}개")
```

---

## 5. apply 함수로 사용자 정의 변환

### 5.1 apply 기본

```python
# 단일 함수 적용
def calculate_profit_rate(row):
    """수익률 계산"""
    if row['unit_price'] == 0:
        return 0
    profit = row['final_amount'] - (row['quantity'] * row['unit_price'] * 0.6)
    return (profit / row['final_amount']) * 100

df['profit_rate'] = df.apply(calculate_profit_rate, axis=1)
print("=== 수익률 TOP 10 ===")
print(df.nlargest(10, 'profit_rate')[['order_id', 'profit_rate']])
```

### 5.2 Lambda와 함께 사용

```python
# 등급 분류
df['amount_grade'] = df['final_amount'].apply(
    lambda x: 'VIP' if x >= 1000000 else 'Gold' if x >= 500000 else 'Silver'
)

print("=== 금액 등급 분포 ===")
print(df['amount_grade'].value_counts())

# 할인 여부 판단
df['has_discount'] = df['discount_rate'].apply(lambda x: 'Yes' if x > 0 else 'No')
```

### 5.3 여러 컬럼 동시 처리

```python
# 새 컬럼 여러 개 생성
def categorize_customer(row):
    age = row['customer_age']
    amount = row['final_amount']
    
    # 연령대
    if pd.isna(age):
        age_group = 'Unknown'
    elif age < 30:
        age_group = '20대'
    elif age < 40:
        age_group = '30대'
    elif age < 50:
        age_group = '40대'
    else:
        age_group = '50대+'
    
    # 구매 등급
    if amount >= 1000000:
        grade = 'Premium'
    elif amount >= 500000:
        grade = 'Standard'
    else:
        grade = 'Basic'
    
    return pd.Series({'age_group': age_group, 'purchase_grade': grade})

# 적용
df[['age_group', 'purchase_grade']] = df.apply(categorize_customer, axis=1)
print(df[['customer_age', 'age_group', 'final_amount', 'purchase_grade']].head(10))
```

---

## 6. 실무 예제: 데이터 정제 파이프라인

```python
def clean_sales_data(filepath):
    """
    판매 데이터 전체 정제 파이프라인
    """
    print("1️⃣ 데이터 로드...")
    df = pd.read_csv(filepath)
    print(f"   원본 크기: {df.shape}")
    
    print("\n2️⃣ 날짜 변환...")
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['weekday'] = df['order_date'].dt.day_name()
    
    print("\n3️⃣ 결측치 처리...")
    # 나이: 평균값으로 채우기
    df['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
    # 지역: 최빈값으로 채우기
    df['region'].fillna(df['region'].mode()[0], inplace=True)
    print(f"   결측치 처리 후: {df.isnull().sum().sum()}개")
    
    print("\n4️⃣ 중복 제거...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['order_id'])
    print(f"   제거된 행: {original_len - len(df)}개")
    
    print("\n5️⃣ 파생 변수 생성...")
    # 수익 계산 (단순화)
    df['profit'] = df['final_amount'] * 0.3
    
    # 연령대 분류
    df['age_group'] = pd.cut(
        df['customer_age'], 
        bins=[0, 30, 40, 50, 60, 100],
        labels=['20대', '30대', '40대', '50대', '60대+']
    )
    
    # 가격대 분류
    df['price_range'] = pd.cut(
        df['final_amount'],
        bins=[0, 100000, 500000, 1000000, float('inf')],
        labels=['저가', '중가', '고가', '프리미엄']
    )
    
    print("\n✅ 정제 완료!")
    print(f"   최종 크기: {df.shape}")
    print(f"   새로 생성된 컬럼: {len(df.columns) - 15}개")
    
    return df

# 실행
df_clean = clean_sales_data('../data/sales_data.csv')
print("\n=== 정제된 데이터 샘플 ===")
print(df_clean.head())

# 정제된 데이터 저장
df_clean.to_csv('../output/cleaned_sales_data.csv', index=False, encoding='utf-8-sig')
print("\n💾 정제된 데이터 저장 완료!")
```

---

## 💪 실습 문제

### 문제 1: 고객 데이터 정제

`customer_data.csv`를 불러와서 다음 작업을 수행하세요:
1. 결측치 확인 및 적절한 방법으로 처리
2. 가입 날짜를 datetime으로 변환
3. 고객 등급별 평균 구매액 계산

```python
# TODO: 코드 작성
```

### 문제 2: 이상치 탐지

판매 데이터에서 이상치를 찾으세요:
1. final_amount가 0보다 작거나 1000만원 초과인 경우
2. quantity가 50개 이상인 경우
3. 해당 이상치 제거 후 통계 비교

```python
# TODO: 코드 작성
```

---

## 📝 정리

✅ **결측치 처리**: dropna, fillna, interpolate  
✅ **중복 제거**: drop_duplicates  
✅ **타입 변환**: astype, to_datetime  
✅ **문자열 처리**: str 접근자  
✅ **날짜 처리**: dt 접근자  
✅ **apply 함수**: 사용자 정의 변환  
✅ **실무 파이프라인**: 전체 정제 프로세스

---

**수고하셨습니다! 🎉**
