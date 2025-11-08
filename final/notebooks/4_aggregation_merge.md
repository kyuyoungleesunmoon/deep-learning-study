# 4교시: Pandas 심화 - 집계 & 병합

> **학습 시간**: 1시간  
> **난이도**: ⭐⭐⭐  
> **목표**: 데이터를 그룹화하여 집계하고 여러 데이터를 병합하는 방법을 익힙니다.

---

## 📚 학습 내용

1. GroupBy 집계
2. Pivot Table
3. 데이터 병합 (Merge)
4. 데이터 연결 (Concat)

---

## 1. GroupBy 집계

### 1.1 기본 GroupBy

```python
import pandas as pd

# 데이터 불러오기
df = pd.read_csv('../data/sales_data.csv')
df['order_date'] = pd.to_datetime(df['order_date'])

# 카테고리별 총 매출
category_sales = df.groupby('product_category')['final_amount'].sum()
print("=== 카테고리별 총 매출 ===")
print(category_sales.sort_values(ascending=False))

# 여러 통계량 한 번에
category_stats = df.groupby('product_category')['final_amount'].agg(['sum', 'mean', 'count'])
print("\n=== 카테고리별 상세 통계 ===")
print(category_stats)
```

### 1.2 다중 컬럼 그룹화

```python
# 지역 + 카테고리별 집계
region_category = df.groupby(['region', 'product_category'])['final_amount'].agg({
    '총매출': 'sum',
    '평균': 'mean',
    '건수': 'count'
})
print("=== 지역 + 카테고리별 매출 ===")
print(region_category.sort_values('총매출', ascending=False).head(10))
```

### 1.3 사용자 정의 집계 함수

```python
def sales_range(x):
    """최대값 - 최소값"""
    return x.max() - x.min()

# 여러 함수 동시 적용
agg_result = df.groupby('product_category')['final_amount'].agg([
    '총합계': 'sum',
    '평균': 'mean',
    '최대': 'max',
    '최소': 'min',
    '범위': sales_range
])
print(agg_result)
```

### 1.4 transform과 filter

```python
# 각 그룹의 평균을 모든 행에 추가
df['category_avg'] = df.groupby('product_category')['final_amount'].transform('mean')

# 각 카테고리 평균과 비교
df['vs_avg'] = df['final_amount'] - df['category_avg']
print(df[['product_category', 'final_amount', 'category_avg', 'vs_avg']].head(20))

# 평균 매출이 100만원 이상인 카테고리만 필터링
high_avg = df.groupby('product_category').filter(lambda x: x['final_amount'].mean() >= 1000000)
print(f"\n고매출 카테고리 데이터: {len(high_avg)}건")
```

---

## 2. Pivot Table

### 2.1 기본 Pivot Table

```python
# 지역(행) × 카테고리(열) 매출 합계
pivot = df.pivot_table(
    values='final_amount',
    index='region',
    columns='product_category',
    aggfunc='sum',
    fill_value=0
)
print("=== 지역 × 카테고리 Pivot ===")
print(pivot)
```

### 2.2 다중 집계 함수

```python
# 여러 통계량 동시에
pivot_multi = df.pivot_table(
    values='final_amount',
    index='region',
    columns='product_category',
    aggfunc=['sum', 'mean', 'count'],
    fill_value=0
)
print("\n=== 다중 집계 Pivot ===")
print(pivot_multi)
```

### 2.3 날짜 기반 Pivot

```python
# 월별 매출 추이
df['month'] = df['order_date'].dt.to_period('M')
monthly_pivot = df.pivot_table(
    values='final_amount',
    index='month',
    columns='product_category',
    aggfunc='sum',
    fill_value=0
)
print("\n=== 월별 카테고리 매출 ===")
print(monthly_pivot.head(12))
```

---

## 3. 데이터 병합 (Merge)

### 3.1 Inner Join

```python
# 고객 데이터와 판매 데이터 병합
df_customer = pd.read_csv('../data/customer_data.csv')
df_sales = pd.read_csv('../data/sales_data.csv')

# customer_id 기준으로 병합
merged = pd.merge(
    df_sales,
    df_customer[['customer_id', 'customer_name', 'member_type']],
    on='customer_id',
    how='inner'
)
print("=== 병합 결과 ===")
print(f"원본 판매 데이터: {len(df_sales)}건")
print(f"병합 후: {len(merged)}건")
print(merged.head())
```

### 3.2 Left/Right/Outer Join

```python
# Left Join: 왼쪽 데이터 모두 유지
left_merged = pd.merge(df_sales, df_customer, on='customer_id', how='left')

# Right Join: 오른쪽 데이터 모두 유지
right_merged = pd.merge(df_sales, df_customer, on='customer_id', how='right')

# Outer Join: 양쪽 모두 유지
outer_merged = pd.merge(df_sales, df_customer, on='customer_id', how='outer')

print(f"Left Join: {len(left_merged)}건")
print(f"Right Join: {len(right_merged)}건")
print(f"Outer Join: {len(outer_merged)}건")
```

### 3.3 여러 키로 병합

```python
# 복합 키로 병합 (예시)
# merged = pd.merge(df1, df2, on=['key1', 'key2'], how='inner')
```

---

## 4. 데이터 연결 (Concat)

### 4.1 세로로 연결

```python
# 여러 DataFrame을 위아래로 쌓기
df1 = df_sales.head(100)
df2 = df_sales.tail(100)
concatenated = pd.concat([df1, df2], ignore_index=True)
print(f"연결 후: {len(concatenated)}건")
```

### 4.2 가로로 연결

```python
# 열 방향으로 연결
df_left = df_sales[['order_id', 'final_amount']]
df_right = df_sales[['customer_id', 'product_category']]
side_by_side = pd.concat([df_left, df_right], axis=1)
print(side_by_side.head())
```

---

## 💪 실습 예제

### 예제 1: 월별 카테고리 매출 분석

```python
# 월별, 카테고리별 매출 분석
df['year_month'] = df['order_date'].dt.to_period('M')
monthly_analysis = df.groupby(['year_month', 'product_category']).agg({
    'final_amount': ['sum', 'mean', 'count'],
    'quantity': 'sum'
})
monthly_analysis.columns = ['총매출', '평균매출', '거래건수', '총수량']
print("=== 월별 카테고리 분석 ===")
print(monthly_analysis.head(20))
```

### 예제 2: VIP 고객 분석

```python
# 고객별 총 구매액 계산
customer_total = df_sales.groupby('customer_id').agg({
    'final_amount': 'sum',
    'order_id': 'count'
}).reset_index()
customer_total.columns = ['customer_id', 'total_purchase', 'order_count']

# 고객 정보와 병합
customer_analysis = pd.merge(customer_total, df_customer, on='customer_id', how='left')

# VIP 고객 추출 (500만원 이상)
vip = customer_analysis[customer_analysis['total_purchase'] >= 5000000]
print(f"VIP 고객: {len(vip)}명")
print(vip.sort_values('total_purchase', ascending=False).head(10))
```

---

**수고하셨습니다! 🎉**
