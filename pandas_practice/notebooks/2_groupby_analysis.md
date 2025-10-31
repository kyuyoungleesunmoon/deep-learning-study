# 📚 2교시: groupby()를 활용한 카테고리별 집계

> **학습 목표:** `groupby()` 메서드를 활용하여 카테고리별로 데이터를 그룹화하고 집계하는 방법을 학습합니다.  
> **소요 시간:** 1시간  
> **사용 데이터:** 판매 데이터셋 (sales_data.csv)

---

## 🧠 이론 설명

### 2.1 groupby()란?

**`groupby()`**는 Pandas에서 데이터를 특정 기준에 따라 그룹으로 나누고, 각 그룹에 대해 집계 연산을 수행하는 강력한 메서드입니다.

#### 📊 실생활 비유
레스토랑에서 일일 매출을 분석한다고 생각해보세요:
- **그룹화**: 메뉴 카테고리별 (음료, 식사, 디저트)
- **집계**: 각 카테고리의 총 매출, 평균 단가, 판매 건수 계산

### 2.2 groupby()의 동작 원리 (Split-Apply-Combine)

```
1. Split (분할): 데이터를 그룹으로 나눔
2. Apply (적용): 각 그룹에 함수 적용
3. Combine (결합): 결과를 하나의 데이터 구조로 결합
```

```python
# 기본 문법
df.groupby('그룹컬럼')['집계컬럼'].집계함수()

# 예시
df.groupby('카테고리')['판매액'].sum()  # 카테고리별 판매액 합계
```

### 2.3 주요 집계 함수

| 함수 | 설명 | 예시 |
|------|------|------|
| `sum()` | 합계 | 총 판매액 |
| `mean()` | 평균 | 평균 가격 |
| `count()` | 개수 | 주문 건수 |
| `max()` | 최댓값 | 최고 가격 |
| `min()` | 최솟값 | 최저 가격 |
| `median()` | 중앙값 | 가격 중앙값 |
| `std()` | 표준편차 | 가격 분산 정도 |
| `agg()` | 여러 함수 동시 적용 | 합계, 평균, 개수 한번에 |

### 2.4 다중 그룹화

여러 컬럼을 기준으로 그룹화할 수 있습니다:

```python
# 지역별, 카테고리별 집계
df.groupby(['지역', '카테고리'])['판매액'].sum()
```

---

## 💻 실습 코드

### 환경 설정 및 데이터 로드

```python
# 필수 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
sales = pd.read_csv('../data/sales_data.csv')

print("=" * 80)
print("📊 판매 데이터셋 기본 정보")
print("=" * 80)
print(f"데이터 크기: {sales.shape[0]} 행, {sales.shape[1]} 열")
print(f"\n컬럼명: {list(sales.columns)}")
print(f"\n데이터 타입:")
print(sales.dtypes)
print(f"\n상위 5개 데이터:")
print(sales.head())
print(f"\n기초 통계:")
print(sales.describe())
```

### 실습 1: 단일 컬럼 그룹화 - 기본

```python
print("\n" + "=" * 80)
print("📌 실습 1: 카테고리별 평균 판매가격")
print("=" * 80)

# 카테고리별 평균 판매가격
avg_price_by_category = sales.groupby('카테고리')['판매가격'].mean()

print("\n카테고리별 평균 판매가격:")
print(avg_price_by_category)
print(f"\n데이터 타입: {type(avg_price_by_category)}")  # Series

# DataFrame으로 변환
avg_price_df = sales.groupby('카테고리')['판매가격'].mean().reset_index()
avg_price_df.columns = ['카테고리', '평균판매가격']
print("\nDataFrame 형태로 변환:")
print(avg_price_df)
```

### 실습 2: 여러 집계 함수 적용

```python
print("\n" + "=" * 80)
print("📌 실습 2: 카테고리별 다양한 통계")
print("=" * 80)

# 카테고리별 판매가격 통계 (sum, mean, count, max, min)
category_stats = sales.groupby('카테고리')['판매가격'].agg([
    'sum',      # 총합
    'mean',     # 평균
    'count',    # 개수
    'max',      # 최댓값
    'min',      # 최솟값
    'std'       # 표준편차
])

print("\n카테고리별 판매가격 통계:")
print(category_stats.round(2))

# 열 이름을 한글로 변경
category_stats_kr = category_stats.copy()
category_stats_kr.columns = ['총판매액', '평균가격', '건수', '최고가격', '최저가격', '표준편차']
print("\n[한글 컬럼명으로 표시]")
print(category_stats_kr.round(2))
```

### 실습 3: 여러 컬럼 동시 집계

```python
print("\n" + "=" * 80)
print("📌 실습 3: 카테고리별 판매가격 & 판매량 집계")
print("=" * 80)

# 카테고리별로 판매가격과 판매량을 동시에 집계
multi_col_agg = sales.groupby('카테고리')[['판매가격', '판매량']].agg({
    '판매가격': ['mean', 'sum'],
    '판매량': ['sum', 'mean']
})

print("\n카테고리별 판매가격 & 판매량 통계:")
print(multi_col_agg.round(2))

# MultiIndex 컬럼을 단순화
multi_col_agg.columns = ['평균가격', '총판매액', '총판매량', '평균판매량']
print("\n[단순화된 컬럼명]")
print(multi_col_agg.round(2))
```

### 실습 4: 다중 그룹화 (2개 이상의 컬럼)

```python
print("\n" + "=" * 80)
print("📌 실습 4: 지역별 + 카테고리별 집계")
print("=" * 80)

# 지역과 카테고리로 다중 그룹화
region_category_sales = sales.groupby(['지역', '카테고리'])['판매가격'].agg([
    'sum',
    'mean',
    'count'
]).reset_index()

region_category_sales.columns = ['지역', '카테고리', '총판매액', '평균가격', '판매건수']

print("\n지역별 + 카테고리별 판매 통계:")
print(region_category_sales)

# 서울 지역만 필터링
print("\n[서울 지역만 필터링]")
seoul_only = region_category_sales[region_category_sales['지역'] == '서울']
print(seoul_only)
```

### 실습 5: 조건부 집계

```python
print("\n" + "=" * 80)
print("📌 실습 5: 조건부 집계 - 고가 제품 분석")
print("=" * 80)

# 판매가격이 500,000원 이상인 제품만 필터링
high_price_products = sales[sales['판매가격'] >= 500000]

# 카테고리별 고가 제품 통계
high_price_stats = high_price_products.groupby('카테고리').agg({
    '판매가격': ['count', 'mean', 'sum'],
    '판매량': 'sum'
})

print("\n고가 제품 (50만원 이상) 카테고리별 통계:")
print(high_price_stats)

# 전체 판매액 대비 고가 제품 판매액 비율
total_sales = sales['판매가격'].sum()
high_price_sales = high_price_products['판매가격'].sum()
ratio = (high_price_sales / total_sales) * 100

print(f"\n💡 인사이트:")
print(f"전체 판매액: {total_sales:,}원")
print(f"고가 제품 판매액: {high_price_sales:,}원")
print(f"고가 제품 비율: {ratio:.2f}%")
```

### 실습 6: 제품명별 총 판매량 Top 5

```python
print("\n" + "=" * 80)
print("📌 실습 6: 제품명별 판매량 Top 5")
print("=" * 80)

# 제품명별 총 판매량 계산
product_sales = sales.groupby('제품명')['판매량'].sum().sort_values(ascending=False)

print("\n제품명별 총 판매량 Top 5:")
print(product_sales.head())

# 시각화
plt.figure(figsize=(10, 6))
product_sales.head(10).plot(kind='barh', color='skyblue')
plt.title('제품명별 판매량 Top 10', fontsize=16, fontweight='bold')
plt.xlabel('총 판매량', fontsize=12)
plt.ylabel('제품명', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../output/top10_products_by_sales.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/top10_products_by_sales.png")
plt.show()
```

### 실습 7: 월별 판매 추이 분석

```python
print("\n" + "=" * 80)
print("📌 실습 7: 월별 판매 추이")
print("=" * 80)

# 판매일을 datetime으로 변환
sales['판매일'] = pd.to_datetime(sales['판매일'])

# 월 추출
sales['월'] = sales['판매일'].dt.month

# 월별 총 판매액과 평균 판매가격
monthly_sales = sales.groupby('월').agg({
    '판매가격': ['sum', 'mean', 'count']
}).reset_index()

monthly_sales.columns = ['월', '총판매액', '평균가격', '판매건수']

print("\n월별 판매 통계:")
print(monthly_sales)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 그래프 1: 월별 총 판매액
axes[0].plot(monthly_sales['월'], monthly_sales['총판매액'], 
             marker='o', linewidth=2, markersize=8, color='green')
axes[0].set_title('월별 총 판매액 추이', fontsize=14, fontweight='bold')
axes[0].set_xlabel('월', fontsize=12)
axes[0].set_ylabel('총 판매액 (원)', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(monthly_sales['월'])

# 그래프 2: 월별 판매 건수
axes[1].bar(monthly_sales['월'], monthly_sales['판매건수'], color='coral', alpha=0.7)
axes[1].set_title('월별 판매 건수', fontsize=14, fontweight='bold')
axes[1].set_xlabel('월', fontsize=12)
axes[1].set_ylabel('판매 건수', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xticks(monthly_sales['월'])

plt.tight_layout()
plt.savefig('../output/monthly_sales_trend.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/monthly_sales_trend.png")
plt.show()
```

### 실습 8: 실무 예제 - 지역별 매출 순위

```python
print("\n" + "=" * 80)
print("📌 실습 8: 지역별 매출 순위 및 점유율")
print("=" * 80)

# 지역별 총 판매액
region_sales = sales.groupby('지역')['판매가격'].sum().sort_values(ascending=False)

# 점유율 계산
total = region_sales.sum()
region_share = (region_sales / total * 100).round(2)

# DataFrame으로 정리
region_analysis = pd.DataFrame({
    '총판매액': region_sales,
    '점유율(%)': region_share,
    '순위': range(1, len(region_sales) + 1)
})

print("\n지역별 매출 순위:")
print(region_analysis)

# 원형 차트로 시각화
plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(range(len(region_sales)))
plt.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%',
        startangle=90, colors=colors, textprops={'fontsize': 12})
plt.title('지역별 매출 점유율', fontsize=16, fontweight='bold', pad=20)
plt.savefig('../output/region_sales_share.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/region_sales_share.png")
plt.show()
```

### 실습 9: 사용자 정의 집계 함수

```python
print("\n" + "=" * 80)
print("📌 실습 9: 사용자 정의 집계 함수")
print("=" * 80)

# 범위(최댓값 - 최솟값) 계산 함수
def price_range(series):
    return series.max() - series.min()

# 변동계수(CV) 계산 함수
def coefficient_of_variation(series):
    return (series.std() / series.mean()) * 100

# 카테고리별 가격 범위와 변동계수
custom_agg = sales.groupby('카테고리')['판매가격'].agg([
    ('최댓값', 'max'),
    ('최솟값', 'min'),
    ('가격범위', price_range),
    ('표준편차', 'std'),
    ('변동계수(%)', coefficient_of_variation)
])

print("\n카테고리별 가격 분석:")
print(custom_agg.round(2))

print("\n💡 변동계수(CV) 해석:")
print("- CV가 높을수록 가격 변동성이 큼")
print("- CV < 15%: 낮은 변동성")
print("- CV 15~30%: 중간 변동성")
print("- CV > 30%: 높은 변동성")
```

---

## 🧩 퀴즈

### 문제 1
다음 코드의 결과로 얻어지는 데이터 타입은?

```python
result = df.groupby('category')['price'].mean()
```

1. DataFrame
2. Series
3. List
4. Dictionary

<details>
<summary>정답 보기</summary>

**정답: 2번 Series**

**해설:**
- `groupby()` 후 단일 컬럼을 선택하고 집계 함수를 적용하면 **Series**가 반환됩니다.
- DataFrame으로 변환하려면 `.reset_index()` 또는 `[[컬럼명]]` 사용:
  ```python
  df.groupby('category')[['price']].mean()  # DataFrame
  df.groupby('category')['price'].mean().reset_index()  # DataFrame
  ```
</details>

---

### 문제 2
다음 중 `groupby()`와 함께 사용할 수 없는 집계 함수는?

1. `sum()`
2. `mean()`
3. `concat()`
4. `count()`

<details>
<summary>정답 보기</summary>

**정답: 3번 `concat()`**

**해설:**
- `concat()`은 DataFrame을 연결하는 함수로, `groupby()`의 집계 함수가 아닙니다.
- `groupby()`에서 주로 사용하는 집계 함수:
  - 수치 연산: `sum()`, `mean()`, `median()`, `std()`, `var()`
  - 개수: `count()`, `size()`
  - 극값: `max()`, `min()`
  - 기타: `first()`, `last()`, `agg()`
</details>

---

### 문제 3
다음 코드의 출력 결과를 설명하시오:

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A'],
    'value': [10, 20, 30, 40, 50]
})
result = df.groupby('category')['value'].sum()
```

<details>
<summary>정답 보기</summary>

**정답:**
```
category
A    90
B    60
Name: value, dtype: int64
```

**해설:**
- 'A' 카테고리: 10 + 30 + 50 = 90
- 'B' 카테고리: 20 + 40 = 60
- `groupby()`는 동일한 카테고리를 하나의 그룹으로 묶어 집계합니다.
- 결과는 Series 타입으로 반환되며, index는 그룹 키(category)입니다.
</details>

---

### 문제 4
다중 그룹화에서 여러 개의 컬럼을 기준으로 그룹화하려면?

```python
# 지역과 카테고리로 동시에 그룹화
df.groupby(______)['sales'].sum()
```

빈칸에 들어갈 올바른 코드는?

1. `'region', 'category'`
2. `['region', 'category']`
3. `('region' and 'category')`
4. `{'region': 'category'}`

<details>
<summary>정답 보기</summary>

**정답: 2번 `['region', 'category']`**

**해설:**
- 다중 그룹화는 **리스트 형태**로 컬럼명을 전달합니다:
  ```python
  df.groupby(['region', 'category'])['sales'].sum()
  ```
- 이렇게 하면 'region'과 'category'의 모든 조합별로 그룹이 생성됩니다.
- 결과는 MultiIndex를 가진 Series가 됩니다.

**예시:**
```python
# 서울-전자제품, 서울-의류, 부산-전자제품, 부산-의류 등으로 그룹화
region  category
서울     전자제품      1000000
        의류          500000
부산     전자제품       800000
        의류          400000
```
</details>

---

### 문제 5
다음 실무 상황에서 적절한 groupby 코드를 작성하시오:

**상황:** 온라인 쇼핑몰의 주문 데이터에서 '고객등급'별로 '주문금액'의 평균, 합계, 건수를 동시에 구하려고 합니다.

<details>
<summary>정답 보기</summary>

**정답:**
```python
result = df.groupby('고객등급')['주문금액'].agg(['mean', 'sum', 'count'])
```

또는

```python
result = df.groupby('고객등급')['주문금액'].agg({
    '주문금액': ['mean', 'sum', 'count']
})
```

또는 열 이름을 명확히 지정:

```python
result = df.groupby('고객등급')['주문금액'].agg([
    ('평균주문금액', 'mean'),
    ('총주문금액', 'sum'),
    ('주문건수', 'count')
])
```

**해설:**
- `.agg()` 메서드를 사용하면 여러 집계 함수를 동시에 적용할 수 있습니다.
- 리스트로 함수명을 전달하면 각 함수가 순차적으로 적용됩니다.
- 튜플 형태로 `(새로운_열이름, 함수명)`을 지정하면 결과 컬럼명을 커스터마이징할 수 있습니다.

**출력 예시:**
```
고객등급      평균주문금액    총주문금액    주문건수
VIP        250000      5000000      20
골드        150000      3000000      20
실버         80000      1600000      20
일반         50000      1000000      20
```
</details>

---

## ✅ 2교시 학습 완료 체크리스트

- [ ] groupby()의 개념과 동작 원리 (Split-Apply-Combine) 이해
- [ ] 기본 집계 함수 (sum, mean, count, max, min) 사용
- [ ] agg()를 활용한 다중 집계 함수 적용
- [ ] 다중 그룹화 (2개 이상의 컬럼) 수행
- [ ] 조건부 집계 및 필터링
- [ ] 실무 예제 (지역별 매출 분석, 제품 순위) 실습
- [ ] 사용자 정의 집계 함수 작성
- [ ] 퀴즈 5문제 모두 풀이 완료

---

**이전 학습:** [1교시 - 결측치 처리](./1_missing_values.md)  
**다음 학습:** [3교시 - 데이터 병합 (merge/join)](./3_merge_join.md)

**학습 완료일:** _____________  
**소요 시간:** _____________  
**이해도 (1~5):** ⭐⭐⭐⭐⭐
