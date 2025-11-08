# 2교시: Pandas 기초 - 데이터 불러오기 & 탐색

> **학습 시간**: 1시간  
> **난이도**: ⭐⭐  
> **목표**: Pandas DataFrame을 사용하여 데이터를 불러오고 기본 탐색 방법을 익힙니다.

---

## 📚 학습 내용

1. Pandas 소개 및 설치
2. DataFrame 생성
3. 데이터 불러오기 (CSV)
4. 기본 탐색 메서드
5. 행/열 선택 (loc, iloc)
6. 조건 필터링
7. 데이터 정렬

---

## 1. Pandas 소개

**Pandas**는 Python에서 데이터 분석을 위한 가장 인기 있는 라이브러리입니다.

### 특징
- 📊 **DataFrame**: 표 형태의 데이터 구조
- 🔍 **데이터 탐색**: 쉽고 직관적인 데이터 조회
- 🛠️ **데이터 처리**: 필터링, 집계, 병합 등
- 📁 **파일 입출력**: CSV, Excel, SQL 등 지원

### 설치 및 임포트

```python
# 설치 (이미 설치되어 있으면 생략)
# pip install pandas

# 임포트
import pandas as pd
import numpy as np

print(f"Pandas 버전: {pd.__version__}")
```

---

## 2. DataFrame 생성

### 2.1 딕셔너리로 생성

```python
# 판매 데이터 생성
data = {
    '상품명': ['노트북', '마우스', '키보드', '모니터', '헤드셋'],
    '판매가': [1200000, 25000, 85000, 350000, 120000],
    '원가': [900000, 15000, 50000, 250000, 80000],
    '재고': [15, 120, 85, 42, 67]
}

df = pd.DataFrame(data)
print("=== 판매 데이터 ===")
print(df)
```

출력:
```
    상품명      판매가     원가   재고
0   노트북  1200000  900000   15
1   마우스    25000   15000  120
2  키보드    85000   50000   85
3  모니터   350000  250000   42
4  헤드셋   120000   80000   67
```

### 2.2 리스트로 생성

```python
# 2차원 리스트
data_list = [
    ['김철수', 32, '영업팀', 3500000],
    ['이영희', 28, '개발팀', 4200000],
    ['박민수', 35, '기획팀', 3800000]
]

columns = ['이름', '나이', '부서', '급여']
df_emp = pd.DataFrame(data_list, columns=columns)
print("\n=== 직원 데이터 ===")
print(df_emp)
```

---

## 3. 데이터 불러오기

### 3.1 CSV 파일 읽기

```python
# 실습 데이터 불러오기
df_sales = pd.read_csv('../data/sales_data.csv')

print("=== 판매 데이터 불러오기 ===")
print(f"데이터 크기: {df_sales.shape}")  # (행, 열)
print(f"행 개수: {df_sales.shape[0]}")
print(f"열 개수: {df_sales.shape[1]}")
```

### 3.2 인코딩 문제 해결

```python
# 한글이 깨질 경우
df_sales = pd.read_csv('../data/sales_data.csv', encoding='utf-8-sig')

# 또는
# df_sales = pd.read_csv('../data/sales_data.csv', encoding='cp949')
```

---

## 4. 기본 탐색 메서드

### 4.1 head() / tail()

```python
# 상위 5개 행
print("=== 상위 5개 데이터 ===")
print(df_sales.head())

# 상위 3개 행
print("\n=== 상위 3개 데이터 ===")
print(df_sales.head(3))

# 하위 5개 행
print("\n=== 하위 5개 데이터 ===")
print(df_sales.tail())
```

### 4.2 info()

```python
# 데이터 구조 정보
print("\n=== 데이터 구조 정보 ===")
print(df_sales.info())
```

출력 예시:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 15 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   order_id          1000 non-null   object 
 1   order_date        1000 non-null   object 
 2   customer_id       1000 non-null   object 
 3   product_category  1000 non-null   object 
 ...
```

### 4.3 describe()

```python
# 수치형 컬럼 통계
print("\n=== 기술 통계 ===")
print(df_sales.describe())
```

출력 예시:
```
         quantity    unit_price    customer_age   total_amount
count  1000.000000  1.000000e+03     950.000000  1.000000e+03
mean      4.998000  2.500000e+05      44.500000  1.250000e+06
std       2.582000  1.410000e+05      14.300000  7.200000e+05
min       1.000000  1.000000e+04      20.000000  1.000000e+04
25%       3.000000  1.300000e+05      32.000000  6.500000e+05
...
```

### 4.4 기타 유용한 메서드

```python
# 컬럼명 확인
print(f"컬럼명: {df_sales.columns.tolist()}")

# 데이터 타입 확인
print("\n=== 데이터 타입 ===")
print(df_sales.dtypes)

# 결측치 확인
print("\n=== 결측치 개수 ===")
print(df_sales.isnull().sum())

# 고유값 개수
print(f"\n상품 카테고리 종류: {df_sales['product_category'].nunique()}개")
print(f"카테고리 목록: {df_sales['product_category'].unique()}")
```

---

## 5. 행/열 선택

### 5.1 컬럼 선택

```python
# 단일 컬럼 (Series 반환)
category = df_sales['product_category']
print(f"데이터 타입: {type(category)}")
print(category.head())

# 여러 컬럼 (DataFrame 반환)
selected = df_sales[['order_id', 'product_category', 'final_amount']]
print("\n=== 선택한 컬럼 ===")
print(selected.head())
```

### 5.2 loc (라벨 기반 인덱싱)

```python
# 특정 행 선택
print("\n=== 첫 번째 행 ===")
print(df_sales.loc[0])

# 행 범위 선택 (끝 포함!)
print("\n=== 0~4번 행 ===")
print(df_sales.loc[0:4])

# 행과 열 동시 선택
print("\n=== 0~2번 행의 특정 컬럼 ===")
print(df_sales.loc[0:2, ['product_category', 'final_amount']])
```

### 5.3 iloc (위치 기반 인덱싱)

```python
# 위치로 행 선택
print("\n=== 첫 번째 행 (iloc) ===")
print(df_sales.iloc[0])

# 행 범위 선택 (끝 미포함!)
print("\n=== 0~4번 행 (iloc) ===")
print(df_sales.iloc[0:5])

# 위치로 행과 열 선택
print("\n=== 0~2번 행, 0~3번 컬럼 ===")
print(df_sales.iloc[0:3, 0:4])

# 특정 위치의 값 하나만 선택
value = df_sales.iloc[0, 3]  # 0번 행, 3번 컬럼
print(f"\n특정 값: {value}")
```

---

## 6. 조건 필터링

### 6.1 단일 조건

```python
# 최종 금액이 100만원 이상인 거래
high_value = df_sales[df_sales['final_amount'] >= 1000000]
print(f"100만원 이상 거래: {len(high_value)}건")
print(high_value.head())

# 특정 카테고리만 선택
electronics = df_sales[df_sales['product_category'] == '전자제품']
print(f"\n전자제품 거래: {len(electronics)}건")
```

### 6.2 다중 조건 (AND)

```python
# 전자제품이면서 100만원 이상
condition = (df_sales['product_category'] == '전자제품') & \
            (df_sales['final_amount'] >= 1000000)
filtered = df_sales[condition]
print(f"\n전자제품 & 100만원 이상: {len(filtered)}건")
print(filtered.head())
```

### 6.3 다중 조건 (OR)

```python
# 전자제품 또는 가구
condition = (df_sales['product_category'] == '전자제품') | \
            (df_sales['product_category'] == '가구')
filtered = df_sales[condition]
print(f"\n전자제품 또는 가구: {len(filtered)}건")
```

### 6.4 isin() 메서드

```python
# 여러 카테고리 동시 선택
categories = ['전자제품', '가구', '화장품']
filtered = df_sales[df_sales['product_category'].isin(categories)]
print(f"\n선택한 카테고리: {len(filtered)}건")
```

### 6.5 문자열 조건

```python
# 고객ID가 'CUST0001'로 시작하는 거래
condition = df_sales['customer_id'].str.startswith('CUST0001')
filtered = df_sales[condition]
print(f"\nCUST0001로 시작: {len(filtered)}건")

# 지역에 '서울'이 포함된 거래 (결측치 제외)
condition = df_sales['region'].str.contains('서울', na=False)
filtered = df_sales[condition]
print(f"서울 지역: {len(filtered)}건")
```

---

## 7. 데이터 정렬

### 7.1 단일 컬럼 정렬

```python
# 최종 금액 기준 내림차순
sorted_desc = df_sales.sort_values('final_amount', ascending=False)
print("=== 금액 높은 순 TOP 10 ===")
print(sorted_desc[['order_id', 'product_category', 'final_amount']].head(10))

# 오름차순
sorted_asc = df_sales.sort_values('final_amount', ascending=True)
print("\n=== 금액 낮은 순 TOP 10 ===")
print(sorted_asc[['order_id', 'product_category', 'final_amount']].head(10))
```

### 7.2 다중 컬럼 정렬

```python
# 카테고리별로 정렬한 후, 같은 카테고리 내에서는 금액 순
sorted_multi = df_sales.sort_values(
    ['product_category', 'final_amount'], 
    ascending=[True, False]
)
print("\n=== 카테고리별 금액 높은 순 ===")
print(sorted_multi[['product_category', 'final_amount']].head(20))
```

### 7.3 인덱스 재설정

```python
# 정렬 후 인덱스 재설정
sorted_df = df_sales.sort_values('final_amount', ascending=False)
sorted_df = sorted_df.reset_index(drop=True)
print("\n=== 인덱스 재설정 ===")
print(sorted_df.head())
```

---

## 8. 실무 예제

### 예제 1: 카테고리별 매출 분석

```python
# 실습 데이터 불러오기
df = pd.read_csv('../data/sales_data.csv')

print("=== 카테고리별 매출 분석 ===")
print(f"전체 거래: {len(df):,}건")
print(f"총 매출: {df['final_amount'].sum():,}원\n")

# 각 카테고리별 통계
categories = df['product_category'].unique()
for cat in categories:
    cat_data = df[df['product_category'] == cat]
    total = cat_data['final_amount'].sum()
    count = len(cat_data)
    avg = cat_data['final_amount'].mean()
    
    print(f"{cat}:")
    print(f"  거래 건수: {count:,}건")
    print(f"  총 매출: {total:,}원")
    print(f"  평균 거래액: {avg:,.0f}원")
    print()
```

### 예제 2: TOP 고객 분석

```python
# 고객별 총 구매액 계산
customer_total = df.groupby('customer_id')['final_amount'].agg(['sum', 'count'])
customer_total.columns = ['총구매액', '구매횟수']
customer_total = customer_total.sort_values('총구매액', ascending=False)

print("=== TOP 10 고객 ===")
print(customer_total.head(10))

# VIP 고객 (500만원 이상 구매)
vip_customers = customer_total[customer_total['총구매액'] >= 5000000]
print(f"\nVIP 고객 수: {len(vip_customers)}명")
```

### 예제 3: 지역별 매출 현황

```python
# 지역별 집계 (결측치 제외)
df_region = df[df['region'].notna()]

print("=== 지역별 매출 ===")
region_sales = df_region.groupby('region')['final_amount'].agg(['sum', 'mean', 'count'])
region_sales.columns = ['총매출', '평균매출', '거래건수']
region_sales = region_sales.sort_values('총매출', ascending=False)

print(region_sales)
```

### 예제 4: 연령대별 구매 패턴

```python
# 연령대 구간 생성 (결측치 제외)
df_age = df[df['customer_age'].notna()].copy()

# 연령대 분류 함수
def classify_age(age):
    if age < 30:
        return '20대'
    elif age < 40:
        return '30대'
    elif age < 50:
        return '40대'
    elif age < 60:
        return '50대'
    else:
        return '60대 이상'

df_age['연령대'] = df_age['customer_age'].apply(classify_age)

print("=== 연령대별 구매 패턴 ===")
age_pattern = df_age.groupby('연령대')['final_amount'].agg(['mean', 'count'])
age_pattern.columns = ['평균구매액', '구매건수']
print(age_pattern)
```

---

## 💪 실습 문제

### 문제 1: 고액 거래 분석

100만원 이상 거래에 대해 다음을 분석하세요:
1. 거래 건수
2. 총 매출
3. 가장 많이 팔린 카테고리 TOP 3

```python
# TODO: 코드 작성
```

### 문제 2: 할인 효과 분석

할인을 받은 거래와 받지 않은 거래를 비교하세요:
1. 각각의 평균 거래액
2. 각각의 거래 건수

```python
# TODO: 코드 작성
# 힌트: discount_rate > 0 조건 사용
```

### 문제 3: 결제 수단별 분석

결제 수단별로 다음을 분석하세요:
1. 거래 건수
2. 평균 거래액
3. 가장 선호하는 카테고리

```python
# TODO: 코드 작성
```

---

## 📝 정리

이번 시간에 배운 내용:

✅ **Pandas 기본**: DataFrame 생성 및 구조  
✅ **데이터 불러오기**: CSV 읽기  
✅ **기본 탐색**: head, info, describe  
✅ **행/열 선택**: loc, iloc  
✅ **조건 필터링**: 단일/다중 조건, isin  
✅ **데이터 정렬**: sort_values  
✅ **실무 예제**: 카테고리별, 고객별, 지역별 분석

---

## 🔗 다음 시간 예고

**3교시: 데이터 정제 & 변환**

- 결측치 처리
- 중복 데이터 제거
- 데이터 타입 변환
- 문자열 처리
- 날짜/시간 처리

---

**수고하셨습니다! 🎉**
