# 📊 Pandas 실무 데이터분석 입문

> **대상:** 재직자 대상 실무형 데이터 분석 교육  
> **환경:** Python 3.x + Pandas, Windows + Miniconda + VSCode (CPU 환경)  
> **총 소요 시간:** 3시간 30분 (1교시 1시간, 2교시 1시간, 3교시 1시간 30분)

---

## 📚 목차

1. [🕐 1교시 - Pandas loc, iloc 인덱서를 이용한 데이터프레임 행/열 선택](#-1교시---pandas-loc-iloc-인덱서를-이용한-데이터프레임-행열-선택)
2. [🕐 2교시 - 조건(Boolean)을 활용한 행(Row) 필터링](#-2교시---조건boolean을-활용한-행row-필터링)
3. [🕐 3교시 - sort_values()를 이용한 데이터 정렬 및 기초 통계 계산](#-3교시---sort_values를-이용한-데이터-정렬-및-기초-통계-계산)

---

## 🕐 1교시 - Pandas loc, iloc 인덱서를 이용한 데이터프레임 행/열 선택

### 📘 이론 요약

#### 1.1 DataFrame 인덱서란?

Pandas DataFrame에서 특정 행(row)과 열(column)을 선택하기 위해 사용하는 도구입니다.

**두 가지 핵심 인덱서:**
- **`loc`**: **라벨(Label) 기반** 인덱싱 → 행과 열의 **이름**으로 접근
- **`iloc`**: **정수(Integer) 기반** 인덱싱 → 행과 열의 **위치(0부터 시작)**로 접근

#### 1.2 loc vs iloc 비교

| 구분 | loc | iloc |
|------|-----|------|
| **기준** | 라벨(이름) | 정수 위치 (0-based) |
| **행 선택** | `df.loc['row_label']` | `df.iloc[0]` (첫 번째 행) |
| **열 선택** | `df.loc[:, 'column_name']` | `df.iloc[:, 0]` (첫 번째 열) |
| **범위 선택** | `df.loc[0:5]` (5 포함) | `df.iloc[0:5]` (5 미포함) |
| **실무 용도** | 컬럼명으로 데이터 탐색 | 순서/위치 기반 데이터 추출 |

#### 1.3 기본 문법

```python
# loc 문법
df.loc[행_라벨, 열_라벨]
df.loc[행_라벨]  # 열 생략 시 전체 열 선택

# iloc 문법
df.iloc[행_위치, 열_위치]
df.iloc[행_위치]  # 열 생략 시 전체 열 선택
```

#### 1.4 실무 활용 예시

- **고객 ID로 특정 고객 정보 조회**: `df.loc[df['고객ID'] == 'C001']`
- **상위 10개 데이터 추출**: `df.iloc[:10]`
- **특정 위치의 데이터 값 확인**: `df.iloc[2, 4]` (3번째 행, 5번째 열)

---

### 💻 실습 코드

```python
# 필수 라이브러리 import
import pandas as pd
import numpy as np

# 공개 데이터셋 로드 (Superstore 판매 데이터)
url = "sample_superstore_data.csv"
df = pd.read_csv(url)

print("=" * 80)
print("📊 Superstore 데이터셋 기본 정보")
print("=" * 80)
print(f"데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")
print(f"컬럼명: {list(df.columns)}")
print("\n상위 5개 데이터:")
print(df.head())

# ============================================================
# 1. loc을 이용한 행 선택 (라벨 기반)
# ============================================================
print("\n" + "=" * 80)
print("1️⃣ loc - 라벨 기반 인덱싱")
print("=" * 80)

# 첫 번째 행 선택 (인덱스 0)
first_row = df.loc[0]
print("\n📌 첫 번째 행 (df.loc[0]):")
print(first_row)

# 특정 범위 행 선택 (0~4번 인덱스, 끝 포함!)
print("\n📌 0~4번 행 선택 (df.loc[0:4]):")
print(df.loc[0:4])

# 특정 열만 선택
print("\n📌 특정 열 선택 (df.loc[:, 'Category']):")
print(df.loc[:5, 'Category'])  # 0~5번 행의 Category 열

# 행과 열 동시 선택
print("\n📌 행과 열 동시 선택 (df.loc[0:2, ['Category', 'Sales']]):")
print(df.loc[0:2, ['Category', 'Sales']])

# ============================================================
# 2. iloc을 이용한 행 선택 (정수 위치 기반)
# ============================================================
print("\n" + "=" * 80)
print("2️⃣ iloc - 정수 위치 기반 인덱싱")
print("=" * 80)

# 첫 번째 행 선택
first_row_iloc = df.iloc[0]
print("\n📌 첫 번째 행 (df.iloc[0]):")
print(first_row_iloc)

# 특정 범위 행 선택 (0~4번 위치, 끝 미포함!)
print("\n📌 0~4번 행 선택 (df.iloc[0:5]):")
print(df.iloc[0:5])

# 특정 위치 열만 선택 (3번째 열)
print("\n📌 3번째 열 선택 (df.iloc[:, 2]):")
print(df.iloc[:5, 2])  # 0~4번 행의 3번째 열

# 행과 열 동시 선택 (위치 기반)
print("\n📌 행과 열 동시 선택 (df.iloc[0:3, [2, 4]]):")
print(df.iloc[0:3, [2, 4]])

# ============================================================
# 3. 실습 목표: 3번째 행, 5번째 열의 특정 데이터 값 선택
# ============================================================
print("\n" + "=" * 80)
print("3️⃣ 실습 목표: 3번째 행, 5번째 열 데이터 선택")
print("=" * 80)

# iloc 사용 (정수 위치 기반 - 0부터 시작)
value_iloc = df.iloc[2, 4]  # 3번째 행(인덱스 2), 5번째 열(인덱스 4)
print(f"\n📌 iloc 사용: df.iloc[2, 4] = {value_iloc}")
print(f"   해당 열 이름: {df.columns[4]}")

# loc 사용 (라벨 기반)
column_name = df.columns[4]
value_loc = df.loc[2, column_name]
print(f"\n📌 loc 사용: df.loc[2, '{column_name}'] = {value_loc}")

# 두 값이 동일한지 확인
print(f"\n✅ 두 방법의 결과가 동일한가? {value_iloc == value_loc}")

# ============================================================
# 4. loc과 iloc의 범위 선택 차이 명확히 비교
# ============================================================
print("\n" + "=" * 80)
print("4️⃣ loc과 iloc의 범위 선택 차이")
print("=" * 80)

print("\n📌 loc[0:3] - 끝 인덱스 포함 (0, 1, 2, 3번 행):")
print(df.loc[0:3, 'Category'])

print("\n📌 iloc[0:3] - 끝 인덱스 미포함 (0, 1, 2번 행):")
print(df.iloc[0:3, 2])  # 2번째 열 (Category)

# ============================================================
# 5. 다양한 인덱싱 예제
# ============================================================
print("\n" + "=" * 80)
print("5️⃣ 실무 활용 예제")
print("=" * 80)

# 예제 1: 마지막 행 선택
print("\n📌 마지막 행 선택:")
print(df.iloc[-1])

# 예제 2: 처음 10개 행의 특정 컬럼만 선택
print("\n📌 처음 10개 행의 Category, Sales, Profit 컬럼:")
print(df.loc[:9, ['Category', 'Sales', 'Profit']])

# 예제 3: 2개씩 건너뛰며 선택 (슬라이싱)
print("\n📌 2개씩 건너뛰며 행 선택 (0, 2, 4, 6, 8번 행):")
print(df.iloc[0:10:2, [2, 4]])

print("\n" + "=" * 80)
print("✅ 1교시 실습 완료!")
print("=" * 80)
```

---

### 🧩 퀴즈

#### **문제 1:**  
다음 중 `loc`과 `iloc`의 차이를 올바르게 설명한 것은?

A. `loc`은 정수 인덱스 기반, `iloc`은 라벨 기반  
B. `loc`은 라벨 기반, `iloc`은 정수 위치 기반  
C. 둘 다 동일하게 동작한다  
D. `loc`은 행 선택만, `iloc`은 열 선택만 가능하다

<details>
<summary>✅ 정답 보기</summary>

**정답:** B  

**해설:**  
- `loc`은 **라벨(이름) 기반** 인덱싱으로, 행과 열의 이름으로 접근합니다.
- `iloc`은 **정수(위치) 기반** 인덱싱으로, 행과 열의 위치(0부터 시작)로 접근합니다.
- 예: `df.loc[0:5]`는 0~5번 인덱스 **모두 포함**, `df.iloc[0:5]`는 0~4번만 선택 (5 미포함)

</details>

---

#### **문제 2:**  
DataFrame `df`에서 10번째 행, 3번째 열의 값을 선택하는 올바른 코드는? (행과 열 모두 1부터 세는 경우)

A. `df.loc[10, 3]`  
B. `df.iloc[10, 3]`  
C. `df.iloc[9, 2]`  
D. `df.loc[9, df.columns[2]]`

<details>
<summary>✅ 정답 보기</summary>

**정답:** C  

**해설:**  
- "10번째 행, 3번째 열"을 **사람이 세는 방식**으로 표현한 것이므로, **프로그래밍에서는 인덱스를 -1** 해야 합니다.
- 10번째 행 → 인덱스 9 (0부터 시작)
- 3번째 열 → 인덱스 2 (0부터 시작)
- 따라서 `df.iloc[9, 2]`가 정답입니다.
- `df.loc[9, df.columns[2]]`도 동일한 결과를 반환하지만, 문제에서 iloc을 사용하는 것이 더 적절합니다.

</details>

---

#### **문제 3:**  
다음 코드의 결과로 선택되는 행의 개수는?

```python
df.loc[5:10]
df.iloc[5:10]
```

A. loc: 5개, iloc: 5개  
B. loc: 6개, iloc: 5개  
C. loc: 5개, iloc: 6개  
D. loc: 6개, iloc: 6개

<details>
<summary>✅ 정답 보기</summary>

**정답:** B  

**해설:**  
- `df.loc[5:10]`: 라벨 기반 슬라이싱으로, **끝 인덱스(10)를 포함**합니다.  
  → 5, 6, 7, 8, 9, 10번 인덱스 = **6개 행**
  
- `df.iloc[5:10]`: 정수 위치 기반 슬라이싱으로, **끝 인덱스(10)를 미포함**합니다.  
  → 5, 6, 7, 8, 9번 인덱스 = **5개 행**

**핵심 차이점:**
- `loc`의 슬라이싱은 **끝 포함** (Python의 일반적인 슬라이싱과 다름!)
- `iloc`의 슬라이싱은 **끝 미포함** (Python의 일반적인 슬라이싱과 동일)

</details>

---
---

## 🕐 2교시 - 조건(Boolean)을 활용한 행(Row) 필터링

### 📘 이론 요약

#### 2.1 Boolean 인덱싱이란?

Boolean 인덱싱은 **조건식을 사용하여 True/False 값을 가진 마스크를 생성**하고, 이를 통해 원하는 행만 필터링하는 기법입니다.

**핵심 개념:**
```python
# 1단계: 조건식 생성 → Boolean Series 반환
condition = df['Sales'] > 1000

# 2단계: Boolean 마스크 적용 → 조건을 만족하는 행만 선택
filtered_df = df[condition]
```

#### 2.2 주요 비교 연산자

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `==` | 같음 | `df['Category'] == 'Furniture'` |
| `!=` | 같지 않음 | `df['Region'] != 'East'` |
| `>` | 크다 | `df['Sales'] > 1000` |
| `<` | 작다 | `df['Profit'] < 0` |
| `>=` | 크거나 같다 | `df['Quantity'] >= 5` |
| `<=` | 작거나 같다 | `df['Discount'] <= 0.2` |

#### 2.3 복합 조건 (여러 조건 결합)

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `&` | AND (그리고) | `(df['Sales'] > 1000) & (df['Profit'] > 100)` |
| `|` | OR (또는) | `(df['Category'] == 'Furniture') | (df['Category'] == 'Technology')` |
| `~` | NOT (부정) | `~(df['Region'] == 'East')` |

**⚠️ 주의사항:**
- 각 조건을 **괄호 `()`로 묶어야** 합니다.
- Python의 `and`, `or`가 아닌 `&`, `|`를 사용합니다.

#### 2.4 기타 유용한 필터링 메서드

```python
# 특정 값 목록에 포함되는지 확인
df[df['Category'].isin(['Furniture', 'Technology'])]

# 문자열 포함 여부 확인
df[df['Product Name'].str.contains('Chair')]

# 결측값(NaN) 필터링
df[df['Profit'].notna()]  # Profit이 NaN이 아닌 행
```

#### 2.5 실무 활용 사례

- **특정 제품 카테고리의 판매 내역 추출**
- **매출이 일정 금액 이상인 거래만 조회**
- **적자 거래(Profit < 0) 분석**
- **특정 지역의 고객 데이터 필터링**

---

### 💻 실습 코드

```python
# 필수 라이브러리 import
import pandas as pd
import numpy as np

# 공개 데이터셋 로드
url = "sample_superstore_data.csv"
df = pd.read_csv(url)

print("=" * 80)
print("📊 Superstore 데이터셋 기본 정보")
print("=" * 80)
print(f"전체 데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")
print("\n데이터 샘플:")
print(df.head())

# ============================================================
# 1. 단일 조건 필터링 (==, >, <)
# ============================================================
print("\n" + "=" * 80)
print("1️⃣ 단일 조건 필터링")
print("=" * 80)

# 조건 1: Category가 'Furniture'인 행만 선택
print("\n📌 조건: Category == 'Furniture'")
furniture_df = df[df['Category'] == 'Furniture']
print(f"필터링 결과: {len(furniture_df)}개 행")
print(furniture_df.head())

# 조건 2: Sales가 1000 이상인 행만 선택
print("\n📌 조건: Sales > 1000")
high_sales_df = df[df['Sales'] > 1000]
print(f"필터링 결과: {len(high_sales_df)}개 행")
print(high_sales_df[['Category', 'Sales', 'Profit']].head())

# 조건 3: Profit이 0보다 작은 행 (적자 거래)
print("\n📌 조건: Profit < 0 (적자 거래)")
loss_df = df[df['Profit'] < 0]
print(f"필터링 결과: {len(loss_df)}개 행")
print(loss_df[['Category', 'Sales', 'Profit']].head())

# ============================================================
# 2. 복합 조건 필터링 (AND, OR)
# ============================================================
print("\n" + "=" * 80)
print("2️⃣ 복합 조건 필터링")
print("=" * 80)

# AND 조건: Sales > 1000 AND Profit > 100
print("\n📌 조건: (Sales > 1000) & (Profit > 100)")
high_profit_df = df[(df['Sales'] > 1000) & (df['Profit'] > 100)]
print(f"필터링 결과: {len(high_profit_df)}개 행")
print(high_profit_df[['Category', 'Sales', 'Profit']].head())

# OR 조건: Category가 'Furniture' 또는 'Technology'
print("\n📌 조건: (Category == 'Furniture') | (Category == 'Technology')")
furn_tech_df = df[(df['Category'] == 'Furniture') | (df['Category'] == 'Technology')]
print(f"필터링 결과: {len(furn_tech_df)}개 행")
print(furn_tech_df['Category'].value_counts())

# NOT 조건: Region이 'East'가 아닌 행
print("\n📌 조건: ~(Region == 'East')")
not_east_df = df[~(df['Region'] == 'East')]
print(f"필터링 결과: {len(not_east_df)}개 행")
print(not_east_df['Region'].value_counts())

# ============================================================
# 3. 실습 목표: 특정 제품의 판매 내역만 추출
# ============================================================
print("\n" + "=" * 80)
print("3️⃣ 실습 목표: 특정 제품의 판매 내역 추출")
print("=" * 80)

# 목표 1: 'Office Supplies' 카테고리의 모든 판매 내역
print("\n📌 Office Supplies 카테고리 판매 내역:")
office_supplies = df[df['Category'] == 'Office Supplies']
print(f"총 {len(office_supplies)}개 거래")
print(office_supplies[['Sub-Category', 'Sales', 'Profit']].head(10))

# 목표 2: 'Technology' 카테고리 + 매출 500 이상
print("\n📌 Technology 카테고리 중 매출 500 이상:")
tech_high_sales = df[(df['Category'] == 'Technology') & (df['Sales'] >= 500)]
print(f"총 {len(tech_high_sales)}개 거래")
print(tech_high_sales[['Sub-Category', 'Sales', 'Profit']].head(10))

# ============================================================
# 4. isin() 메서드를 이용한 필터링
# ============================================================
print("\n" + "=" * 80)
print("4️⃣ isin() 메서드 활용")
print("=" * 80)

# 여러 카테고리를 한 번에 필터링
categories_to_filter = ['Furniture', 'Technology']
print(f"\n📌 카테고리가 {categories_to_filter}에 포함되는 행:")
filtered_categories = df[df['Category'].isin(categories_to_filter)]
print(f"총 {len(filtered_categories)}개 거래")
print(filtered_categories['Category'].value_counts())

# 특정 Region만 선택
regions = ['West', 'Central']
print(f"\n📌 Region이 {regions}에 포함되는 행:")
filtered_regions = df[df['Region'].isin(regions)]
print(f"총 {len(filtered_regions)}개 거래")
print(filtered_regions['Region'].value_counts())

# ============================================================
# 5. 문자열 필터링 (str.contains)
# ============================================================
print("\n" + "=" * 80)
print("5️⃣ 문자열 필터링 (str.contains)")
print("=" * 80)

# Sub-Category에 'Chair'가 포함된 제품
print("\n📌 Sub-Category에 'Chair'가 포함된 제품:")
chair_products = df[df['Sub-Category'].str.contains('Chair', na=False)]
print(f"총 {len(chair_products)}개 거래")
print(chair_products[['Sub-Category', 'Sales', 'Profit']].head())

# ============================================================
# 6. 실무 응용: 조건별 매출 분석
# ============================================================
print("\n" + "=" * 80)
print("6️⃣ 실무 응용: 조건별 매출 분석")
print("=" * 80)

# 고매출 거래 (Sales > 1000)
high_sales = df[df['Sales'] > 1000]
print(f"\n📊 고매출 거래 (Sales > 1000): {len(high_sales)}건")
print(f"   총 매출: ${high_sales['Sales'].sum():,.2f}")
print(f"   평균 매출: ${high_sales['Sales'].mean():,.2f}")

# 저수익 거래 (Profit < 50)
low_profit = df[df['Profit'] < 50]
print(f"\n📊 저수익 거래 (Profit < 50): {len(low_profit)}건")
print(f"   총 수익: ${low_profit['Profit'].sum():,.2f}")
print(f"   평균 수익: ${low_profit['Profit'].mean():,.2f}")

# 카테고리별 필터링 후 통계
print("\n📊 카테고리별 통계:")
for category in df['Category'].unique():
    cat_df = df[df['Category'] == category]
    print(f"   {category}: {len(cat_df)}건, 평균 매출 ${cat_df['Sales'].mean():,.2f}")

print("\n" + "=" * 80)
print("✅ 2교시 실습 완료!")
print("=" * 80)
```

---

### 🧩 퀴즈

#### **문제 1:**  
DataFrame에서 `Sales`가 500 이상이고 `Region`이 'West'인 행을 필터링하는 올바른 코드는?

A. `df[df['Sales'] > 500 and df['Region'] == 'West']`  
B. `df[(df['Sales'] >= 500) & (df['Region'] == 'West')]`  
C. `df[df['Sales'] >= 500 | df['Region'] == 'West']`  
D. `df.filter(Sales >= 500, Region == 'West')`

<details>
<summary>✅ 정답 보기</summary>

**정답:** B  

**해설:**  
- Pandas에서 복합 조건을 사용할 때는 **각 조건을 괄호로 묶고**, `&` (AND) 또는 `|` (OR) 연산자를 사용해야 합니다.
- Python의 `and`, `or` 키워드는 사용할 수 없습니다.
- `>=`를 사용하여 500 이상(500 포함)을 표현합니다.

**올바른 형식:**
```python
df[(조건1) & (조건2)]  # AND
df[(조건1) | (조건2)]  # OR
df[~(조건)]           # NOT
```

</details>

---

#### **문제 2:**  
다음 코드의 실행 결과로 선택되는 행은?

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['Seoul', 'Busan', 'Seoul', 'Incheon']
})

result = df[(df['Age'] > 28) & (df['City'] == 'Seoul')]
```

A. Alice  
B. Bob  
C. Charlie  
D. David

<details>
<summary>✅ 정답 보기</summary>

**정답:** C  

**해설:**  
조건을 분석하면:
- `df['Age'] > 28`: Bob (30), Charlie (35), David (40)
- `df['City'] == 'Seoul'`: Alice, Charlie

**AND 조건 (`&`)**: 두 조건을 **모두** 만족하는 행만 선택
- Bob: Age 30 > 28 ✓, City != Seoul ✗
- Charlie: Age 35 > 28 ✓, City == Seoul ✓ → **선택됨**
- David: Age 40 > 28 ✓, City != Seoul ✗

따라서 Charlie만 선택됩니다.

</details>

---

#### **문제 3:**  
다음 중 `isin()` 메서드를 사용하여 `Category`가 'A', 'B', 'C' 중 하나인 행을 선택하는 올바른 코드는?

A. `df[df['Category'] == ['A', 'B', 'C']]`  
B. `df[df['Category'].isin(['A', 'B', 'C'])]`  
C. `df[(df['Category'] == 'A') | ('B') | ('C')]`  
D. `df.filter(Category in ['A', 'B', 'C'])`

<details>
<summary>✅ 정답 보기</summary>

**정답:** B  

**해설:**  
- `isin()` 메서드는 **리스트에 포함된 값**을 가진 행을 선택할 때 사용합니다.
- 여러 OR 조건을 간결하게 표현할 수 있습니다.

**비교:**
```python
# isin() 사용 (권장)
df[df['Category'].isin(['A', 'B', 'C'])]

# OR 조건 사용 (같은 결과, 더 복잡함)
df[(df['Category'] == 'A') | (df['Category'] == 'B') | (df['Category'] == 'C')]
```

**실무 팁:** 여러 값 중 하나를 선택할 때는 `isin()`을 사용하는 것이 코드가 더 깔끔합니다.

</details>

---

#### **문제 4:**  
다음 코드의 결과는?

```python
df = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D'],
    'Price': [100, 200, 150, 250]
})

result = df[~(df['Price'] > 150)]
```

A. Product A, B  
B. Product A, C  
C. Product B, D  
D. Product C, D

<details>
<summary>✅ 정답 보기</summary>

**정답:** B  

**해설:**  
- `~` 연산자는 **NOT (부정)**을 의미합니다.
- `df['Price'] > 150`: B (200), D (250) → True
- `~(df['Price'] > 150)`: True를 False로, False를 True로 반전
  - A (100): False → True ✓
  - B (200): True → False ✗
  - C (150): False → True ✓
  - D (250): True → False ✗

따라서 A와 C가 선택됩니다 (Price가 150 **이하**인 행).

**참고:** `~(df['Price'] > 150)`는 `df['Price'] <= 150`과 동일한 결과입니다.

</details>

---
---

## 🕐 3교시 - sort_values()를 이용한 데이터 정렬 및 기초 통계 계산

### 📘 이론 요약

#### 3.1 sort_values()를 이용한 데이터 정렬

`sort_values()`는 DataFrame의 특정 열을 기준으로 데이터를 정렬하는 메서드입니다.

**기본 문법:**
```python
df.sort_values(by='컬럼명', ascending=True, inplace=False)
```

**주요 매개변수:**
- `by`: 정렬 기준 컬럼명 (문자열 또는 리스트)
- `ascending`: 오름차순(True) / 내림차순(False)
- `inplace`: 원본 DataFrame 수정 여부 (True: 원본 수정, False: 새 DataFrame 반환)
- `na_position`: 결측값 위치 ('first' 또는 'last')

**실무 활용:**
- 매출 상위 10개 제품 찾기
- 가격이 낮은 순서로 제품 정렬
- 날짜순으로 거래 내역 정렬

#### 3.2 여러 컬럼 기준 정렬

```python
# Category로 먼저 정렬하고, 같은 Category 내에서 Sales로 정렬
df.sort_values(by=['Category', 'Sales'], ascending=[True, False])
```

#### 3.3 기초 통계 함수

Pandas는 다양한 통계 계산 메서드를 제공합니다:

| 함수 | 설명 | 예시 |
|------|------|------|
| `sum()` | 합계 | `df['Sales'].sum()` |
| `mean()` | 평균 | `df['Sales'].mean()` |
| `median()` | 중앙값 | `df['Sales'].median()` |
| `min()` | 최솟값 | `df['Sales'].min()` |
| `max()` | 최댓값 | `df['Sales'].max()` |
| `std()` | 표준편차 | `df['Sales'].std()` |
| `count()` | 개수 (NaN 제외) | `df['Sales'].count()` |
| `describe()` | 요약 통계 | `df['Sales'].describe()` |

#### 3.4 groupby()를 이용한 그룹별 통계

`groupby()`는 데이터를 그룹으로 나누고, 각 그룹에 대해 집계 함수를 적용합니다.

**기본 문법:**
```python
df.groupby('그룹_컬럼')['집계_컬럼'].집계함수()
```

**예시:**
```python
# 카테고리별 평균 매출
df.groupby('Category')['Sales'].mean()

# 지역별 총 매출과 평균 수익
df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'mean'
})
```

#### 3.5 실무 활용 시나리오

1. **상위/하위 N개 데이터 추출**
   - 매출 상위 10개: `df.nlargest(10, 'Sales')`
   - 수익 하위 5개: `df.nsmallest(5, 'Profit')`

2. **그룹별 순위 매기기**
   - `df.rank()` 메서드 활용

3. **피벗 테이블 생성**
   - `df.pivot_table()` 활용

---

### 💻 실습 코드

```python
# 필수 라이브러리 import
import pandas as pd
import numpy as np

# 공개 데이터셋 로드
url = "sample_superstore_data.csv"
df = pd.read_csv(url)

print("=" * 80)
print("📊 Superstore 데이터셋 기본 정보")
print("=" * 80)
print(f"전체 데이터 크기: {df.shape[0]} 행, {df.shape[1]} 열")
print("\n데이터 샘플:")
print(df.head())

# ============================================================
# 1. 단일 컬럼 기준 정렬
# ============================================================
print("\n" + "=" * 80)
print("1️⃣ 단일 컬럼 기준 정렬")
print("=" * 80)

# 오름차순 정렬 (기본값)
print("\n📌 Sales 기준 오름차순 정렬 (낮은 값 → 높은 값):")
sorted_asc = df.sort_values(by='Sales', ascending=True)
print(sorted_asc[['Category', 'Sub-Category', 'Sales', 'Profit']].head(10))

# 내림차순 정렬
print("\n📌 Sales 기준 내림차순 정렬 (높은 값 → 낮은 값):")
sorted_desc = df.sort_values(by='Sales', ascending=False)
print(sorted_desc[['Category', 'Sub-Category', 'Sales', 'Profit']].head(10))

# ============================================================
# 2. 여러 컬럼 기준 정렬
# ============================================================
print("\n" + "=" * 80)
print("2️⃣ 여러 컬럼 기준 정렬")
print("=" * 80)

# Category (오름차순) → Sales (내림차순)
print("\n📌 Category(오름차순) → Sales(내림차순) 정렬:")
multi_sorted = df.sort_values(by=['Category', 'Sales'], ascending=[True, False])
print(multi_sorted[['Category', 'Sub-Category', 'Sales', 'Profit']].head(15))

# Region (오름차순) → Profit (내림차순)
print("\n📌 Region(오름차순) → Profit(내림차순) 정렬:")
region_sorted = df.sort_values(by=['Region', 'Profit'], ascending=[True, False])
print(region_sorted[['Region', 'Category', 'Sales', 'Profit']].head(15))

# ============================================================
# 3. 실습 목표: 판매량 기준 상품 정렬
# ============================================================
print("\n" + "=" * 80)
print("3️⃣ 실습 목표: 판매량 기준 상품 정렬")
print("=" * 80)

# Quantity(수량) 기준 정렬
print("\n📌 Quantity 기준 내림차순 정렬 (판매량이 많은 순서):")
quantity_sorted = df.sort_values(by='Quantity', ascending=False)
print(quantity_sorted[['Category', 'Sub-Category', 'Quantity', 'Sales', 'Profit']].head(20))

# 상위 10개만 추출 (nlargest 사용)
print("\n📌 Quantity 상위 10개 (nlargest 사용):")
top10_quantity = df.nlargest(10, 'Quantity')
print(top10_quantity[['Category', 'Sub-Category', 'Quantity', 'Sales', 'Profit']])

# ============================================================
# 4. 기초 통계 계산 (sum, mean, median, etc.)
# ============================================================
print("\n" + "=" * 80)
print("4️⃣ 기초 통계 계산")
print("=" * 80)

print("\n📊 전체 데이터 통계:")
print(f"   총 거래 건수: {len(df):,}건")
print(f"   총 매출(Sales): ${df['Sales'].sum():,.2f}")
print(f"   평균 매출: ${df['Sales'].mean():,.2f}")
print(f"   중앙값 매출: ${df['Sales'].median():,.2f}")
print(f"   최소 매출: ${df['Sales'].min():,.2f}")
print(f"   최대 매출: ${df['Sales'].max():,.2f}")
print(f"   표준편차: ${df['Sales'].std():,.2f}")

print("\n📊 Profit 통계:")
print(f"   총 수익: ${df['Profit'].sum():,.2f}")
print(f"   평균 수익: ${df['Profit'].mean():,.2f}")
print(f"   최소 수익: ${df['Profit'].min():,.2f}")
print(f"   최대 수익: ${df['Profit'].max():,.2f}")

# describe()를 이용한 요약 통계
print("\n📊 Sales 요약 통계 (describe):")
print(df['Sales'].describe())

# ============================================================
# 5. 실습 목표: 전체 평균 판매량 계산
# ============================================================
print("\n" + "=" * 80)
print("5️⃣ 실습 목표: 전체 평균 판매량 계산")
print("=" * 80)

avg_quantity = df['Quantity'].mean()
avg_sales = df['Sales'].mean()
avg_profit = df['Profit'].mean()

print(f"\n📊 전체 평균 통계:")
print(f"   평균 판매량(Quantity): {avg_quantity:.2f}개")
print(f"   평균 매출(Sales): ${avg_sales:,.2f}")
print(f"   평균 수익(Profit): ${avg_profit:,.2f}")

# 평균보다 높은 거래 필터링
above_avg_sales = df[df['Sales'] > avg_sales]
print(f"\n📊 평균 매출 이상 거래:")
print(f"   건수: {len(above_avg_sales):,}건 (전체의 {len(above_avg_sales)/len(df)*100:.1f}%)")
print(f"   총 매출: ${above_avg_sales['Sales'].sum():,.2f}")

# ============================================================
# 6. 실습 목표: 그룹별 요약 통계 (groupby)
# ============================================================
print("\n" + "=" * 80)
print("6️⃣ 실습 목표: 그룹별 요약 통계")
print("=" * 80)

# Category별 통계
print("\n📊 Category별 요약 통계:")
category_stats = df.groupby('Category').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Profit': ['sum', 'mean'],
    'Quantity': ['sum', 'mean']
})
print(category_stats)

# Region별 통계
print("\n📊 Region별 요약 통계:")
region_stats = df.groupby('Region').agg({
    'Sales': ['sum', 'mean'],
    'Profit': ['sum', 'mean'],
    'Quantity': 'sum'
})
print(region_stats)

# Sub-Category별 평균 매출 (상위 10개)
print("\n📊 Sub-Category별 평균 매출 (상위 10개):")
subcategory_avg = df.groupby('Sub-Category')['Sales'].mean().sort_values(ascending=False).head(10)
print(subcategory_avg)

# ============================================================
# 7. 실무 응용: 카테고리별 상위 제품 분석
# ============================================================
print("\n" + "=" * 80)
print("7️⃣ 실무 응용: 카테고리별 상위 제품 분석")
print("=" * 80)

# 각 Category별 Sales 상위 3개 제품
print("\n📊 각 Category별 매출 상위 3개:")
for category in df['Category'].unique():
    print(f"\n▶ {category}:")
    top3 = df[df['Category'] == category].nlargest(3, 'Sales')
    print(top3[['Sub-Category', 'Sales', 'Profit']])

# ============================================================
# 8. 피벗 테이블 생성
# ============================================================
print("\n" + "=" * 80)
print("8️⃣ 피벗 테이블 생성")
print("=" * 80)

# Region과 Category별 평균 매출
print("\n📊 Region과 Category별 평균 매출:")
pivot_table = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Category',
    aggfunc='mean'
)
print(pivot_table)

# Region과 Category별 총 수익
print("\n📊 Region과 Category별 총 수익:")
pivot_profit = df.pivot_table(
    values='Profit',
    index='Region',
    columns='Category',
    aggfunc='sum'
)
print(pivot_profit)

# ============================================================
# 9. 실무 응용: 성과 분석 대시보드 데이터
# ============================================================
print("\n" + "=" * 80)
print("9️⃣ 실무 응용: 성과 분석 대시보드 데이터")
print("=" * 80)

# 전체 요약 통계
print("\n📊 전체 비즈니스 요약:")
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
total_transactions = len(df)
profit_margin = (total_profit / total_sales) * 100

print(f"   총 거래 건수: {total_transactions:,}건")
print(f"   총 매출: ${total_sales:,.2f}")
print(f"   총 수익: ${total_profit:,.2f}")
print(f"   수익률: {profit_margin:.2f}%")

# 카테고리별 성과
print("\n📊 카테고리별 성과 순위:")
category_performance = df.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_values('Sales', ascending=False)
category_performance['수익률(%)'] = (category_performance['Profit'] / category_performance['Sales'] * 100)
print(category_performance)

# Region별 성과
print("\n📊 Region별 성과 순위:")
region_performance = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_values('Profit', ascending=False)
region_performance['수익률(%)'] = (region_performance['Profit'] / region_performance['Sales'] * 100)
print(region_performance)

print("\n" + "=" * 80)
print("✅ 3교시 실습 완료!")
print("=" * 80)
```

---

### 🧩 퀴즈

#### **문제 1:**  
DataFrame을 `Sales` 컬럼 기준으로 **높은 값부터 낮은 값** 순서로 정렬하는 올바른 코드는?

A. `df.sort_values(by='Sales')`  
B. `df.sort_values(by='Sales', ascending=True)`  
C. `df.sort_values(by='Sales', ascending=False)`  
D. `df.sort_values('Sales', reverse=True)`

<details>
<summary>✅ 정답 보기</summary>

**정답:** C  

**해설:**  
- `ascending=False`: 내림차순 정렬 (높은 값 → 낮은 값)
- `ascending=True` (기본값): 오름차순 정렬 (낮은 값 → 높은 값)

**기억하기:**
- **ascending** = "오름차순"
- `True` → 오름차순 (⬆️ 작은 값부터)
- `False` → 내림차순 (⬇️ 큰 값부터)

</details>

---

#### **문제 2:**  
다음 코드의 실행 결과는?

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Score': [85, 92, 78]
})

result = df['Score'].mean()
```

A. 255  
B. 85.0  
C. 85  
D. 85.0

<details>
<summary>✅ 정답 보기</summary>

**정답:** B (또는 D - 둘 다 같은 값)  

**해설:**  
- `mean()` 함수는 평균을 계산합니다.
- (85 + 92 + 78) / 3 = 255 / 3 = 85.0
- Pandas의 `mean()`은 기본적으로 float 타입을 반환합니다.

**주요 통계 함수 정리:**
- `sum()`: 합계 → 255
- `mean()`: 평균 → 85.0
- `median()`: 중앙값 → 85
- `min()`: 최솟값 → 78
- `max()`: 최댓값 → 92

</details>

---

#### **문제 3:**  
다음 중 `groupby()`를 사용하여 Category별 평균 Sales를 계산하는 올바른 코드는?

A. `df.groupby('Category').mean('Sales')`  
B. `df.groupby('Category')['Sales'].mean()`  
C. `df['Sales'].groupby('Category').mean()`  
D. `df.mean().groupby('Category')['Sales']`

<details>
<summary>✅ 정답 보기</summary>

**정답:** B  

**해설:**  
- `groupby()` 기본 패턴: `df.groupby('그룹컬럼')['대상컬럼'].집계함수()`

**단계별 설명:**
1. `df.groupby('Category')`: Category별로 그룹 생성
2. `['Sales']`: Sales 컬럼만 선택
3. `.mean()`: 각 그룹의 평균 계산

**다른 예시:**
```python
# 여러 컬럼 집계
df.groupby('Category').agg({
    'Sales': 'mean',
    'Profit': 'sum'
})

# 여러 그룹 기준
df.groupby(['Region', 'Category'])['Sales'].sum()
```

</details>

---

#### **문제 4:**  
다음 코드의 실행 결과는?

```python
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Sales': [100, 200, 150, 250, 120]
})

result = df.groupby('Category')['Sales'].sum()
```

A. Category A: 370, B: 450  
B. Category A: 123.33, B: 225  
C. Category A: 3, B: 2  
D. 전체 합계: 820

<details>
<summary>✅ 정답 보기</summary>

**정답:** A  

**해설:**  
`groupby()` 후 `sum()`은 각 그룹의 **합계**를 계산합니다.

**계산 과정:**
- Category A: 100 + 150 + 120 = **370**
- Category B: 200 + 250 = **450**

**결과 형태:**
```
Category
A    370
B    450
Name: Sales, dtype: int64
```

**다른 집계 함수:**
- `mean()`: A → 123.33, B → 225.0 (평균)
- `count()`: A → 3, B → 2 (개수)
- `max()`: A → 150, B → 250 (최댓값)

</details>

---

#### **문제 5:**  
DataFrame에서 Sales가 가장 높은 상위 5개 행을 선택하는 가장 효율적인 방법은?

A. `df.sort_values(by='Sales', ascending=False).head(5)`  
B. `df.nlargest(5, 'Sales')`  
C. `df.iloc[:5].sort_values('Sales')`  
D. `df[df['Sales'] > df['Sales'].quantile(0.95)]`

<details>
<summary>✅ 정답 보기</summary>

**정답:** B  

**해설:**  
- `nlargest(n, 'column')`: 가장 큰 n개 행을 선택 (최적화된 메서드)
- `nsmallest(n, 'column')`: 가장 작은 n개 행을 선택

**비교:**
```python
# 방법 1: sort_values + head (동작하지만 비효율적)
df.sort_values(by='Sales', ascending=False).head(5)

# 방법 2: nlargest (권장 - 더 빠르고 간결)
df.nlargest(5, 'Sales')
```

**성능 차이:**
- `nlargest()`는 내부적으로 최적화된 알고리즘 사용
- 대용량 데이터에서 `sort_values() + head()`보다 훨씬 빠름

**실무 팁:**
- 상위/하위 N개만 필요할 때 → `nlargest()` / `nsmallest()`
- 전체 정렬이 필요할 때 → `sort_values()`

</details>

---

## 🎓 학습 완료 체크리스트

### ✅ 1교시: loc, iloc 인덱서
- [ ] loc과 iloc의 차이점을 설명할 수 있다
- [ ] 특정 행과 열을 선택할 수 있다
- [ ] 범위 선택 시 차이점(끝 포함 여부)을 이해했다
- [ ] 실무에서 언제 loc/iloc을 사용할지 판단할 수 있다

### ✅ 2교시: Boolean 필터링
- [ ] 비교 연산자를 사용하여 조건을 만들 수 있다
- [ ] 복합 조건(AND, OR, NOT)을 사용할 수 있다
- [ ] isin() 메서드를 활용할 수 있다
- [ ] 특정 조건을 만족하는 데이터만 추출할 수 있다

### ✅ 3교시: 정렬 및 통계
- [ ] sort_values()로 데이터를 정렬할 수 있다
- [ ] 기초 통계 함수(sum, mean, etc.)를 사용할 수 있다
- [ ] groupby()로 그룹별 통계를 계산할 수 있다
- [ ] 피벗 테이블을 생성할 수 있다

---

## 📚 추가 학습 자료

### 공식 문서
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Pandas 10분 완성 가이드](https://pandas.pydata.org/docs/user_guide/10min.html)

### 실습 데이터셋
- [Plotly Datasets](https://github.com/plotly/datasets)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

### 다음 단계 학습 주제
1. **데이터 전처리**: 결측값 처리, 중복 제거, 데이터 타입 변환
2. **데이터 병합**: merge, join, concat을 이용한 데이터 결합
3. **시계열 데이터**: datetime 타입 다루기, 날짜 기반 필터링
4. **시각화**: Matplotlib, Seaborn을 이용한 데이터 시각화

---

## 🎯 실무 프로젝트 아이디어

이 과정을 마친 후 다음과 같은 실무 프로젝트를 시도해보세요:

1. **매출 분석 대시보드**
   - 월별/분기별 매출 추이 분석
   - 제품 카테고리별 성과 비교
   - 지역별 판매 현황 파악

2. **고객 세그먼테이션**
   - 구매 패턴별 고객 분류
   - 고가치 고객(VIP) 식별
   - 재구매율 분석

3. **재고 최적화**
   - 판매량 기반 재고 예측
   - 재고 회전율 계산
   - 저성과 제품 식별

---

**🎉 축하합니다! Pandas 실무 데이터분석 입문 과정을 완료하셨습니다! 🎉**

이제 실무에서 데이터를 자유자재로 다룰 수 있는 기초를 다졌습니다.  
계속해서 다양한 데이터셋으로 연습하며 실력을 키워나가세요! 💪
