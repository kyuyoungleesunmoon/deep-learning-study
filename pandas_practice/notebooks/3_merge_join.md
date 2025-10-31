# 📚 3교시: 두 개 이상의 데이터프레임 병합 (merge/join)

> **학습 목표:** `merge()`와 `join()` 메서드를 활용하여 여러 데이터프레임을 결합하는 방법을 학습합니다.  
> **소요 시간:** 1시간  
> **사용 데이터:** 고객 데이터 (customers.csv) + 주문 데이터 (orders.csv)

---

## 🧠 이론 설명

### 3.1 데이터 병합이란?

**데이터 병합(Merge/Join)**은 두 개 이상의 데이터프레임을 공통 키(Key)를 기준으로 결합하는 작업입니다.

#### 📊 실생활 비유
- **고객 정보** 테이블: 고객ID, 이름, 나이, 등급
- **주문 정보** 테이블: 주문ID, 고객ID, 주문금액, 주문일
- **목표**: 고객ID를 기준으로 두 테이블을 합쳐서 "각 주문에 고객 정보를 추가"

### 3.2 병합의 종류 (SQL JOIN과 유사)

Pandas의 `merge()`는 SQL의 JOIN 연산과 동일한 개념입니다.

#### 1. Inner Join (내부 조인)
- **양쪽 모두에 있는** 키만 결합
- 교집합 개념

```python
pd.merge(df1, df2, on='key', how='inner')
```

#### 2. Left Join (왼쪽 조인)
- **왼쪽 DataFrame의 모든 행** 유지
- 오른쪽에 매칭되는 데이터가 없으면 NaN

```python
pd.merge(df1, df2, on='key', how='left')
```

#### 3. Right Join (오른쪽 조인)
- **오른쪽 DataFrame의 모든 행** 유지
- 왼쪽에 매칭되는 데이터가 없으면 NaN

```python
pd.merge(df1, df2, on='key', how='right')
```

#### 4. Outer Join (외부 조인)
- **양쪽 DataFrame의 모든 행** 유지
- 매칭되지 않는 부분은 NaN
- 합집합 개념

```python
pd.merge(df1, df2, on='key', how='outer')
```

### 3.3 시각적 이해

```
df1 (고객):          df2 (주문):
고객ID  이름          주문ID  고객ID  금액
C001   홍길동        O001   C001   10000
C002   김철수        O002   C001   20000
C003   이영희        O003   C002   15000

Inner Join (교집합):
고객ID  이름    주문ID  금액
C001   홍길동   O001   10000
C001   홍길동   O002   20000
C002   김철수   O003   15000

Left Join (왼쪽 기준):
고객ID  이름    주문ID  금액
C001   홍길동   O001   10000
C001   홍길동   O002   20000
C002   김철수   O003   15000
C003   이영희   NaN    NaN      <- 주문이 없어도 표시
```

### 3.4 병합 시 주의사항

1. **키 컬럼 확인**: 병합 기준이 되는 컬럼이 양쪽에 존재해야 함
2. **중복 키**: 한쪽에 중복 키가 있으면 결과 행이 늘어남 (1:N 관계)
3. **컬럼명 충돌**: 같은 이름의 컬럼이 있으면 `_x`, `_y` 접미사 자동 추가
4. **데이터 타입**: 키 컬럼의 데이터 타입이 일치해야 함

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
customers = pd.read_csv('../data/customers.csv')
orders = pd.read_csv('../data/orders.csv')

print("=" * 80)
print("📊 데이터셋 기본 정보")
print("=" * 80)

print("\n[1] 고객 데이터 (customers)")
print(f"크기: {customers.shape}")
print(customers.head())

print("\n[2] 주문 데이터 (orders)")
print(f"크기: {orders.shape}")
print(orders.head())

# 공통 키 확인
print("\n[3] 공통 키 컬럼 확인")
print(f"customers의 고객ID 개수: {customers['고객ID'].nunique()}")
print(f"orders의 고객ID 개수: {orders['고객ID'].nunique()}")
```

### 실습 1: Inner Join (내부 조인)

```python
print("\n" + "=" * 80)
print("📌 실습 1: Inner Join - 주문이 있는 고객만")
print("=" * 80)

# Inner Join: 양쪽 모두에 있는 고객ID만 결합
inner_merged = pd.merge(customers, orders, on='고객ID', how='inner')

print(f"\n병합 결과 크기: {inner_merged.shape}")
print("\n병합 결과 (상위 10개):")
print(inner_merged.head(10))

# 고객별 주문 건수 확인
orders_per_customer = inner_merged.groupby('고객명')['주문ID'].count().sort_values(ascending=False)
print("\n고객별 주문 건수 Top 5:")
print(orders_per_customer.head())

print("\n💡 Inner Join 특징:")
print("- 주문이 없는 고객은 결과에서 제외됨")
print("- 주문이 여러 건인 고객은 여러 행으로 나타남")
```

### 실습 2: Left Join (왼쪽 조인)

```python
print("\n" + "=" * 80)
print("📌 실습 2: Left Join - 모든 고객 표시 (주문 없어도)")
print("=" * 80)

# Left Join: 모든 고객 유지, 주문이 없으면 NaN
left_merged = pd.merge(customers, orders, on='고객ID', how='left')

print(f"\n병합 결과 크기: {left_merged.shape}")
print("\n병합 결과 (상위 10개):")
print(left_merged.head(10))

# 주문이 없는 고객 확인
no_orders = left_merged[left_merged['주문ID'].isna()]
print(f"\n주문이 없는 고객 수: {no_orders['고객ID'].nunique()}")
print("\n주문이 없는 고객 리스트:")
print(no_orders[['고객ID', '고객명', '등급']].drop_duplicates())

print("\n💡 Left Join 특징:")
print("- 왼쪽(고객) 데이터는 모두 유지")
print("- 주문이 없는 고객은 주문 관련 컬럼이 NaN")
```

### 실습 3: Right Join (오른쪽 조인)

```python
print("\n" + "=" * 80)
print("📌 실습 3: Right Join - 모든 주문 표시")
print("=" * 80)

# Right Join: 모든 주문 유지
right_merged = pd.merge(customers, orders, on='고객ID', how='right')

print(f"\n병합 결과 크기: {right_merged.shape}")
print("\n병합 결과 (상위 10개):")
print(right_merged.head(10))

# 고객 정보가 없는 주문 확인 (데이터 오류)
invalid_orders = right_merged[right_merged['고객명'].isna()]
print(f"\n고객 정보가 없는 주문: {len(invalid_orders)}")

print("\n💡 Right Join 특징:")
print("- 오른쪽(주문) 데이터는 모두 유지")
print("- 이 예제에서는 모든 주문에 고객 정보가 있음")
```

### 실습 4: Outer Join (외부 조인)

```python
print("\n" + "=" * 80)
print("📌 실습 4: Outer Join - 모든 데이터 표시")
print("=" * 80)

# Outer Join: 양쪽 모두 유지
outer_merged = pd.merge(customers, orders, on='고객ID', how='outer')

print(f"\n병합 결과 크기: {outer_merged.shape}")
print("\n병합 결과 (상위 10개):")
print(outer_merged.head(10))

# 결측치 확인
print("\n결측치 현황:")
print(outer_merged.isnull().sum())

print("\n💡 Outer Join 특징:")
print("- 고객과 주문 모두의 데이터 유지")
print("- 매칭되지 않는 부분은 NaN")
print("- 완전한 데이터 파악에 유용")
```

### 실습 5: 병합 결과 비교

```python
print("\n" + "=" * 80)
print("📌 실습 5: 병합 방법별 결과 비교")
print("=" * 80)

comparison = pd.DataFrame({
    '병합방법': ['Inner', 'Left', 'Right', 'Outer'],
    '결과행수': [
        len(inner_merged),
        len(left_merged),
        len(right_merged),
        len(outer_merged)
    ],
    '고유고객수': [
        inner_merged['고객ID'].nunique(),
        left_merged['고객ID'].nunique(),
        right_merged['고객ID'].nunique(),
        outer_merged['고객ID'].nunique()
    ],
    '주문건수': [
        inner_merged['주문ID'].count(),
        left_merged['주문ID'].count(),
        right_merged['주문ID'].count(),
        outer_merged['주문ID'].count()
    ]
})

print("\n병합 방법별 비교:")
print(comparison)

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.25

ax.bar(x - width, comparison['결과행수'], width, label='결과 행수', alpha=0.8)
ax.bar(x, comparison['고유고객수'], width, label='고유 고객수', alpha=0.8)
ax.bar(x + width, comparison['주문건수'], width, label='주문 건수', alpha=0.8)

ax.set_xlabel('병합 방법', fontsize=12)
ax.set_ylabel('개수', fontsize=12)
ax.set_title('병합 방법별 결과 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison['병합방법'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/merge_comparison.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/merge_comparison.png")
plt.show()
```

### 실습 6: 다른 컬럼명으로 병합

```python
print("\n" + "=" * 80)
print("📌 실습 6: 다른 컬럼명으로 병합")
print("=" * 80)

# customers의 '고객ID'와 orders의 '고객ID'가 다른 이름일 경우
# 예시를 위해 컬럼명 변경
orders_renamed = orders.rename(columns={'고객ID': 'customer_id'})
customers_renamed = customers.rename(columns={'고객ID': 'cust_id'})

# left_on, right_on 파라미터 사용
merged_diff_cols = pd.merge(
    customers_renamed, 
    orders_renamed, 
    left_on='cust_id', 
    right_on='customer_id', 
    how='inner'
)

print("\n다른 컬럼명으로 병합 결과:")
print(merged_diff_cols.head())

print("\n💡 주의:")
print("- 병합 키 컬럼이 두 개 생성됨 (cust_id, customer_id)")
print("- 필요 없는 컬럼은 drop()으로 제거")
```

### 실습 7: 여러 컬럼을 키로 사용

```python
print("\n" + "=" * 80)
print("📌 실습 7: 여러 컬럼을 키로 병합")
print("=" * 80)

# 예제 데이터 생성
df1 = pd.DataFrame({
    '지역': ['서울', '서울', '부산', '부산'],
    '년도': [2023, 2024, 2023, 2024],
    '인구': [1000, 1050, 350, 360]
})

df2 = pd.DataFrame({
    '지역': ['서울', '서울', '부산'],
    '년도': [2023, 2024, 2023],
    'GDP': [500, 520, 180]
})

print("\ndf1 (인구 데이터):")
print(df1)
print("\ndf2 (GDP 데이터):")
print(df2)

# 지역과 년도를 동시에 키로 사용
multi_key_merged = pd.merge(df1, df2, on=['지역', '년도'], how='left')

print("\n병합 결과 (지역 + 년도 기준):")
print(multi_key_merged)

print("\n💡 다중 키 병합:")
print("- 여러 컬럼이 모두 일치해야 병합")
print("- 시계열 + 지역 데이터 병합에 유용")
```

### 실습 8: 실무 예제 - 고객 등급별 주문 분석

```python
print("\n" + "=" * 80)
print("📌 실습 8: 실무 예제 - 고객 등급별 주문 분석")
print("=" * 80)

# 고객과 주문 병합
customer_orders = pd.merge(customers, orders, on='고객ID', how='inner')

# 고객 등급별 주문 통계
grade_analysis = customer_orders.groupby('등급').agg({
    '주문ID': 'count',
    '주문금액': ['sum', 'mean']
}).reset_index()

grade_analysis.columns = ['고객등급', '주문건수', '총주문금액', '평균주문금액']

print("\n고객 등급별 주문 분석:")
print(grade_analysis)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 그래프 1: 등급별 총 주문금액
axes[0].bar(grade_analysis['고객등급'], grade_analysis['총주문금액'], 
            color='skyblue', alpha=0.7)
axes[0].set_title('고객 등급별 총 주문금액', fontsize=14, fontweight='bold')
axes[0].set_xlabel('고객 등급', fontsize=12)
axes[0].set_ylabel('총 주문금액 (원)', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# 그래프 2: 등급별 평균 주문금액
axes[1].bar(grade_analysis['고객등급'], grade_analysis['평균주문금액'], 
            color='coral', alpha=0.7)
axes[1].set_title('고객 등급별 평균 주문금액', fontsize=14, fontweight='bold')
axes[1].set_xlabel('고객 등급', fontsize=12)
axes[1].set_ylabel('평균 주문금액 (원)', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/customer_grade_analysis.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/customer_grade_analysis.png")
plt.show()

# VIP 고객 분석
vip_customers = customer_orders[customer_orders['등급'] == 'VIP']
print(f"\nVIP 고객 수: {vip_customers['고객ID'].nunique()}명")
print(f"VIP 총 주문금액: {vip_customers['주문금액'].sum():,}원")
print(f"VIP 평균 주문금액: {vip_customers['주문금액'].mean():,.0f}원")
```

### 실습 9: concat()을 이용한 수직/수평 결합

```python
print("\n" + "=" * 80)
print("📌 실습 9: concat()을 이용한 결합")
print("=" * 80)

# merge vs concat 차이
# - merge: 공통 키 기준으로 병합 (SQL JOIN)
# - concat: 단순히 붙이기 (행 또는 열 방향)

# 예제 데이터
df_2023 = pd.DataFrame({
    '제품': ['A', 'B', 'C'],
    '판매량_2023': [100, 150, 200]
})

df_2024 = pd.DataFrame({
    '제품': ['A', 'B', 'C'],
    '판매량_2024': [120, 180, 210]
})

print("\ndf_2023:")
print(df_2023)
print("\ndf_2024:")
print(df_2024)

# 방법 1: 수평 결합 (열 추가)
concat_horizontal = pd.concat([df_2023, df_2024['판매량_2024']], axis=1)
print("\n수평 결합 (axis=1):")
print(concat_horizontal)

# 방법 2: 수직 결합 (행 추가)
df_2023_long = df_2023.rename(columns={'판매량_2023': '판매량'})
df_2023_long['년도'] = 2023
df_2024_long = df_2024.rename(columns={'판매량_2024': '판매량'})
df_2024_long['년도'] = 2024

concat_vertical = pd.concat([df_2023_long, df_2024_long], axis=0, ignore_index=True)
print("\n수직 결합 (axis=0):")
print(concat_vertical)

print("\n💡 concat() 활용:")
print("- axis=0: 행 방향 결합 (아래로 붙이기)")
print("- axis=1: 열 방향 결합 (옆으로 붙이기)")
print("- ignore_index=True: 인덱스 재설정")
```

---

## 🧩 퀴즈

### 문제 1
Inner Join과 Outer Join의 차이는?

<details>
<summary>정답 보기</summary>

**답변:**

| 구분 | Inner Join | Outer Join |
|------|-----------|-----------|
| **개념** | 교집합 | 합집합 |
| **결과** | 양쪽 모두에 있는 키만 | 양쪽 모든 키 포함 |
| **매칭 안되면** | 해당 행 제외 | NaN으로 채움 |
| **행 개수** | 적음 | 많음 |

**예시:**
```python
# Inner: 주문이 있는 고객만
# Outer: 주문 유무와 관계없이 모든 고객 + 모든 주문
```

**실무 활용:**
- **Inner**: 확실한 매칭 데이터만 분석할 때
- **Outer**: 누락 데이터까지 파악하고 싶을 때
</details>

---

### 문제 2
다음 코드의 결과는?

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
result = pd.merge(df1, df2, on='key', how='inner')
```

결과 DataFrame의 행 개수는?

<details>
<summary>정답 보기</summary>

**정답: 2개 행**

**해설:**
```
df1:
key  value1
A    1
B    2
C    3

df2:
key  value2
B    4
C    5
D    6

Inner Join 결과:
key  value1  value2
B    2       4
C    3       5
```

- Inner Join은 양쪽 모두에 있는 키만 남김
- 'B'와 'C'만 양쪽에 존재
- 따라서 **2개 행**
</details>

---

### 문제 3
Left Join 시 오른쪽 DataFrame에 매칭되는 데이터가 없으면 어떻게 되나요?

1. 해당 행이 삭제됨
2. 에러 발생
3. 해당 컬럼이 NaN으로 채워짐
4. 0으로 채워짐

<details>
<summary>정답 보기</summary>

**정답: 3번 해당 컬럼이 NaN으로 채워짐**

**해설:**
```python
customers:          orders:
고객ID  이름          고객ID  금액
C001   홍길동        C001   10000
C002   김철수

Left Join 결과:
고객ID  이름    금액
C001   홍길동  10000
C002   김철수  NaN     <- 주문이 없어서 NaN
```

- Left Join은 왼쪽 데이터는 무조건 유지
- 오른쪽에 매칭되는 데이터가 없으면 **NaN**으로 표시
- 이후 `fillna(0)` 등으로 처리 가능
</details>

---

### 문제 4
다음 중 `merge()`와 `concat()`의 차이를 올바르게 설명한 것은?

<details>
<summary>정답 보기</summary>

**답변:**

| 구분 | merge() | concat() |
|------|---------|----------|
| **목적** | 공통 키 기준 병합 | 단순 결합 |
| **유사 개념** | SQL JOIN | UNION (수직) / Append (수평) |
| **키 필요** | 필요 (on 파라미터) | 불필요 |
| **사용 상황** | 관계형 데이터 병합 | 같은 구조 데이터 합치기 |

**merge() 예시:**
```python
# 고객ID 기준으로 고객 정보 + 주문 정보 병합
pd.merge(customers, orders, on='고객ID')
```

**concat() 예시:**
```python
# 2023년 데이터 + 2024년 데이터를 아래로 붙이기
pd.concat([df_2023, df_2024], axis=0)
```

💡 **기억하기:**
- **merge = 키 기준 결합** (관계가 있는 데이터)
- **concat = 단순 붙이기** (독립적인 데이터)
</details>

---

### 문제 5
실무 상황: 다음 중 어떤 Join을 사용해야 할까요?

**상황 1:** 주문 데이터에 고객 정보를 추가하고 싶다. 모든 주문은 유지되어야 한다.

**상황 2:** 이번 달에 구매한 고객만 분석하고 싶다.

**상황 3:** 전체 고객 중 누가 구매했고 누가 구매하지 않았는지 파악하고 싶다.

<details>
<summary>정답 보기</summary>

**정답:**

**상황 1: Right Join 또는 Left Join**
```python
# 주문 기준으로 병합 (모든 주문 유지)
pd.merge(customers, orders, on='고객ID', how='right')
# 또는
pd.merge(orders, customers, on='고객ID', how='left')
```
- 주문이 왼쪽에 있으면 Left, 오른쪽에 있으면 Right

**상황 2: Inner Join**
```python
# 주문이 있는 고객만
pd.merge(customers, orders, on='고객ID', how='inner')
```
- 양쪽 모두에 있는 고객만 필요하므로 Inner Join

**상황 3: Left Join**
```python
# 모든 고객 유지, 주문 없으면 NaN
pd.merge(customers, orders, on='고객ID', how='left')

# 구매 여부 확인
result['구매여부'] = result['주문ID'].notna()
```
- 모든 고객을 보려면 Left Join
- 주문ID가 NaN이면 미구매 고객

💡 **실무 팁:**
- "모든 A를 유지" → A를 왼쪽에 두고 Left Join
- "매칭되는 것만" → Inner Join
- "전체 파악" → Outer Join
</details>

---

## ✅ 3교시 학습 완료 체크리스트

- [ ] 데이터 병합의 개념과 필요성 이해
- [ ] Inner Join (교집합) 이해 및 실습
- [ ] Left Join (왼쪽 기준) 이해 및 실습
- [ ] Right Join (오른쪽 기준) 이해 및 실습
- [ ] Outer Join (합집합) 이해 및 실습
- [ ] 다중 키 병합 실습
- [ ] merge()와 concat()의 차이점 이해
- [ ] 실무 예제 (고객 등급별 분석) 완료
- [ ] 퀴즈 5문제 모두 풀이 완료

---

**이전 학습:** [2교시 - groupby 집계](./2_groupby_analysis.md)  
**다음 학습:** [4교시 - apply 함수 활용](./4_apply_function.md)

**학습 완료일:** _____________  
**소요 시간:** _____________  
**이해도 (1~5):** ⭐⭐⭐⭐⭐
