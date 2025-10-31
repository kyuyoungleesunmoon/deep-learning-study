# 📚 4교시: apply() 함수로 사용자 정의 함수 적용

> **학습 목표:** `apply()` 메서드와 `lambda` 함수를 활용하여 데이터에 사용자 정의 함수를 적용하는 방법을 학습합니다.  
> **소요 시간:** 1시간  
> **사용 데이터:** 제품 가격 데이터 (products_usd.csv)

---

## 🧠 이론 설명

### 4.1 apply()란?

**`apply()`**는 Pandas에서 DataFrame이나 Series의 각 행 또는 열에 함수를 적용하는 강력한 메서드입니다.

#### 📊 실생활 비유
공장에서 제품이 컨베이어 벨트를 따라 이동하면서 각 제품에 동일한 작업(검사, 포장 등)을 수행하는 것과 같습니다.

### 4.2 apply()의 장점

- ✅ **유연성**: 어떤 복잡한 함수도 적용 가능
- ✅ **가독성**: 코드가 간결하고 이해하기 쉬움
- ✅ **재사용성**: 함수를 정의하면 여러 곳에서 재사용 가능

### 4.3 기본 문법

```python
# Series에 적용
df['컬럼'].apply(함수)

# DataFrame에 적용 (행 단위)
df.apply(함수, axis=1)

# DataFrame에 적용 (열 단위)
df.apply(함수, axis=0)
```

### 4.4 lambda 함수

**lambda 함수**는 이름 없는 간단한 함수를 한 줄로 정의하는 방법입니다.

```python
# 일반 함수
def add_ten(x):
    return x + 10

# lambda 함수로 동일한 기능
lambda x: x + 10

# 사용 예시
df['new'] = df['value'].apply(lambda x: x + 10)
```

### 4.5 apply() vs 벡터 연산

| 구분 | 벡터 연산 | apply() |
|------|-----------|---------|
| **속도** | 빠름 (Numpy 기반) | 느림 (Python 반복) |
| **사용 시기** | 간단한 산술 연산 | 복잡한 로직, 조건문 |
| **예시** | `df['A'] * 2` | `df['A'].apply(복잡한함수)` |

💡 **원칙**: 가능하면 벡터 연산을 사용하고, 복잡한 로직이 필요할 때만 apply() 사용

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
products = pd.read_csv('../data/products_usd.csv')

print("=" * 80)
print("📊 제품 가격 데이터셋 (달러 기준)")
print("=" * 80)
print(f"데이터 크기: {products.shape[0]} 행, {products.shape[1]} 열")
print(f"\n컬럼명: {list(products.columns)}")
print(f"\n상위 10개 데이터:")
print(products.head(10))
print(f"\n기초 통계:")
print(products.describe())
```

### 실습 1: 기본 apply() - 환율 변환

```python
print("\n" + "=" * 80)
print("📌 실습 1: 달러를 원화로 변환")
print("=" * 80)

# 환율 설정 (2024년 기준 약 1,300원)
EXCHANGE_RATE = 1300

# 방법 1: 벡터 연산 (권장)
products['가격_원화_벡터'] = products['가격_달러'] * EXCHANGE_RATE

# 방법 2: apply() 사용
def usd_to_krw(usd_price):
    """달러를 원화로 변환하는 함수"""
    return usd_price * EXCHANGE_RATE

products['가격_원화_apply'] = products['가격_달러'].apply(usd_to_krw)

# 방법 3: lambda 함수
products['가격_원화_lambda'] = products['가격_달러'].apply(lambda x: x * EXCHANGE_RATE)

print("\n변환 결과 비교:")
print(products[['제품명', '가격_달러', '가격_원화_벡터', '가격_원화_apply', '가격_원화_lambda']].head())

print("\n💡 세 방법 모두 동일한 결과를 생성합니다!")
print("단, 속도는 벡터 연산이 가장 빠릅니다.")

# 최종 컬럼 정리 (벡터 연산 결과만 유지)
products['가격_원화'] = products['가격_원화_벡터']
products = products.drop(['가격_원화_벡터', '가격_원화_apply', '가격_원화_lambda'], axis=1)
```

### 실습 2: 조건부 로직 적용

```python
print("\n" + "=" * 80)
print("📌 실습 2: 가격대 분류")
print("=" * 80)

# 가격대 분류 함수
def classify_price(krw_price):
    """원화 가격을 기준으로 가격대를 분류"""
    if krw_price < 50000:
        return '저가'
    elif krw_price < 200000:
        return '중가'
    elif krw_price < 400000:
        return '고가'
    else:
        return '프리미엄'

# apply() 적용
products['가격대'] = products['가격_원화'].apply(classify_price)

print("\n가격대별 제품 분포:")
print(products['가격대'].value_counts())

# 시각화
plt.figure(figsize=(10, 6))
price_dist = products['가격대'].value_counts()
colors = ['#90EE90', '#FFD700', '#FF6347', '#9370DB']
plt.bar(price_dist.index, price_dist.values, color=colors, alpha=0.7)
plt.title('가격대별 제품 분포', fontsize=16, fontweight='bold')
plt.xlabel('가격대', fontsize=12)
plt.ylabel('제품 수', fontsize=12)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(price_dist.values):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/price_range_distribution.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/price_range_distribution.png")
plt.show()
```

### 실습 3: 여러 컬럼을 사용하는 함수 (axis=1)

```python
print("\n" + "=" * 80)
print("📌 실습 3: 재고 상태 평가 (여러 컬럼 활용)")
print("=" * 80)

# 재고 상태 평가 함수
def evaluate_stock(row):
    """재고와 가격대를 함께 고려하여 재고 상태 평가"""
    stock = row['재고수량']
    price_range = row['가격대']
    
    if stock == 0:
        return '품절'
    elif stock < 10:
        if price_range in ['고가', '프리미엄']:
            return '적정재고'  # 고가 제품은 재고 적어도 OK
        else:
            return '재고부족'
    elif stock < 30:
        return '적정재고'
    else:
        return '재고과다'

# axis=1: 행(row) 단위로 함수 적용
products['재고상태'] = products.apply(evaluate_stock, axis=1)

print("\n재고 상태별 제품 분포:")
print(products['재고상태'].value_counts())

# 결과 샘플
print("\n제품별 재고 상태 (샘플):")
print(products[['제품명', '재고수량', '가격대', '재고상태']].head(15))

# 가격대별 재고 상태 크로스탭
crosstab = pd.crosstab(products['가격대'], products['재고상태'])
print("\n가격대별 재고 상태 분포:")
print(crosstab)
```

### 실습 4: 문자열 처리

```python
print("\n" + "=" * 80)
print("📌 실습 4: 제품 코드에서 정보 추출")
print("=" * 80)

# 제품 코드에서 번호 추출 (PROD001 -> 1)
products['제품번호'] = products['제품코드'].apply(lambda x: int(x.replace('PROD', '')))

# 제품명을 대문자로 변환
products['제품명_대문자'] = products['제품명'].apply(lambda x: x.upper())

# 카테고리 이름 길이
products['카테고리길이'] = products['카테고리'].apply(len)

print("\n문자열 처리 결과:")
print(products[['제품코드', '제품번호', '제품명', '제품명_대문자', 
                '카테고리', '카테고리길이']].head(10))

# 제품번호 통계
print(f"\n제품번호 범위: {products['제품번호'].min()} ~ {products['제품번호'].max()}")
```

### 실습 5: 복잡한 계산 - 할인가 계산

```python
print("\n" + "=" * 80)
print("📌 실습 5: 재고 수량에 따른 할인가 계산")
print("=" * 80)

def calculate_discount_price(row):
    """
    재고 상황과 가격대를 고려한 할인가 계산
    - 재고과다: 30% 할인
    - 재고부족: 할인 없음 (프리미엄 가격)
    - 품절: 가격 동일
    - 적정재고: 10% 할인
    """
    original_price = row['가격_원화']
    stock_status = row['재고상태']
    
    if stock_status == '재고과다':
        discount_rate = 0.30
    elif stock_status == '재고부족':
        discount_rate = 0.00  # 할인 없음
    elif stock_status == '품절':
        discount_rate = 0.00
    else:  # 적정재고
        discount_rate = 0.10
    
    discounted_price = original_price * (1 - discount_rate)
    return discounted_price

products['할인가'] = products.apply(calculate_discount_price, axis=1)
products['할인율'] = ((products['가격_원화'] - products['할인가']) / products['가격_원화'] * 100).round(1)

print("\n할인가 적용 결과:")
print(products[['제품명', '가격_원화', '재고상태', '할인가', '할인율']].head(15))

# 할인율 통계
print(f"\n평균 할인율: {products['할인율'].mean():.1f}%")
print(f"최대 할인율: {products['할인율'].max():.1f}%")

# 할인액 계산
products['할인액'] = products['가격_원화'] - products['할인가']
total_discount = products['할인액'].sum()
print(f"\n총 할인액 (전체 제품): {total_discount:,.0f}원")
```

### 실습 6: applymap() - DataFrame 전체에 함수 적용

```python
print("\n" + "=" * 80)
print("📌 실습 6: applymap()으로 전체 수치 포맷팅")
print("=" * 80)

# 수치형 컬럼만 선택
numeric_cols = products[['가격_달러', '가격_원화', '재고수량', '할인가', '할인액']]

# 천 단위 구분 포맷팅 함수
def format_number(x):
    """숫자를 천 단위 구분 문자열로 변환"""
    if isinstance(x, (int, float)):
        return f"{x:,.0f}"
    return x

# applymap(): DataFrame의 모든 요소에 함수 적용
# Pandas 2.1.0 이상에서는 map() 사용 권장
try:
    formatted_nums = numeric_cols.map(format_number)
except AttributeError:
    # 구버전 Pandas
    formatted_nums = numeric_cols.applymap(format_number)

print("\n포맷팅된 숫자 (상위 10개):")
print(formatted_nums.head(10))

print("\n💡 applymap():")
print("- DataFrame의 모든 셀에 함수 적용")
print("- 주로 포맷팅, 타입 변환 등에 사용")
print("- Pandas 2.1.0+에서는 map() 권장")
```

### 실습 7: 성능 비교 - 벡터 연산 vs apply()

```python
print("\n" + "=" * 80)
print("📌 실습 7: 성능 비교")
print("=" * 80)

import time

# 대용량 데이터 생성
large_df = pd.DataFrame({
    'value': np.random.randint(1, 1000, 10000)
})

# 방법 1: 벡터 연산
start = time.time()
result1 = large_df['value'] * 2
time_vectorized = time.time() - start

# 방법 2: apply()
start = time.time()
result2 = large_df['value'].apply(lambda x: x * 2)
time_apply = time.time() - start

# 방법 3: for 루프 (비교용, 실제로는 사용 지양)
start = time.time()
result3 = []
for val in large_df['value']:
    result3.append(val * 2)
time_loop = time.time() - start

print(f"\n성능 비교 (10,000개 데이터):")
print(f"1. 벡터 연산: {time_vectorized*1000:.2f}ms")
print(f"2. apply():  {time_apply*1000:.2f}ms")
print(f"3. for 루프:  {time_loop*1000:.2f}ms")

print(f"\n💡 결론:")
print(f"- 벡터 연산이 apply()보다 약 {time_apply/time_vectorized:.1f}배 빠름")
print(f"- apply()가 for 루프보다 약 {time_loop/time_apply:.1f}배 빠름")
print(f"- 가능하면 벡터 연산 사용!")
```

### 실습 8: 실무 예제 - 판매 전략 수립

```python
print("\n" + "=" * 80)
print("📌 실습 8: 실무 예제 - 판매 전략 수립")
print("=" * 80)

def recommend_strategy(row):
    """
    제품별 판매 전략 추천
    재고 상태, 가격대, 카테고리를 종합적으로 고려
    """
    stock = row['재고상태']
    price = row['가격대']
    category = row['카테고리']
    
    strategies = []
    
    # 재고 기반 전략
    if stock == '재고과다':
        strategies.append('할인판매')
    elif stock == '재고부족':
        strategies.append('긴급발주')
    elif stock == '품절':
        strategies.append('재입고대기')
    
    # 가격대 기반 전략
    if price == '프리미엄':
        strategies.append('VIP마케팅')
    elif price == '저가':
        strategies.append('대량판매')
    
    # 카테고리 기반 전략
    if category == '가전':
        strategies.append('AS홍보')
    elif category == '의류':
        strategies.append('시즌이벤트')
    
    return ', '.join(strategies) if strategies else '정상운영'

products['판매전략'] = products.apply(recommend_strategy, axis=1)

print("\n제품별 판매 전략:")
print(products[['제품명', '카테고리', '가격대', '재고상태', '판매전략']].head(20))

# 전략별 제품 수
strategies_list = []
for strategies in products['판매전략']:
    strategies_list.extend(strategies.split(', '))

strategy_counts = pd.Series(strategies_list).value_counts()
print("\n전략별 적용 건수:")
print(strategy_counts)

# 시각화
plt.figure(figsize=(10, 6))
strategy_counts.plot(kind='barh', color='teal', alpha=0.7)
plt.title('판매 전략별 적용 건수', fontsize=16, fontweight='bold')
plt.xlabel('적용 건수', fontsize=12)
plt.ylabel('전략', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../output/sales_strategy_distribution.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/sales_strategy_distribution.png")
plt.show()
```

### 실습 9: 최종 결과 저장

```python
print("\n" + "=" * 80)
print("📌 실습 9: 최종 가공 데이터 저장")
print("=" * 80)

# 최종 결과 컬럼 정리
final_products = products[[
    '제품코드', '제품명', '카테고리', 
    '가격_달러', '가격_원화', '할인가', '할인율',
    '재고수량', '재고상태', '판매전략'
]]

# CSV 저장
output_file = '../output/products_processed.csv'
final_products.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 최종 데이터 저장 완료: {output_file}")
print(f"저장된 데이터 크기: {final_products.shape[0]} 행, {final_products.shape[1]} 열")
print("\n최종 데이터 미리보기:")
print(final_products.head(10))

# 요약 통계
print("\n" + "=" * 80)
print("📊 최종 데이터 요약")
print("=" * 80)
print(f"총 제품 수: {len(final_products)}")
print(f"평균 원화 가격: {final_products['가격_원화'].mean():,.0f}원")
print(f"평균 할인율: {final_products['할인율'].mean():.1f}%")
print(f"\n카테고리별 평균 가격:")
print(final_products.groupby('카테고리')['가격_원화'].mean().round(0))
```

---

## 🧩 퀴즈

### 문제 1
`apply()`와 `map()`의 차이는?

<details>
<summary>정답 보기</summary>

**답변:**

| 구분 | apply() | map() |
|------|---------|-------|
| **적용 대상** | Series, DataFrame | Series만 |
| **사용 방법** | 함수, lambda | 함수, lambda, 딕셔너리 |
| **반환 타입** | Series 또는 DataFrame | Series |

**예시:**
```python
# apply()
df['new'] = df['col'].apply(lambda x: x * 2)

# map() - 딕셔너리 매핑 가능
df['grade'] = df['score'].map({90:'A', 80:'B', 70:'C'})
```

💡 **실무 팁:**
- **apply()**: 복잡한 함수 적용
- **map()**: 간단한 매핑, 치환
</details>

---

### 문제 2
다음 코드의 `axis` 파라미터 의미는?

```python
df.apply(my_function, axis=1)
```

1. 열(column) 단위로 함수 적용
2. 행(row) 단위로 함수 적용
3. 전체 DataFrame에 함수 적용

<details>
<summary>정답 보기</summary>

**정답: 2번 행(row) 단위로 함수 적용**

**해설:**
- `axis=0`: 열(column) 방향, 세로로 적용
- `axis=1`: 행(row) 방향, 가로로 적용

```python
# axis=0 (기본값): 각 열에 함수 적용
df.apply(sum, axis=0)  # 각 열의 합계

# axis=1: 각 행에 함수 적용
df.apply(sum, axis=1)  # 각 행의 합계
```

💡 **기억하는 법:**
- axis=0: 0번째 축(행)을 따라 이동 → 열 방향
- axis=1: 1번째 축(열)을 따라 이동 → 행 방향
</details>

---

### 문제 3
lambda 함수의 장단점은?

<details>
<summary>정답 보기</summary>

**장점:**
- ✅ 간결함: 한 줄로 함수 정의
- ✅ 즉시 사용: 따로 함수 정의 불필요
- ✅ 가독성: 간단한 로직은 오히려 이해하기 쉬움

**단점:**
- ❌ 복잡한 로직 불가: 한 줄 제한
- ❌ 재사용 어려움: 이름이 없어서 다시 호출 불가
- ❌ 디버깅 어려움: 에러 추적이 힘듦

**사용 기준:**
```python
# ✅ Good: 간단한 로직
df['double'] = df['value'].apply(lambda x: x * 2)

# ❌ Bad: 복잡한 로직 (일반 함수 사용 권장)
df['result'] = df.apply(lambda row: 
    row['A'] * 2 if row['B'] > 100 else row['A'] / 2, axis=1)

# ✅ Better: 일반 함수로 정의
def calculate_result(row):
    if row['B'] > 100:
        return row['A'] * 2
    else:
        return row['A'] / 2

df['result'] = df.apply(calculate_result, axis=1)
```
</details>

---

### 문제 4
다음 중 가장 빠른 방법은?

```python
# A. 벡터 연산
df['new'] = df['value'] * 2

# B. apply() + lambda
df['new'] = df['value'].apply(lambda x: x * 2)

# C. for 루프
result = []
for val in df['value']:
    result.append(val * 2)
df['new'] = result
```

<details>
<summary>정답 보기</summary>

**정답: A. 벡터 연산**

**성능 순위:**
1. 🥇 **벡터 연산** (가장 빠름)
   - Numpy C 기반 최적화
   - 10,000건 기준: ~1ms
   
2. 🥈 **apply()**
   - Python 레벨 반복
   - 10,000건 기준: ~50ms
   
3. 🥉 **for 루프** (가장 느림)
   - 순수 Python 반복
   - 10,000건 기준: ~100ms

**실무 가이드:**
```python
# ✅ 1순위: 벡터 연산 (가능하면 항상)
df['new'] = df['A'] + df['B'] * 2

# ✅ 2순위: apply() (복잡한 로직)
df['new'] = df.apply(complex_function, axis=1)

# ❌ 3순위: for 루프 (절대 사용 금지)
# Pandas에서 for 루프는 성능이 매우 나쁨
```
</details>

---

### 문제 5
실무 상황: 다음 중 어떤 방법을 사용해야 할까?

**상황:** 고객 데이터에서 나이(age)를 기준으로 연령대(age_group)를 분류하려고 합니다.
- 0-19세: 청소년
- 20-34세: 청년
- 35-49세: 중년
- 50세 이상: 장년

<details>
<summary>정답 보기</summary>

**방법 1: apply() + 함수 (권장)**
```python
def classify_age_group(age):
    if age < 20:
        return '청소년'
    elif age < 35:
        return '청년'
    elif age < 50:
        return '중년'
    else:
        return '장년'

df['age_group'] = df['age'].apply(classify_age_group)
```

**방법 2: pd.cut() 사용 (더 권장!)**
```python
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 20, 35, 50, 100],
    labels=['청소년', '청년', '중년', '장년'],
    right=False
)
```

**방법 3: np.select() 사용**
```python
conditions = [
    df['age'] < 20,
    (df['age'] >= 20) & (df['age'] < 35),
    (df['age'] >= 35) & (df['age'] < 50),
    df['age'] >= 50
]
choices = ['청소년', '청년', '중년', '장년']
df['age_group'] = np.select(conditions, choices)
```

**💡 권장 순위:**
1. `pd.cut()`: 구간 분할에 최적화, 가장 빠름
2. `np.select()`: 복잡한 조건, 빠름
3. `apply()`: 가장 유연하지만 느림

**결론:** 단순 구간 분류는 `pd.cut()`, 복잡한 조건은 `np.select()`, 매우 복잡한 로직은 `apply()` 사용!
</details>

---

## ✅ 4교시 학습 완료 체크리스트

- [ ] apply() 메서드의 개념과 동작 원리 이해
- [ ] lambda 함수 작성 및 활용
- [ ] Series에 apply() 적용 (단일 컬럼)
- [ ] DataFrame에 apply() 적용 (axis=1, 여러 컬럼)
- [ ] 조건부 로직을 포함한 복잡한 함수 작성
- [ ] 벡터 연산 vs apply() 성능 차이 이해
- [ ] 실무 예제 (가격 변환, 재고 관리, 판매 전략) 완료
- [ ] 퀴즈 5문제 모두 풀이 완료

---

**이전 학습:** [3교시 - 데이터 병합](./3_merge_join.md)  
**다음 학습:** [5교시 - 종합 실습 (1) 데이터 불러오기 및 구조 파악](./5_data_cleaning.md)

**학습 완료일:** _____________  
**소요 시간:** _____________  
**이해도 (1~5):** ⭐⭐⭐⭐⭐
