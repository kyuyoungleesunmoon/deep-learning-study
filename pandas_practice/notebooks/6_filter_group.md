# 📚 6교시: 종합 실습 (2) 필터링 및 그룹핑

> **학습 목표:** 5교시에서 파악한 데이터를 조건별로 필터링하고 다양한 기준으로 그룹화하여 심화 분석을 수행합니다.  
> **소요 시간:** 1시간  
> **사용 데이터:** 5교시에서 로드한 공공 비즈니스 데이터

---

## 🧠 이론 설명

### 6.1 필터링의 중요성

필터링은 전체 데이터에서 **분석 목적에 맞는 부분집합**을 추출하는 과정입니다.

#### 실무 예시
- 고객 분석: "구매액 100만원 이상 고객"
- 매출 분석: "2024년 상반기 매출"
- 지역 분석: "서울 + 경기 지역 기업"

### 6.2 복합 조건 필터링

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `&` | AND (그리고) | `(df['A'] > 10) & (df['B'] < 20)` |
| `\|` | OR (또는) | `(df['지역'] == '서울') \| (df['지역'] == '경기')` |
| `~` | NOT (부정) | `~(df['상태'] == '폐업')` |

⚠️ **주의**: 각 조건을 괄호 `()`로 반드시 묶어야 합니다!

### 6.3 그룹 분석의 활용

- **월별 분석**: 시간에 따른 추세 파악
- **지역별 분석**: 지역간 차이 비교
- **카테고리별 분석**: 업종, 등급 등의 특성 비교

---

## 💻 실습 코드

### 환경 설정 및 데이터 로드

```python
# 필수 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
df = pd.read_csv('../data/public_business_data.csv', encoding='utf-8-sig')

print("=" * 80)
print("🎯 6교시: 필터링 및 그룹 분석")
print("=" * 80)
print(f"데이터 크기: {df.shape[0]:,} 행 × {df.shape[1]:,} 열")
```

### 실습 1: 단순 조건 필터링

```python
print("\n" + "=" * 80)
print("📌 실습 1: 단순 조건 필터링")
print("=" * 80)

# 1. 매출액 1억 이상 기업
print("\n[1] 매출액 1억 이상 기업")
print("-" * 80)
high_revenue = df[df['매출액'] >= 100_000_000]
print(f"필터링 결과: {len(high_revenue):,}개 ({len(high_revenue)/len(df)*100:.1f}%)")
print("\n상위 5개 기업:")
print(high_revenue.sort_values('매출액', ascending=False)[['지역', '업종', '매출액', '종업원수']].head())

# 2. 서울 지역 기업
print("\n[2] 서울 지역 기업")
print("-" * 80)
seoul_companies = df[df['지역'] == '서울']
print(f"필터링 결과: {len(seoul_companies):,}개 ({len(seoul_companies)/len(df)*100:.1f}%)")

# 3. 제조업 기업
print("\n[3] 제조업 기업")
print("-" * 80)
manufacturing = df[df['업종'] == '제조업']
print(f"필터링 결과: {len(manufacturing):,}개 ({len(manufacturing)/len(df)*100:.1f}%)")
print(f"제조업 평균 매출액: {manufacturing['매출액'].mean():,.0f}원")
print(f"제조업 평균 종업원수: {manufacturing['종업원수'].mean():.1f}명")
```

### 실습 2: 복합 조건 필터링 (AND)

```python
print("\n" + "=" * 80)
print("📌 실습 2: 복합 조건 필터링 (AND)")
print("=" * 80)

# 서울 지역 AND 매출액 5천만원 이상
print("\n[1] 서울 + 매출액 5천만원 이상")
print("-" * 80)
condition1 = (df['지역'] == '서울') & (df['매출액'] >= 50_000_000)
result1 = df[condition1]
print(f"필터링 결과: {len(result1):,}개")

# 제조업 AND 종업원 100명 이상
print("\n[2] 제조업 + 종업원 100명 이상")
print("-" * 80)
condition2 = (df['업종'] == '제조업') & (df['종업원수'] >= 100)
result2 = df[condition2]
print(f"필터링 결과: {len(result2):,}개")
print("\n샘플 데이터:")
print(result2[['지역', '업종', '매출액', '종업원수']].head())

# 수출 기업 AND 인증 보유
print("\n[3] 수출 기업 + 인증 보유")
print("-" * 80)
condition3 = (df['수출여부'] == 'Y') & (df['인증보유'].notna())
result3 = df[condition3]
print(f"필터링 결과: {len(result3):,}개")
```

### 실습 3: 복합 조건 필터링 (OR)

```python
print("\n" + "=" * 80)
print("📌 실습 3: 복합 조건 필터링 (OR)")
print("=" * 80)

# 서울 OR 경기 지역
print("\n[1] 서울 OR 경기")
print("-" * 80)
condition1 = (df['지역'] == '서울') | (df['지역'] == '경기')
result1 = df[condition1]
print(f"필터링 결과: {len(result1):,}개 ({len(result1)/len(df)*100:.1f}%)")

# isin() 메서드 사용 (더 간결)
result1_alt = df[df['지역'].isin(['서울', '경기'])]
print(f"isin() 사용 결과: {len(result1_alt):,}개")

# 매출액 1억 이상 OR 종업원 200명 이상
print("\n[2] 매출액 1억 이상 OR 종업원 200명 이상")
print("-" * 80)
condition2 = (df['매출액'] >= 100_000_000) | (df['종업원수'] >= 200)
result2 = df[condition2]
print(f"필터링 결과: {len(result2):,}개")
```

### 실습 4: 부정 조건 (NOT)

```python
print("\n" + "=" * 80)
print("📌 실습 4: 부정 조건 (NOT)")
print("=" * 80)

# 서울이 아닌 지역
print("\n[1] 서울이 아닌 지역")
print("-" * 80)
not_seoul = df[~(df['지역'] == '서울')]
# 또는
not_seoul_alt = df[df['지역'] != '서울']
print(f"필터링 결과: {len(not_seoul):,}개")

# 수출하지 않는 기업
print("\n[2] 수출하지 않는 기업")
print("-" * 80)
not_export = df[df['수출여부'] != 'Y']
print(f"필터링 결과: {len(not_export):,}개 ({len(not_export)/len(df)*100:.1f}%)")

# 인증이 없는 기업
print("\n[3] 인증이 없는 기업")
print("-" * 80)
no_cert = df[df['인증보유'] == '없음']
print(f"필터링 결과: {len(no_cert):,}개 ({len(no_cert)/len(df)*100:.1f}%)")
```

### 실습 5: 지역별 집계 분석

```python
print("\n" + "=" * 80)
print("📌 실습 5: 지역별 집계 분석")
print("=" * 80)

# 지역별 통계
region_stats = df.groupby('지역').agg({
    '매출액': ['count', 'sum', 'mean', 'median'],
    '종업원수': ['mean', 'sum']
}).round(0)

region_stats.columns = ['기업수', '총매출', '평균매출', '중앙매출', '평균종업원', '총종업원']
region_stats = region_stats.sort_values('총매출', ascending=False)

print("\n지역별 통계:")
print(region_stats)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 지역별 기업 수
axes[0, 0].bar(region_stats.index, region_stats['기업수'], color='skyblue', alpha=0.7)
axes[0, 0].set_title('지역별 기업 수', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('지역', fontsize=12)
axes[0, 0].set_ylabel('기업 수', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. 지역별 총 매출
axes[0, 1].bar(region_stats.index, region_stats['총매출']/1e8, color='lightcoral', alpha=0.7)
axes[0, 1].set_title('지역별 총 매출 (억원)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('지역', fontsize=12)
axes[0, 1].set_ylabel('총 매출 (억원)', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. 지역별 평균 매출
axes[1, 0].bar(region_stats.index, region_stats['평균매출']/1e6, color='lightgreen', alpha=0.7)
axes[1, 0].set_title('지역별 평균 매출 (백만원)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('지역', fontsize=12)
axes[1, 0].set_ylabel('평균 매출 (백만원)', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. 지역별 평균 종업원
axes[1, 1].bar(region_stats.index, region_stats['평균종업원'], color='plum', alpha=0.7)
axes[1, 1].set_title('지역별 평균 종업원 수', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('지역', fontsize=12)
axes[1, 1].set_ylabel('평균 종업원 (명)', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/region_analysis.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/region_analysis.png")
plt.show()
```

### 실습 6: 업종별 집계 분석

```python
print("\n" + "=" * 80)
print("📌 실습 6: 업종별 집계 분석")
print("=" * 80)

# 업종별 통계
industry_stats = df.groupby('업종').agg({
    '매출액': ['count', 'mean', 'median', 'std'],
    '종업원수': ['mean', 'median'],
    '수출여부': lambda x: (x == 'Y').sum()  # 수출 기업 수
}).round(0)

industry_stats.columns = ['기업수', '평균매출', '중앙매출', '매출표준편차', 
                          '평균종업원', '중앙종업원', '수출기업수']
industry_stats = industry_stats.sort_values('평균매출', ascending=False)

print("\n업종별 통계:")
print(industry_stats)

# 수출 비율 추가
industry_stats['수출비율(%)'] = (industry_stats['수출기업수'] / industry_stats['기업수'] * 100).round(1)

print("\n업종별 수출 비율:")
print(industry_stats[['기업수', '수출기업수', '수출비율(%)']].sort_values('수출비율(%)', ascending=False))
```

### 실습 7: 다중 그룹화 (지역 + 업종)

```python
print("\n" + "=" * 80)
print("📌 실습 7: 다중 그룹화 (지역 + 업종)")
print("=" * 80)

# 지역별, 업종별 교차 분석
cross_analysis = df.groupby(['지역', '업종']).agg({
    '매출액': ['count', 'mean'],
    '종업원수': 'mean'
}).round(0)

cross_analysis.columns = ['기업수', '평균매출', '평균종업원']
cross_analysis = cross_analysis.sort_values('평균매출', ascending=False)

print("\n지역 × 업종 교차 분석 (상위 15개):")
print(cross_analysis.head(15))

# 피벗 테이블로 변환
pivot_table = df.pivot_table(
    values='매출액',
    index='지역',
    columns='업종',
    aggfunc='mean',
    fill_value=0
).round(0)

print("\n피벗 테이블 (평균 매출액):")
print(pivot_table)

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table/1e6, annot=True, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': '평균 매출액 (백만원)'})
plt.title('지역 × 업종별 평균 매출액 (백만원)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('업종', fontsize=12)
plt.ylabel('지역', fontsize=12)
plt.tight_layout()
plt.savefig('../output/heatmap_region_industry.png', dpi=100, bbox_inches='tight')
print("\n📊 히트맵 저장 완료: ../output/heatmap_region_industry.png")
plt.show()
```

### 실습 8: 월별 추이 분석

```python
print("\n" + "=" * 80)
print("📌 실습 8: 월별 추이 분석")
print("=" * 80)

# 접수일자를 datetime으로 변환
df['접수일자'] = pd.to_datetime(df['접수일자'])

# 월 추출
df['월'] = df['접수일자'].dt.month

# 월별 통계
monthly_stats = df.groupby('월').agg({
    '매출액': ['count', 'mean', 'sum'],
    '종업원수': 'mean'
}).round(0)

monthly_stats.columns = ['접수건수', '평균매출', '총매출', '평균종업원']

print("\n월별 통계:")
print(monthly_stats)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 월별 접수 건수
axes[0].plot(monthly_stats.index, monthly_stats['접수건수'], 
             marker='o', linewidth=2, markersize=8, color='blue')
axes[0].set_title('월별 접수 건수 추이', fontsize=14, fontweight='bold')
axes[0].set_xlabel('월', fontsize=12)
axes[0].set_ylabel('접수 건수', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(monthly_stats.index)

# 월별 평균 매출
axes[1].plot(monthly_stats.index, monthly_stats['평균매출']/1e6, 
             marker='s', linewidth=2, markersize=8, color='green')
axes[1].set_title('월별 평균 매출 추이 (백만원)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('월', fontsize=12)
axes[1].set_ylabel('평균 매출 (백만원)', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(monthly_stats.index)

plt.tight_layout()
plt.savefig('../output/monthly_trend.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/monthly_trend.png")
plt.show()
```

### 실습 9: 고급 필터링 - query() 메서드

```python
print("\n" + "=" * 80)
print("📌 실습 9: 고급 필터링 - query() 메서드")
print("=" * 80)

# query() 메서드: SQL과 유사한 문법
print("\n[1] 기본 query 예제")
print("-" * 80)

# 매출액 5천만원 이상
result1 = df.query('매출액 >= 50_000_000')
print(f"매출액 5천만원 이상: {len(result1):,}개")

# 서울이면서 제조업
result2 = df.query('지역 == "서울" and 업종 == "제조업"')
print(f"서울 + 제조업: {len(result2):,}개")

# 매출액 상위 25% (복잡한 조건)
q75 = df['매출액'].quantile(0.75)
result3 = df.query(f'매출액 >= {q75}')
print(f"매출액 상위 25%: {len(result3):,}개")

print("\n💡 query() 장점:")
print("- SQL 스타일 문법으로 가독성이 좋음")
print("- 복잡한 조건도 간결하게 표현")
print("- 변수를 @기호로 참조 가능")
```

### 실습 10: 종합 분석 - 기업 규모별 분류

```python
print("\n" + "=" * 80)
print("📌 실습 10: 종합 분석 - 기업 규모별 분류")
print("=" * 80)

# 기업 규모 분류 함수
def classify_company_size(row):
    """매출액과 종업원수를 기준으로 기업 규모 분류"""
    revenue = row['매출액']
    employees = row['종업원수']
    
    if pd.isna(revenue) or pd.isna(employees):
        return '정보부족'
    
    if revenue >= 100_000_000 and employees >= 100:
        return '대기업'
    elif revenue >= 50_000_000 or employees >= 50:
        return '중기업'
    elif revenue >= 10_000_000:
        return '소기업'
    else:
        return '영세기업'

# 규모 분류 적용
df['기업규모'] = df.apply(classify_company_size, axis=1)

# 규모별 통계
size_stats = df.groupby('기업규모').agg({
    '매출액': ['count', 'mean'],
    '종업원수': 'mean',
    '수출여부': lambda x: (x == 'Y').sum()
})

size_stats.columns = ['기업수', '평균매출', '평균종업원', '수출기업수']
size_stats = size_stats.round(0)

print("\n기업 규모별 통계:")
print(size_stats)

# 규모별 수출 비율
size_stats['수출비율(%)'] = (size_stats['수출기업수'] / size_stats['기업수'] * 100).round(1)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 원형 차트: 규모별 기업 수
sizes = size_stats['기업수']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
explode = (0.1, 0, 0, 0, 0)  # 첫 번째 조각 강조

axes[0].pie(sizes, labels=sizes.index, autopct='%1.1f%%', 
            startangle=90, colors=colors[:len(sizes)], explode=explode[:len(sizes)])
axes[0].set_title('기업 규모별 분포', fontsize=14, fontweight='bold')

# 막대 차트: 규모별 수출 비율
axes[1].bar(size_stats.index, size_stats['수출비율(%)'], color='teal', alpha=0.7)
axes[1].set_title('기업 규모별 수출 비율', fontsize=14, fontweight='bold')
axes[1].set_xlabel('기업 규모', fontsize=12)
axes[1].set_ylabel('수출 비율 (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/company_size_analysis.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/company_size_analysis.png")
plt.show()

# 핵심 인사이트
print("\n💡 핵심 인사이트:")
print("-" * 80)
print(f"1. 가장 많은 기업 규모: {size_stats['기업수'].idxmax()} ({size_stats['기업수'].max()}개)")
print(f"2. 평균 매출이 가장 높은 규모: {size_stats['평균매출'].idxmax()} ({size_stats['평균매출'].max():,.0f}원)")
print(f"3. 수출 비율이 가장 높은 규모: {size_stats['수출비율(%)'].idxmax()} ({size_stats['수출비율(%)'].max():.1f}%)")
```

---

## 🧩 퀴즈

### 문제 1
복합 조건 필터링에서 반드시 필요한 것은?

<details>
<summary>정답 보기</summary>

**정답: 각 조건을 괄호 `()`로 묶기**

**올바른 예:**
```python
df[(df['A'] > 10) & (df['B'] < 20)]
```

**잘못된 예:**
```python
df[df['A'] > 10 & df['B'] < 20]  # 에러 발생!
```

**이유:**
Python의 연산자 우선순위 때문에 괄호가 필수입니다.
`&`, `|`, `~` 연산자의 우선순위가 비교 연산자보다 높아서 괄호 없이 사용하면 예상과 다른 결과가 나옵니다.
</details>

---

### 문제 2
다음 중 `isin()` 메서드의 용도는?

<details>
<summary>정답 보기</summary>

**답변: 특정 값 목록에 포함되는지 확인**

**예시:**
```python
# OR 조건을 간결하게 표현
# 방법 1: 복잡함
df[(df['지역'] == '서울') | (df['지역'] == '경기') | (df['지역'] == '부산')]

# 방법 2: isin() 사용 (권장)
df[df['지역'].isin(['서울', '경기', '부산'])]
```

💡 **실무 팁**: 여러 값 중 하나를 찾을 때는 `isin()`이 가장 깔끔합니다!
</details>

---

### 문제 3
`pivot_table()`과 `groupby()`의 차이는?

<details>
<summary>정답 보기</summary>

**답변:**

| 구분 | groupby() | pivot_table() |
|------|-----------|---------------|
| **용도** | 그룹별 집계 | 교차 테이블 생성 |
| **결과 형태** | 긴 형태 (Long) | 넓은 형태 (Wide) |
| **시각적 이해** | 목록 형태 | 표 형태 |

**예시:**

**groupby() 결과:**
```
지역    업종     평균매출
서울    제조업   1000
서울    서비스업  800
경기    제조업   900
```

**pivot_table() 결과:**
```
업종     제조업  서비스업
지역
서울     1000    800
경기     900     750
```

💡 **선택 기준:**
- 간단한 집계: `groupby()`
- 교차 분석, 히트맵: `pivot_table()`
</details>

---

## ✅ 6교시 학습 완료 체크리스트

- [ ] 단순 조건 필터링 수행
- [ ] 복합 조건 필터링 (AND, OR, NOT) 수행
- [ ] `isin()` 메서드로 다중 값 필터링
- [ ] 지역별, 업종별 집계 분석
- [ ] 다중 그룹화 (2개 이상 기준)
- [ ] `pivot_table()`로 교차 분석
- [ ] 월별 추이 분석
- [ ] `query()` 메서드 사용
- [ ] 퀴즈 3문제 모두 풀이 완료

---

**이전 학습:** [5교시 - 데이터 구조 파악](./5_data_cleaning.md)  
**다음 학습:** [7교시 - 종합 실습 (3) 결과 저장](./7_save_output.md)

**학습 완료일:** _____________  
**소요 시간:** _____________  
**이해도 (1~5):** ⭐⭐⭐⭐⭐
