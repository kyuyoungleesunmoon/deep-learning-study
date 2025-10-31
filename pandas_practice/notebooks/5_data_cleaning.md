# 📚 5교시: 종합 실습 (1) 데이터 불러오기 및 구조 파악

> **학습 목표:** 실제 공공데이터를 불러와서 데이터의 구조를 파악하고 품질을 평가하는 방법을 학습합니다.  
> **소요 시간:** 1시간  
> **사용 데이터:** 공공 비즈니스 데이터 (public_business_data.csv)

---

## 🧠 이론 설명

### 5.1 데이터 분석 프로세스

```
1. 데이터 수집 ← 우리는 여기부터 시작
2. 데이터 이해 및 탐색 (EDA)
3. 데이터 정제 및 가공
4. 데이터 분석
5. 결과 시각화 및 보고
```

### 5.2 데이터 구조 파악의 중요성

실전 데이터 분석에서 가장 많은 시간을 소비하는 단계입니다 (전체 시간의 60~80%).

#### 주요 확인 사항
1. **데이터 크기**: 행과 열의 개수
2. **데이터 타입**: 숫자형, 문자형, 날짜형 등
3. **결측치 현황**: 어느 컬럼에 얼마나 있는지
4. **데이터 분포**: 최솟값, 최댓값, 평균, 중앙값
5. **이상치**: 비정상적으로 크거나 작은 값
6. **중복 데이터**: 동일한 레코드가 여러 번 있는지

### 5.3 데이터 품질 평가 기준

| 기준 | 설명 | 평가 방법 |
|------|------|----------|
| **완전성** | 결측치 비율 | `isnull().sum()` |
| **정확성** | 논리적 오류 여부 | 범위 체크, 타입 체크 |
| **일관성** | 데이터 형식 통일 | 중복값, 형식 확인 |
| **최신성** | 데이터 갱신 시점 | 날짜 컬럼 확인 |

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

print("=" * 80)
print("🎯 데이터 분석 시작")
print("=" * 80)
print(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

### 실습 1: 데이터 불러오기

```python
print("\n" + "=" * 80)
print("📌 실습 1: 데이터 불러오기")
print("=" * 80)

# 데이터 로드
data_file = '../data/public_business_data.csv'
df = pd.read_csv(data_file, encoding='utf-8-sig')

print(f"\n✅ 데이터 로드 완료: {data_file}")
print(f"📊 데이터 크기: {df.shape[0]:,} 행 × {df.shape[1]:,} 열")

# 메모리 사용량 확인
memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
print(f"💾 메모리 사용량: {memory_usage:.2f} MB")
```

### 실습 2: 기본 정보 확인

```python
print("\n" + "=" * 80)
print("📌 실습 2: 기본 정보 확인")
print("=" * 80)

print("\n[1] 컬럼명 확인")
print("-" * 80)
print(f"컬럼 개수: {len(df.columns)}")
print(f"컬럼 목록:\n{list(df.columns)}")

print("\n[2] 데이터 타입 확인")
print("-" * 80)
print(df.dtypes)
print(f"\n타입별 컬럼 수:")
print(df.dtypes.value_counts())

print("\n[3] 상위 5개 데이터")
print("-" * 80)
print(df.head())

print("\n[4] 하위 5개 데이터")
print("-" * 80)
print(df.tail())

print("\n[5] 무작위 샘플 5개")
print("-" * 80)
print(df.sample(5))
```

### 실습 3: 결측치 현황 분석

```python
print("\n" + "=" * 80)
print("📌 실습 3: 결측치 현황 분석")
print("=" * 80)

# 결측치 개수
missing_counts = df.isnull().sum()
missing_ratio = (missing_counts / len(df) * 100).round(2)

# 결측치 요약 테이블
missing_summary = pd.DataFrame({
    '컬럼명': missing_counts.index,
    '결측치개수': missing_counts.values,
    '결측치비율(%)': missing_ratio.values,
    '유효데이터': len(df) - missing_counts.values
})

# 결측치가 있는 컬럼만 필터링
missing_summary = missing_summary[missing_summary['결측치개수'] > 0].sort_values(
    by='결측치비율(%)', ascending=False
)

print("\n결측치 현황:")
print(missing_summary.to_string(index=False))

# 결측치 시각화
if len(missing_summary) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 그래프 1: 결측치 개수
    axes[0].barh(missing_summary['컬럼명'], missing_summary['결측치개수'], 
                 color='coral', alpha=0.7)
    axes[0].set_xlabel('결측치 개수', fontsize=12)
    axes[0].set_title('컬럼별 결측치 개수', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 그래프 2: 결측치 비율
    axes[1].barh(missing_summary['컬럼명'], missing_summary['결측치비율(%)'], 
                 color='skyblue', alpha=0.7)
    axes[1].set_xlabel('결측치 비율 (%)', fontsize=12)
    axes[1].set_title('컬럼별 결측치 비율', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../output/missing_analysis.png', dpi=100, bbox_inches='tight')
    print("\n📊 시각화 저장 완료: ../output/missing_analysis.png")
    plt.show()

# 결측치 품질 평가
print("\n💡 결측치 품질 평가:")
high_missing = missing_summary[missing_summary['결측치비율(%)'] > 30]
if len(high_missing) > 0:
    print(f"⚠️  경고: {len(high_missing)}개 컬럼의 결측치가 30% 초과")
    print(high_missing[['컬럼명', '결측치비율(%)']].to_string(index=False))
else:
    print("✅ 모든 컬럼의 결측치 비율이 30% 이하입니다.")
```

### 실습 4: 기초 통계 분석

```python
print("\n" + "=" * 80)
print("📌 실습 4: 기초 통계 분석")
print("=" * 80)

# 수치형 컬럼만 선택
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n수치형 컬럼 ({len(numeric_cols)}개):")
print(numeric_cols)

# 기초 통계량
print("\n기초 통계량:")
print(df[numeric_cols].describe())

# 사분위수 추가 정보
print("\n사분위수 상세:")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q2 = df[col].quantile(0.50)  # 중앙값
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    print(f"\n{col}:")
    print(f"  Q1 (25%): {q1:,.2f}")
    print(f"  Q2 (중앙값): {q2:,.2f}")
    print(f"  Q3 (75%): {q3:,.2f}")
    print(f"  IQR: {iqr:,.2f}")
    print(f"  이상치 범위: [{q1 - 1.5*iqr:,.2f}, {q3 + 1.5*iqr:,.2f}]")
```

### 실습 5: 범주형 데이터 분석

```python
print("\n" + "=" * 80)
print("📌 실습 5: 범주형 데이터 분석")
print("=" * 80)

# 범주형(문자열) 컬럼 선택
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# 날짜 컬럼 제외
categorical_cols = [col for col in categorical_cols if col != '접수일자']

print(f"\n범주형 컬럼 ({len(categorical_cols)}개):")
print(categorical_cols)

# 각 범주형 컬럼의 고유값 개수
print("\n범주형 컬럼별 고유값 개수:")
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count}개")
    
    # 상위 5개 빈도수
    print(f"  Top 5 빈도:")
    print(df[col].value_counts().head().to_string())
    print()
```

### 실습 6: 데이터 분포 시각화

```python
print("\n" + "=" * 80)
print("📌 실습 6: 데이터 분포 시각화")
print("=" * 80)

# 매출액 분포 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 히스토그램
axes[0, 0].hist(df['매출액'].dropna(), bins=30, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('매출액 분포 (히스토그램)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('매출액 (원)', fontsize=12)
axes[0, 0].set_ylabel('빈도', fontsize=12)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. 박스플롯
axes[0, 1].boxplot(df['매출액'].dropna(), vert=True)
axes[0, 1].set_title('매출액 분포 (박스플롯)', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('매출액 (원)', fontsize=12)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. 종업원수 히스토그램
axes[1, 0].hist(df['종업원수'].dropna(), bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('종업원수 분포 (히스토그램)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('종업원수 (명)', fontsize=12)
axes[1, 0].set_ylabel('빈도', fontsize=12)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. 지역별 기업 수
region_counts = df['지역'].value_counts()
axes[1, 1].bar(region_counts.index, region_counts.values, color='lightgreen', alpha=0.7)
axes[1, 1].set_title('지역별 기업 수', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('지역', fontsize=12)
axes[1, 1].set_ylabel('기업 수', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/data_distribution.png', dpi=100, bbox_inches='tight')
print("\n📊 시각화 저장 완료: ../output/data_distribution.png")
plt.show()
```

### 실습 7: 이상치 탐지

```python
print("\n" + "=" * 80)
print("📌 실습 7: 이상치 탐지")
print("=" * 80)

def detect_outliers_iqr(series, multiplier=1.5):
    """IQR 방법으로 이상치 탐지"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound

# 매출액 이상치 탐지
print("\n[1] 매출액 이상치 탐지")
print("-" * 80)
outliers_sales, lower_sales, upper_sales = detect_outliers_iqr(df['매출액'].dropna())
print(f"정상 범위: {lower_sales:,.0f} ~ {upper_sales:,.0f}")
print(f"이상치 개수: {len(outliers_sales)} 개 ({len(outliers_sales)/len(df)*100:.2f}%)")
if len(outliers_sales) > 0:
    print(f"\n이상치 샘플 (상위 5개):")
    print(outliers_sales.sort_values(ascending=False).head())

# 종업원수 이상치 탐지
print("\n[2] 종업원수 이상치 탐지")
print("-" * 80)
outliers_emp, lower_emp, upper_emp = detect_outliers_iqr(df['종업원수'].dropna())
print(f"정상 범위: {lower_emp:,.0f} ~ {upper_emp:,.0f}")
print(f"이상치 개수: {len(outliers_emp)} 개 ({len(outliers_emp)/len(df)*100:.2f}%)")
if len(outliers_emp) > 0:
    print(f"\n이상치 샘플 (상위 5개):")
    print(outliers_emp.sort_values(ascending=False).head())

print("\n💡 이상치 해석:")
print("- 이상치가 반드시 오류는 아닙니다")
print("- 대기업이나 특수한 케이스일 수 있습니다")
print("- 도메인 지식을 바탕으로 판단 필요")
```

### 실습 8: 중복 데이터 확인

```python
print("\n" + "=" * 80)
print("📌 실습 8: 중복 데이터 확인")
print("=" * 80)

# 완전히 동일한 행 확인
duplicates = df.duplicated()
duplicate_count = duplicates.sum()

print(f"\n완전 중복 행 개수: {duplicate_count}")

if duplicate_count > 0:
    print("\n중복된 행 샘플:")
    print(df[duplicates].head())
    
    # 중복 제거 후 크기
    df_no_dup = df.drop_duplicates()
    print(f"\n중복 제거 전 크기: {df.shape}")
    print(f"중복 제거 후 크기: {df_no_dup.shape}")
else:
    print("✅ 완전 중복 행이 없습니다.")

# 특정 컬럼 기준 중복 확인
print("\n일련번호 기준 중복 확인:")
dup_ids = df['일련번호'].duplicated()
print(f"중복된 일련번호: {dup_ids.sum()}개")
```

### 실습 9: 데이터 품질 리포트 생성

```python
print("\n" + "=" * 80)
print("📌 실습 9: 데이터 품질 종합 리포트")
print("=" * 80)

quality_report = {
    '항목': [],
    '값': [],
    '평가': []
}

# 1. 데이터 크기
quality_report['항목'].append('총 행 수')
quality_report['값'].append(f"{len(df):,}")
quality_report['평가'].append('✅' if len(df) >= 100 else '⚠️')

# 2. 총 컬럼 수
quality_report['항목'].append('총 컬럼 수')
quality_report['값'].append(f"{len(df.columns)}")
quality_report['평가'].append('✅')

# 3. 결측치 비율
total_missing_ratio = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
quality_report['항목'].append('전체 결측치 비율')
quality_report['값'].append(f"{total_missing_ratio:.2f}%")
quality_report['평가'].append('✅' if total_missing_ratio < 20 else '⚠️')

# 4. 중복 행
quality_report['항목'].append('중복 행 수')
quality_report['값'].append(f"{duplicate_count}")
quality_report['평가'].append('✅' if duplicate_count == 0 else '⚠️')

# 5. 메모리 사용량
quality_report['항목'].append('메모리 사용량')
quality_report['값'].append(f"{memory_usage:.2f} MB")
quality_report['평가'].append('✅' if memory_usage < 100 else '⚠️')

# 리포트 출력
report_df = pd.DataFrame(quality_report)
print("\n" + "=" * 80)
print("📋 데이터 품질 리포트")
print("=" * 80)
print(report_df.to_string(index=False))

# 종합 평가
warnings_count = (report_df['평가'] == '⚠️').sum()
if warnings_count == 0:
    print("\n✅ 전체 평가: 우수 - 모든 품질 기준 통과")
elif warnings_count <= 2:
    print(f"\n⚠️  전체 평가: 양호 - {warnings_count}개 항목 주의 필요")
else:
    print(f"\n❌ 전체 평가: 개선 필요 - {warnings_count}개 항목 문제 발견")
```

### 실습 10: 탐색적 데이터 분석 요약

```python
print("\n" + "=" * 80)
print("📌 실습 10: 탐색적 데이터 분석 (EDA) 요약")
print("=" * 80)

print("\n📊 주요 발견사항:")
print("-" * 80)

# 1. 데이터 규모
print(f"1. 데이터 규모: {len(df):,}개 기업, {len(df.columns)}개 변수")

# 2. 지역 분포
top_region = df['지역'].value_counts().idxmax()
top_region_pct = (df['지역'].value_counts().iloc[0] / len(df) * 100)
print(f"2. 지역 분포: {top_region}이 {top_region_pct:.1f}%로 가장 많음")

# 3. 업종 분포
top_industry = df['업종'].value_counts().idxmax()
top_industry_pct = (df['업종'].value_counts().iloc[0] / len(df) * 100)
print(f"3. 업종 분포: {top_industry}이 {top_industry_pct:.1f}%로 가장 많음")

# 4. 매출액 통계
print(f"4. 매출액: 평균 {df['매출액'].mean():,.0f}원, 중앙값 {df['매출액'].median():,.0f}원")

# 5. 종업원수 통계
print(f"5. 종업원수: 평균 {df['종업원수'].mean():.1f}명, 중앙값 {df['종업원수'].median():.0f}명")

# 6. 결측치 요약
print(f"6. 결측치: 전체 {total_missing_ratio:.2f}%, 최대 {missing_ratio.max():.2f}% ({missing_ratio.idxmax()})")

print("\n💡 다음 단계 권장사항:")
print("-" * 80)
print("1. 결측치가 10% 이상인 컬럼은 처리 방안 검토 필요")
print("2. 이상치는 도메인 전문가와 상의하여 처리 방법 결정")
print("3. 중복 데이터가 있다면 제거 또는 통합 필요")
print("4. 다음 교시에서 필터링 및 그룹 분석 수행 예정")

# 분석 종료 시간
print("\n" + "=" * 80)
print(f"분석 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
```

---

## 🧩 퀴즈

### 문제 1
데이터 분석 프로세스에서 데이터 이해 및 탐색 단계가 차지하는 시간은?

<details>
<summary>정답 보기</summary>

**정답: 전체 시간의 60~80%**

**해설:**
실무 데이터 분석에서 가장 많은 시간을 소비하는 단계입니다.
- 데이터 수집 및 이해: 20-30%
- 데이터 정제 및 가공: 40-50%
- 모델링 및 분석: 10-20%
- 결과 해석 및 보고: 10-20%

💡 "쓰레기가 들어가면 쓰레기가 나온다" (Garbage In, Garbage Out)
품질 좋은 데이터가 좋은 분석 결과를 만듭니다!
</details>

---

### 문제 2
IQR 방법으로 이상치를 탐지할 때 기준은?

<details>
<summary>정답 보기</summary>

**정답:**
- 하한: Q1 - 1.5 × IQR
- 상한: Q3 + 1.5 × IQR

**해설:**
```python
Q1 = 25% 백분위수
Q3 = 75% 백분위수
IQR = Q3 - Q1 (사분위 범위)

이상치 = 값 < (Q1 - 1.5×IQR) or 값 > (Q3 + 1.5×IQR)
```

**예시:**
데이터: [10, 12, 13, 14, 15, 16, 17, 100]
- Q1 = 12.5
- Q3 = 16.5
- IQR = 4
- 하한 = 12.5 - 1.5×4 = 6.5
- 상한 = 16.5 + 1.5×4 = 22.5
- 이상치: 100 (상한 초과)

</details>

---

### 문제 3
다음 중 데이터 품질 평가 기준이 아닌 것은?

1. 완전성 (Completeness)
2. 정확성 (Accuracy)
3. 복잡성 (Complexity)
4. 일관성 (Consistency)

<details>
<summary>정답 보기</summary>

**정답: 3번 복잡성 (Complexity)**

**해설:**
데이터 품질 4대 기준:
1. **완전성**: 결측치가 적은가?
2. **정확성**: 값이 올바른가?
3. **일관성**: 형식이 통일되어 있는가?
4. **최신성**: 데이터가 최신인가?

복잡성은 데이터 품질 기준이 아니라 분석의 난이도를 나타냅니다.
</details>

---

## ✅ 5교시 학습 완료 체크리스트

- [ ] 데이터 불러오기 및 기본 정보 확인
- [ ] 데이터 타입과 구조 파악
- [ ] 결측치 현황 분석 및 시각화
- [ ] 기초 통계량 계산
- [ ] 범주형 데이터 분포 확인
- [ ] 이상치 탐지 (IQR 방법)
- [ ] 중복 데이터 확인
- [ ] 데이터 품질 리포트 생성
- [ ] 퀴즈 3문제 모두 풀이 완료

---

**이전 학습:** [4교시 - apply 함수](./4_apply_function.md)  
**다음 학습:** [6교시 - 종합 실습 (2) 필터링 및 그룹핑](./6_filter_group.md)

**학습 완료일:** _____________  
**소요 시간:** _____________  
**이해도 (1~5):** ⭐⭐⭐⭐⭐
