# 5교시: Heatmap - 변수 간 상관관계 인사이트 도출

**학습 목표:**
- 상관행렬(Correlation Matrix) 생성 및 해석
- Heatmap으로 다차원 관계 시각화
- 업종/지역/시간대별 패턴 발견
- 실무 의사결정을 위한 상관관계 활용

**예상 소요 시간:** 60분

---

## 1. Heatmap과 상관관계 소개

### 1.1 Heatmap이란?

**Heatmap(히트맵)**은 데이터를 색상으로 표현하는 시각화 기법입니다.
- **용도**: 많은 데이터를 한눈에 파악, 패턴 발견
- **색상**: 값의 크기를 색상의 진하기로 표현
- **실무 활용**: 변수 간 관계, 시간대별 패턴, 지역별 차이 등

**비유:** 날씨 지도에서 온도를 색상으로 표시하는 것과 같습니다.

### 1.2 상관계수(Correlation Coefficient)

**상관계수**는 두 변수 간 선형 관계의 강도와 방향을 나타내는 지표입니다.

- **범위**: -1 ~ +1
- **+1**: 완벽한 양의 상관 (한 변수 ↑ → 다른 변수 ↑)
- **0**: 상관 없음
- **-1**: 완벽한 음의 상관 (한 변수 ↑ → 다른 변수 ↓)

**해석 기준:**
- |r| ≥ 0.7: 강한 상관
- 0.4 ≤ |r| < 0.7: 중간 상관
- |r| < 0.4: 약한 상관

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("data/서울시_상권_추정매출.csv", encoding="cp949")
```

---

## 2. 기본 Heatmap: 요일별 매출 상관관계

### 2.1 요일별 매출 데이터 준비

```python
# 요일별 매출 컬럼 선택
weekday_columns = ['월요일매출금액', '화요일매출금액', '수요일매출금액', 
                   '목요일매출금액', '금요일매출금액', '토요일매출금액', '일요일매출금액']

# 상관행렬 계산
weekday_corr = df[weekday_columns].corr()

print("=" * 60)
print("요일별 매출 상관행렬")
print("=" * 60)
print(weekday_corr.round(3))
```

### 2.2 Heatmap 시각화

```python
# 기본 Heatmap
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(weekday_corr,
            annot=True,  # 숫자 표시
            fmt='.2f',  # 소수점 2자리
            cmap='coolwarm',  # 색상 맵 (파란색=낮음, 빨간색=높음)
            center=0,  # 중심값
            square=True,  # 정사각형 셀
            linewidths=1,  # 셀 경계선
            cbar_kws={'label': '상관계수'},
            ax=ax)

# 그래프 꾸미기
ax.set_title('요일별 매출 상관관계 Heatmap', fontsize=16, fontweight='bold', pad=20)

# 축 레이블 간소화
ax.set_xticklabels(['월', '화', '수', '목', '금', '토', '일'], rotation=0)
ax.set_yticklabels(['월', '화', '수', '목', '금', '토', '일'], rotation=0)

plt.tight_layout()
plt.savefig('output/5_weekday_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2.3 결과 해석

**실무 인사이트:**
- **대각선 = 1.0**: 자기 자신과의 상관 (당연히 완벽한 상관)
- **평일 간 높은 상관**: 월~금 서로 비슷한 패턴
- **주말 간 높은 상관**: 토~일 서로 비슷한 패턴
- **평일 vs 주말 상관 낮음**: 소비 패턴이 다름

**비즈니스 활용:**
- 평일과 주말의 운영 전략을 다르게 가져가야 함
- 인력 배치, 재고 관리를 요일별로 차별화

---

## 3. 연령대별 매출 상관관계

### 3.1 연령대별 매출 Heatmap

```python
# 연령대별 매출 컬럼
age_columns = ['10대매출금액', '20대매출금액', '30대매출금액', 
               '40대매출금액', '50대매출금액', '60대이상매출금액']

# 상관행렬
age_corr = df[age_columns].corr()

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# 마스크 생성 (상삼각 부분 숨기기 - 중복 제거)
mask = np.triu(np.ones_like(age_corr, dtype=bool))

sns.heatmap(age_corr,
            mask=mask,  # 상삼각 부분 숨김
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',  # 빨강-노랑-초록
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': '상관계수'},
            ax=ax)

# 그래프 꾸미기
ax.set_title('연령대별 매출 상관관계 Heatmap\n(하삼각만 표시)', 
             fontsize=16, fontweight='bold', pad=20)

# 축 레이블
age_labels = ['10대', '20대', '30대', '40대', '50대', '60대+']
ax.set_xticklabels(age_labels, rotation=45)
ax.set_yticklabels(age_labels, rotation=0)

plt.tight_layout()
plt.savefig('output/5_age_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 가장 상관이 높은 연령대 쌍 찾기
print("\n" + "=" * 60)
print("연령대별 매출 상관관계 분석")
print("=" * 60)

# 대각선과 상삼각 제외
corr_pairs = []
for i in range(len(age_columns)):
    for j in range(i+1, len(age_columns)):
        corr_pairs.append({
            '연령대1': age_labels[i],
            '연령대2': age_labels[j],
            '상관계수': age_corr.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs).sort_values('상관계수', ascending=False)
print("\n상관관계 상위 5개:")
print(corr_df.head())
```

### 3.2 결과 해석

**실무 인사이트:**
- **인접 연령대 높은 상관**: 20대와 30대, 40대와 50대 등
- **멀리 떨어진 연령대 낮은 상관**: 10대와 60대 등
- **비즈니스 활용**:
  - 주 고객층이 20대라면 30대도 함께 타겟팅
  - 마케팅 메시지를 비슷한 연령대로 묶어서 설계

---

## 4. 업종별 상관관계 (Pivot Table + Heatmap)

### 4.1 업종별 평균 매출 Pivot Table

```python
# 주요 업종과 지역 선택
major_industries = ['커피/음료', '한식음식점', '편의점', '치킨전문점', '의류']
major_districts = ['강남구', '마포구', '종로구', '송파구', '영등포구']

# 필터링
df_major = df[(df['서비스업종코드명'].isin(major_industries)) & 
              (df['행정구역명'].isin(major_districts))]

# Pivot Table: 행=업종, 열=지역, 값=평균 월매출
pivot_industry_district = df_major.pivot_table(
    values='월매출금액',
    index='서비스업종코드명',
    columns='행정구역명',
    aggfunc='mean'
)

# Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(pivot_industry_district,
            annot=True,
            fmt='.0f',  # 정수로 표시
            cmap='YlOrRd',  # 노랑-주황-빨강
            linewidths=1,
            cbar_kws={'label': '평균 월매출금액 (원)'},
            ax=ax)

# 그래프 꾸미기
ax.set_title('업종별 × 지역별 평균 월매출 Heatmap', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('행정구역', fontsize=12)
ax.set_ylabel('업종', fontsize=12)

plt.tight_layout()
plt.savefig('output/5_industry_district_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 각 업종별 최고 매출 지역
print("\n" + "=" * 60)
print("업종별 최고 매출 지역")
print("=" * 60)
for industry in major_industries:
    max_district = pivot_industry_district.loc[industry].idxmax()
    max_value = pivot_industry_district.loc[industry].max()
    print(f"{industry}: {max_district} ({max_value/1000000:.2f}M원)")
```

### 4.2 결과 해석

**실무 인사이트:**
- **업종별 핫스팟**: 각 업종이 가장 잘 되는 지역 파악
- **지역별 강점**: 특정 지역이 강한 업종 확인
- **입점 전략**:
  - 커피/음료가 강남구에서 강하다면 → 강남구 입점 우선 고려
  - 마포구에서 약한 업종 찾기 → 블루오션 기회

---

## 5. 시간대별 매출 패턴 Heatmap

### 5.1 요일 × 업종 Heatmap

```python
# 업종별 요일 평균 매출 계산
industry_weekday_data = []

for industry in major_industries:
    industry_df = df[df['서비스업종코드명'] == industry]
    for weekday_col in weekday_columns:
        avg_sales = industry_df[weekday_col].mean()
        weekday_name = weekday_col.replace('매출금액', '').replace('요일', '')
        
        industry_weekday_data.append({
            '업종': industry,
            '요일': weekday_name,
            '평균매출': avg_sales
        })

industry_weekday_df = pd.DataFrame(industry_weekday_data)

# Pivot Table
pivot_industry_weekday = industry_weekday_df.pivot_table(
    values='평균매출',
    index='업종',
    columns='요일',
    aggfunc='mean'
)

# 요일 순서 정렬
weekday_order = ['월', '화', '수', '목', '금', '토', '일']
pivot_industry_weekday = pivot_industry_weekday[weekday_order]

# Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(pivot_industry_weekday,
            annot=True,
            fmt='.0f',
            cmap='viridis',
            linewidths=1,
            cbar_kws={'label': '평균 매출금액 (원)'},
            ax=ax)

# 그래프 꾸미기
ax.set_title('업종별 × 요일별 평균 매출 패턴 Heatmap', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('요일', fontsize=12)
ax.set_ylabel('업종', fontsize=12)

plt.tight_layout()
plt.savefig('output/5_industry_weekday_pattern.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.2 결과 해석

**실무 인사이트:**
- **주말 강세 업종**: 치킨, 주점 등
- **평일 강세 업종**: 편의점, 커피 등
- **요일 무관 업종**: 한식음식점 등

**운영 전략:**
- 주말 강세 업종: 금/토/일 집중 프로모션
- 평일 강세 업종: 월~금 점심 특가 등

---

## 6. 복합 변수 상관관계

### 6.1 매출/점포수/연령대 통합 분석

```python
# 숫자형 변수만 선택
numeric_columns = ['월매출금액', '점포수'] + weekday_columns + age_columns

# 상관행렬
full_corr = df[numeric_columns].corr()

# Heatmap (큰 행렬)
fig, ax = plt.subplots(figsize=(16, 14))

sns.heatmap(full_corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            cbar_kws={'label': '상관계수'},
            ax=ax,
            annot_kws={'size': 8})  # 글자 크기 조정

# 그래프 꾸미기
ax.set_title('전체 변수 상관관계 Heatmap', fontsize=16, fontweight='bold', pad=20)

# 축 레이블 회전
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('output/5_full_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 월매출금액과 가장 상관이 높은 변수 찾기
monthly_corr = full_corr['월매출금액'].drop('월매출금액').sort_values(ascending=False)

print("\n" + "=" * 60)
print("월매출금액과 상관관계가 높은 변수 Top 5")
print("=" * 60)
for idx, (var, corr) in enumerate(monthly_corr.head().items(), 1):
    print(f"{idx}. {var}: {corr:.3f}")
```

### 6.2 결과 해석

**실무 인사이트:**
- 요일별 매출 간 높은 상관: 일관된 매출 패턴
- 특정 연령대와 매출의 상관: 주 고객층 파악
- 점포수와 매출의 관계: 경쟁 또는 집적 효과

---

## 7. 클러스터맵 (Clustermap): 유사한 패턴 그룹화

### 7.1 업종별 연령대 매출 패턴 클러스터링

```python
# 업종별 연령대 평균 매출
industry_age_data = []

for industry in df['서비스업종코드명'].unique()[:10]:  # 상위 10개 업종
    industry_df = df[df['서비스업종코드명'] == industry]
    row_data = {'업종': industry}
    
    for age_col in age_columns:
        age_label = age_col.replace('매출금액', '')
        row_data[age_label] = industry_df[age_col].mean()
    
    industry_age_data.append(row_data)

industry_age_df = pd.DataFrame(industry_age_data).set_index('업종')

# Clustermap: 유사한 패턴끼리 자동 그룹화
g = sns.clustermap(industry_age_df,
                   cmap='YlGnBu',
                   figsize=(12, 10),
                   linewidths=1,
                   annot=True,
                   fmt='.0f',
                   cbar_kws={'label': '평균 매출금액 (원)'})

g.fig.suptitle('업종별 연령대 매출 패턴 클러스터맵\n(유사 패턴 자동 그룹화)', 
               fontsize=16, fontweight='bold', y=0.98)

plt.savefig('output/5_industry_age_clustermap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n** Clustermap 해석 **")
print("비슷한 연령대 선호 패턴을 가진 업종들이 가까이 배치됩니다.")
print("이를 통해 업종 간 유사성을 파악하고 마케팅 전략을 공유할 수 있습니다.")
```

### 7.2 결과 해석

**실무 인사이트:**
- **유사 패턴 업종**: 고객층이 비슷 → 유사 마케팅 전략 적용 가능
- **차별화 업종**: 독특한 고객층 → 맞춤형 전략 필요
- **크로스 프로모션**: 유사 패턴 업종끼리 제휴 마케팅

---

## 8. 실무 적용 예시

### 시나리오: 신규 업종 선택을 위한 상관관계 분석

```python
# 목표: 커피/음료와 가장 독립적인 업종 찾기 (상관이 낮은 업종)
# → 포트폴리오 다각화 전략

# 업종별 월매출 평균 계산
industry_avg_sales = df.groupby('서비스업종코드명')['월매출금액'].mean()

# 상위 12개 업종 선택
top_industries = industry_avg_sales.nlargest(12).index

# 업종 간 상관행렬 (상권별 평균 매출 기준)
industry_pivot = df[df['서비스업종코드명'].isin(top_industries)].pivot_table(
    values='월매출금액',
    index='상권코드명',
    columns='서비스업종코드명',
    aggfunc='mean'
)

industry_corr = industry_pivot.corr()

# Heatmap
fig, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(industry_corr,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',  # 빨강(음의 상관) - 파랑(양의 상관)
            center=0,
            linewidths=1,
            cbar_kws={'label': '상관계수'},
            ax=ax)

ax.set_title('업종 간 매출 상관관계 분석\n(포트폴리오 다각화용)', 
             fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('output/5_industry_portfolio_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 커피/음료와 상관이 낮은 업종 찾기
if '커피/음료' in industry_corr.columns:
    coffee_corr = industry_corr['커피/음료'].drop('커피/음료').sort_values()
    
    print("\n" + "=" * 70)
    print("포트폴리오 다각화 전략: 커피/음료와 독립적인 업종")
    print("=" * 70)
    print("\n상관계수가 낮은 업종 Top 5 (리스크 분산에 유리):")
    for idx, (industry, corr) in enumerate(coffee_corr.head().items(), 1):
        print(f"{idx}. {industry}: 상관계수 {corr:.3f}")
    
    print("\n→ 이들 업종은 커피/음료와 시장 상황이 독립적이므로")
    print("   포트폴리오 다각화 시 리스크를 분산할 수 있습니다.")
```

---

## 9. 학습 정리

### 9.1 핵심 개념

1. **Heatmap**: 데이터를 색상으로 시각화 (많은 정보를 한눈에)
2. **상관행렬**: 여러 변수 간 관계를 한 번에 파악
3. **Pivot Table + Heatmap**: 다차원 데이터 분석
4. **Clustermap**: 유사 패턴 자동 그룹화
5. **실무 활용**: 타겟팅, 운영 전략, 포트폴리오 다각화

### 9.2 실무 활용 팁

- **색상 선택**: 
  - 단일 그라데이션(YlOrRd): 크기 강조
  - 양방향(RdBu): 양/음 상관 구분
  - 접근성: colorblind-friendly 팔레트 사용
  
- **Annotation**: 정확한 수치를 함께 표시
- **Mask**: 중복 정보 제거 (상삼각 숨김)
- **크기 조절**: 변수가 많으면 figsize 크게

---

## 10. 학습 확인 퀴즈

### 문제 1
상관계수가 +0.85라면, 두 변수의 관계는?
1. 강한 음의 상관
2. 약한 양의 상관
3. 강한 양의 상관
4. 상관 없음

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 3. 강한 양의 상관**

**해설:** 
상관계수가 +0.85는 강한 양의 상관관계를 나타냅니다.
- |r| ≥ 0.7: 강한 상관
- 양수(+): 한 변수가 증가하면 다른 변수도 증가
예: 광고비와 매출, 기온과 아이스크림 판매량
</details>

---

### 문제 2
Heatmap에서 상삼각 부분을 숨기는 이유는?
1. 그래프를 예쁘게 만들기 위해
2. 대칭 행렬이므로 중복 정보를 제거하기 위해
3. 계산 오류를 숨기기 위해
4. 음수 값을 숨기기 위해

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 대칭 행렬이므로 중복 정보를 제거하기 위해**

**해설:** 
상관행렬은 대칭 행렬입니다 (A와 B의 상관 = B와 A의 상관). 따라서 상삼각 또는 하삼각 중 하나만 표시해도 모든 정보를 전달할 수 있습니다. 이렇게 하면 그래프가 깔끔해지고 가독성이 향상됩니다.
</details>

---

### 문제 3
Clustermap의 주요 기능은?
1. 데이터를 무작위로 배치
2. 유사한 패턴을 가진 행/열을 자동으로 그룹화
3. 이상치를 제거
4. 평균을 계산

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 유사한 패턴을 가진 행/열을 자동으로 그룹화**

**해설:** 
Clustermap은 계층적 클러스터링을 사용하여 유사한 패턴을 가진 행과 열을 자동으로 그룹화합니다. 이를 통해 데이터의 숨겨진 구조를 발견하고, 비슷한 특성을 가진 그룹을 한눈에 파악할 수 있습니다.
예: 비슷한 고객층을 타겟하는 업종끼리 자동으로 그룹화
</details>

---

### 문제 4
업종별 × 지역별 평균 매출 Heatmap을 만들 때 사용하는 Pandas 함수는?
1. groupby()
2. merge()
3. pivot_table()
4. concat()

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 3. pivot_table()**

**해설:** 
`pivot_table()`은 데이터를 재구조화하여 행과 열에 카테고리를 배치하고, 교차 지점에 집계 값을 계산합니다.
```python
pivot_table(values='월매출금액', 
            index='서비스업종코드명',  # 행
            columns='행정구역명',  # 열
            aggfunc='mean')  # 평균
```
이 결과를 바로 Heatmap으로 시각화할 수 있습니다.
</details>

---

### 문제 5
포트폴리오 다각화를 위해 상관관계를 어떻게 활용하는가?
1. 상관관계가 높은 업종끼리 묶는다
2. 상관관계가 낮은(독립적인) 업종을 선택하여 리스크를 분산한다
3. 상관관계를 무시한다
4. 상관관계가 음수인 업종만 선택한다

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 상관관계가 낮은(독립적인) 업종을 선택하여 리스크를 분산한다**

**해설:** 
포트폴리오 다각화의 핵심은 상관관계가 낮은 자산을 조합하는 것입니다.
- 상관이 낮으면: 한 업종이 부진해도 다른 업종은 독립적으로 성과를 낼 수 있음
- 상관이 높으면: 모든 업종이 동시에 부진할 위험

예: 커피/음료(평일 강세)와 주점(주말 강세)은 상관이 낮아 리스크 분산에 유리
</details>

---

## 다음 교시 예고

**6교시: Folium - 지도 위 매장/상권 위치 시각화**
- Folium 라이브러리 소개
- 지도에 마커 추가
- 마커 클러스터링
- 히트맵으로 밀집도 표현
- 실무: 최적 입점 위치 찾기
