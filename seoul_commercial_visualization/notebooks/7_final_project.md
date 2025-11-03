# 7교시: 종합 프로젝트 - 서울시 상권분석 프로젝트

**학습 목표:**
- 실무 데이터 분석 프로젝트의 전체 과정 경험
- 가설 수립부터 발표까지 end-to-end 수행
- 학습한 모든 시각화 기법 통합 활용
- 비즈니스 인사이트 도출 및 의사결정 지원

**예상 소요 시간:** 6~7시간 (1~7교시)

---

## 프로젝트 개요

### 전체 일정 (11/8 토요일)

| 교시 | 시간 | 내용 |
|-----|------|------|
| 1교시 | 09:00-10:00 | 프로젝트 소개, 주제 선정, 가설 수립 |
| 2교시 | 10:10-11:10 | 데이터 불러오기 및 전처리 |
| 3교시 | 11:20-12:20 | 탐색적 데이터 분석 (EDA) |
| 점심 | 12:20-13:20 | 휴식 |
| 4교시 | 13:20-14:20 | 핵심 시각화 제작 |
| 5교시 | 14:30-15:30 | 인사이트 도출 및 보고서 작성 |
| 6교시 | 15:40-16:40 | 발표 자료 준비 |
| 7교시 | 16:50-17:50 | 발표 및 피드백 |

---

## 1교시: 프로젝트 소개 및 가설 수립

### 1.1 프로젝트 주제 예시

다음 중 하나를 선택하거나 자신만의 주제를 설정하세요:

**주제 1: 신규 카페 입점 지역 분석**
- **목표**: 서울시에서 카페를 새로 열기에 가장 좋은 지역 찾기
- **분석 관점**: 경쟁 현황, 매출 잠재력, 고객 연령층

**주제 2: 업종별 성공 요인 분석**
- **목표**: 높은 매출을 올리는 업종의 공통점 파악
- **분석 관점**: 지역 특성, 요일 패턴, 점포 밀집도

**주제 3: 지역별 상권 특성 비교**
- **목표**: 강남구 vs 마포구 vs 종로구의 상권 특징 비교
- **분석 관점**: 업종 구성, 매출 수준, 연령대 선호도

**주제 4: 요일별 매출 패턴 분석**
- **목표**: 업종별 최적 운영 시간대 및 마케팅 전략 수립
- **분석 관점**: 평일 vs 주말, 업종별 차이

### 1.2 가설 수립 프레임워크

```python
"""
프로젝트 기획서 템플릿
"""

# 프로젝트 정보
PROJECT_INFO = {
    "주제": "신규 카페 최적 입점 지역 분석",
    "분석 기간": "2023년 전체",
    "분석 대상": "서울시 25개 구",
    "데이터": "서울시 상권분석 데이터"
}

# 핵심 질문 (Research Questions)
RESEARCH_QUESTIONS = [
    "1. 어느 지역이 카페 매출이 가장 높은가?",
    "2. 카페가 성공하기 위한 지역적 특성은 무엇인가?",
    "3. 경쟁이 치열한 곳과 블루오션은 어디인가?",
    "4. 주 고객층은 어느 연령대인가?"
]

# 가설 (Hypotheses)
HYPOTHESES = [
    "H1: 유동인구가 많은 강남구, 종로구에서 카페 매출이 높을 것이다.",
    "H2: 카페의 주 고객층은 20-30대이며, 젊은 층이 많은 지역의 매출이 높을 것이다.",
    "H3: 점포수가 많은 곳은 경쟁이 심하지만, 그만큼 수요도 높아 매출도 높을 것이다.",
    "H4: 카페는 평일과 주말의 매출 차이가 크지 않을 것이다. (일상적 소비)"
]

# 분석 계획
ANALYSIS_PLAN = {
    "1단계": "데이터 전처리 - 결측치 확인, 이상치 처리",
    "2단계": "EDA - 기술통계, 분포 확인",
    "3단계": "시각화 - 지역별/업종별 비교",
    "4단계": "인사이트 도출 - 가설 검증",
    "5단계": "결론 및 제언 - 의사결정 지원"
}

print("=" * 70)
print("프로젝트 기획서")
print("=" * 70)
print(f"\n주제: {PROJECT_INFO['주제']}")
print(f"\n핵심 질문:")
for q in RESEARCH_QUESTIONS:
    print(f"  {q}")
print(f"\n가설:")
for h in HYPOTHESES:
    print(f"  {h}")
```

### 1.3 실습: 자신의 프로젝트 기획서 작성

```python
# TODO: 여러분의 프로젝트 정보를 입력하세요

MY_PROJECT = {
    "주제": "",  # 예: "주점 업종 최적 입점 지역 분석"
    "핵심_질문": [
        # 예: "1. 주말 매출이 가장 높은 지역은?"
    ],
    "가설": [
        # 예: "H1: 대학가 주변에서 주점 매출이 높을 것이다."
    ]
}

# 팀원들과 공유하고 피드백 받기
```

---

## 2교시: 데이터 불러오기 및 전처리

### 2.1 데이터 불러오기

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("data/서울시_상권_추정매출.csv", encoding="cp949")

print("=" * 70)
print("데이터 불러오기 완료")
print("=" * 70)
print(f"총 레코드 수: {len(df):,}개")
print(f"컬럼 수: {len(df.columns)}개")
print(f"\n데이터 미리보기:")
print(df.head())
```

### 2.2 데이터 품질 확인

```python
# 1. 결측치 확인
print("\n" + "=" * 70)
print("1. 결측치 확인")
print("=" * 70)
missing = df.isnull().sum()
missing_percent = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '결측치 개수': missing,
    '비율(%)': missing_percent
})
print(missing_df[missing_df['결측치 개수'] > 0])

if missing.sum() == 0:
    print("✓ 결측치 없음")

# 2. 중복 확인
print("\n" + "=" * 70)
print("2. 중복 레코드 확인")
print("=" * 70)
duplicates = df.duplicated().sum()
print(f"중복 레코드: {duplicates}개")

# 3. 데이터 타입 확인
print("\n" + "=" * 70)
print("3. 데이터 타입")
print("=" * 70)
print(df.dtypes)

# 4. 기술통계
print("\n" + "=" * 70)
print("4. 주요 변수 기술통계")
print("=" * 70)
print(df[['월매출금액', '점포수']].describe())
```

### 2.3 데이터 전처리

```python
# 프로젝트에 필요한 데이터만 추출
# 예: 커피/음료 업종만 분석하는 경우

# 업종 선택 (프로젝트 주제에 맞게 수정)
target_industry = '커피/음료'
df_project = df[df['서비스업종코드명'] == target_industry].copy()

print(f"\n{target_industry} 업종 데이터: {len(df_project)}개")

# 이상치 처리 (선택 사항)
# 예: 월매출이 극단적으로 낮거나 높은 경우
Q1 = df_project['월매출금액'].quantile(0.25)
Q3 = df_project['월매출금액'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_project[(df_project['월매출금액'] < lower_bound) | 
                      (df_project['월매출금액'] > upper_bound)]

print(f"\n이상치: {len(outliers)}개 ({len(outliers)/len(df_project)*100:.1f}%)")
print("이상치 처리 방법 선택:")
print("  1) 제거: 분석에서 제외")
print("  2) 유지: 특별한 케이스로 별도 분석")
print("  → 프로젝트 목적에 따라 결정")

# 새로운 변수 생성 (Feature Engineering)
df_project['총요일매출'] = df_project[['월요일매출금액', '화요일매출금액', 
                                      '수요일매출금액', '목요일매출금액', 
                                      '금요일매출금액']].sum(axis=1)

df_project['총주말매출'] = df_project[['토요일매출금액', '일요일매출금액']].sum(axis=1)

df_project['평일주말비율'] = df_project['총요일매출'] / df_project['총주말매출']

print("\n✓ 새로운 변수 생성 완료")
print("  - 총요일매출")
print("  - 총주말매출")
print("  - 평일주말비율")
```

---

## 3교시: 탐색적 데이터 분석 (EDA)

### 3.1 전체적인 분포 파악

```python
# 1. 지역별 매장 수
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1-1. 구별 매장 수
district_counts = df_project['행정구역명'].value_counts().head(10)
axes[0, 0].barh(range(len(district_counts)), district_counts.values, color='skyblue')
axes[0, 0].set_yticks(range(len(district_counts)))
axes[0, 0].set_yticklabels(district_counts.index)
axes[0, 0].set_title(f'{target_industry} - 구별 매장 수 Top 10', fontweight='bold')
axes[0, 0].set_xlabel('매장 수')
axes[0, 0].grid(axis='x', alpha=0.3)

# 1-2. 구별 평균 매출
district_sales = df_project.groupby('행정구역명')['월매출금액'].mean().sort_values(ascending=False).head(10)
axes[0, 1].barh(range(len(district_sales)), district_sales.values, color='coral')
axes[0, 1].set_yticks(range(len(district_sales)))
axes[0, 1].set_yticklabels(district_sales.index)
axes[0, 1].set_title(f'{target_industry} - 구별 평균 월매출 Top 10', fontweight='bold')
axes[0, 1].set_xlabel('평균 월매출금액 (원)')
axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
axes[0, 1].grid(axis='x', alpha=0.3)

# 1-3. 매출 분포 히스토그램
axes[1, 0].hist(df_project['월매출금액'], bins=30, color='lightgreen', edgecolor='black')
axes[1, 0].axvline(df_project['월매출금액'].mean(), color='red', linestyle='--', 
                   label=f'평균: {df_project["월매출금액"].mean()/1000000:.1f}M')
axes[1, 0].axvline(df_project['월매출금액'].median(), color='blue', linestyle='--',
                   label=f'중앙값: {df_project["월매출금액"].median()/1000000:.1f}M')
axes[1, 0].set_title(f'{target_industry} - 월매출 분포', fontweight='bold')
axes[1, 0].set_xlabel('월매출금액 (원)')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 1-4. 평일 vs 주말 매출
weekday_weekend_data = pd.DataFrame({
    '구분': ['평일', '주말'],
    '평균매출': [df_project['총요일매출'].mean() / 5,  # 5일로 나눔
                df_project['총주말매출'].mean() / 2]   # 2일로 나눔
})
axes[1, 1].bar(weekday_weekend_data['구분'], weekday_weekend_data['평균매출'], 
               color=['#457B9D', '#E63946'])
axes[1, 1].set_title(f'{target_industry} - 평일 vs 주말 일평균 매출', fontweight='bold')
axes[1, 1].set_ylabel('일평균 매출금액 (원)')
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
axes[1, 1].grid(axis='y', alpha=0.3)

# 값 표시
for i, row in weekday_weekend_data.iterrows():
    axes[1, 1].text(i, row['평균매출'], f"{row['평균매출']/1000000:.1f}M",
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('output/7_eda_overview.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.2 상관관계 분석

```python
# 연령대별 매출 상관관계
age_columns = ['10대매출금액', '20대매출금액', '30대매출금액', 
               '40대매출금액', '50대매출금액', '60대이상매출금액']

age_corr = df_project[age_columns].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(age_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax)
ax.set_title(f'{target_industry} - 연령대별 매출 상관관계', fontweight='bold', pad=20)

# 축 레이블 간소화
age_labels = ['10대', '20대', '30대', '40대', '50대', '60대+']
ax.set_xticklabels(age_labels, rotation=45)
ax.set_yticklabels(age_labels, rotation=0)

plt.tight_layout()
plt.savefig('output/7_age_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# 주요 고객층 파악
age_sales_avg = df_project[age_columns].mean()
main_customer = age_labels[age_sales_avg.argmax()]
print(f"\n주요 고객층: {main_customer} (평균 {age_sales_avg.max()/1000000:.1f}M원)")
```

---

## 4교시: 핵심 시각화 제작

### 4.1 지역별 비교 시각화

```python
# 상위 5개 구 선택
top5_districts = df_project.groupby('행정구역명')['월매출금액'].mean().nlargest(5).index

# Boxplot으로 분포 비교
fig, ax = plt.subplots(figsize=(14, 8))

data_top5 = df_project[df_project['행정구역명'].isin(top5_districts)]

sns.boxplot(data=data_top5, x='행정구역명', y='월매출금액',
            order=top5_districts, palette='Set2', ax=ax)

# 평균값 추가
means = data_top5.groupby('행정구역명')['월매출금액'].mean()
for i, district in enumerate(top5_districts):
    ax.scatter(i, means[district], marker='D', s=150, color='red', zorder=3)

ax.set_title(f'{target_industry} - 지역별 월매출 분포 비교 (Top 5)\n빨간 다이아몬드 = 평균',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('행정구역', fontsize=12)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/7_district_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4.2 연령대별 매출 패턴

```python
# 상위 3개 구의 연령대별 매출 비교
top3_districts = df_project.groupby('행정구역명')['월매출금액'].mean().nlargest(3).index

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, district in enumerate(top3_districts):
    district_data = df_project[df_project['행정구역명'] == district]
    age_sales = [district_data[col].mean() for col in age_columns]
    
    axes[idx].bar(age_labels, age_sales, color='steelblue', alpha=0.7)
    axes[idx].set_title(f'{district}\n연령대별 평균 매출', fontweight='bold')
    axes[idx].set_ylabel('평균 매출금액 (원)' if idx == 0 else '')
    axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
    axes[idx].grid(axis='y', alpha=0.3)
    
    # 최고 매출 연령대 강조
    max_idx = np.argmax(age_sales)
    axes[idx].patches[max_idx].set_color('orangered')

plt.tight_layout()
plt.savefig('output/7_age_pattern_by_district.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4.3 지도 시각화

```python
# Folium 지도로 지역별 매출 시각화
seoul_center = [37.5665, 126.9780]

m = folium.Map(location=seoul_center, zoom_start=11)

# 구별 평균 매출
district_avg_sales = df_project.groupby('행정구역명').agg({
    '월매출금액': 'mean',
    '위도': 'mean',
    '경도': 'mean'
}).reset_index()

# CircleMarker로 표시
for idx, row in district_avg_sales.iterrows():
    # 매출에 비례하는 반경
    radius = (row['월매출금액'] / district_avg_sales['월매출금액'].max()) * 30 + 10
    
    # 색상 (평균 기준)
    color = 'red' if row['월매출금액'] > district_avg_sales['월매출금액'].median() else 'blue'
    
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=radius,
        popup=f"""
        <b>{row['행정구역명']}</b><br>
        평균 월매출: {row['월매출금액']/1000000:.2f}M원
        """,
        tooltip=row['행정구역명'],
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.6
    ).add_to(m)

m.save('output/7_revenue_map.html')
print("\n지역별 매출 지도 생성: output/7_revenue_map.html")
```

---

## 5교시: 인사이트 도출 및 보고서 작성

### 5.1 가설 검증

```python
print("=" * 70)
print("가설 검증 결과")
print("=" * 70)

# H1: 특정 지역에서 매출이 높은가?
top_district = df_project.groupby('행정구역명')['월매출금액'].mean().idxmax()
top_sales = df_project.groupby('행정구역명')['월매출금액'].mean().max()

print(f"\nH1: 유동인구가 많은 강남구, 종로구에서 매출이 높을 것이다.")
print(f"  → 검증: 최고 매출 지역은 {top_district} ({top_sales/1000000:.2f}M원)")
print(f"  → 결론: {'채택' if top_district in ['강남구', '종로구', '중구'] else '부분 채택'}")

# H2: 주 고객층은 20-30대인가?
age_sales_avg = df_project[age_columns].mean()
main_age = age_labels[age_sales_avg.argmax()]

print(f"\nH2: 주 고객층은 20-30대일 것이다.")
print(f"  → 검증: 주 고객층은 {main_age}")
print(f"  → 결론: {'채택' if main_age in ['20대', '30대'] else '기각'}")

# H3: 점포수와 매출의 관계
corr = df_project['점포수'].corr(df_project['월매출금액'])

print(f"\nH3: 점포수가 많은 곳은 매출도 높을 것이다.")
print(f"  → 검증: 점포수와 월매출의 상관계수 = {corr:.3f}")
print(f"  → 결론: {'채택' if corr > 0.3 else '기각'} (상관계수 {abs(corr):.2f})")

# H4: 평일과 주말 매출 차이
weekday_avg = df_project['총요일매출'].mean() / 5
weekend_avg = df_project['총주말매출'].mean() / 2
diff_ratio = abs(weekday_avg - weekend_avg) / weekday_avg * 100

print(f"\nH4: 평일과 주말의 매출 차이가 크지 않을 것이다.")
print(f"  → 검증: 평일 일평균 {weekday_avg/1000000:.2f}M, 주말 일평균 {weekend_avg/1000000:.2f}M")
print(f"  → 차이: {diff_ratio:.1f}%")
print(f"  → 결론: {'채택' if diff_ratio < 20 else '기각'} (차이 {diff_ratio:.1f}%)")
```

### 5.2 핵심 인사이트 정리

```python
INSIGHTS = f"""
================================
핵심 인사이트 및 발견사항
================================

1. 지역적 특성
   - 최고 매출 지역: {top_district}
   - 평균 대비 {(top_sales / df_project['월매출금액'].mean() - 1) * 100:.1f}% 높음
   - 입점 추천 지역: {', '.join(top5_districts)}

2. 고객 특성
   - 주요 고객층: {main_age}
   - 해당 연령대 평균 매출: {age_sales_avg.max()/1000000:.2f}M원
   - 마케팅 타겟: {main_age} 중심, 인접 연령대 포함

3. 경쟁 환경
   - 점포수-매출 상관: {corr:.2f} ({'양의 상관' if corr > 0 else '음의 상관'})
   - 경쟁이 심한 지역에서도 매출이 높을 수 있음
   - 입지 선정이 핵심

4. 운영 전략
   - 평일 일평균: {weekday_avg/1000000:.2f}M원
   - 주말 일평균: {weekend_avg/1000000:.2f}M원
   - 요일별 차별화 {'필요' if diff_ratio > 20 else '불필요'} (차이 {diff_ratio:.1f}%)

5. 수익성 예측
   - 중앙값 기준 예상 매출: {df_project['월매출금액'].median()/1000000:.2f}M원/월
   - 상위 25% 매출: {df_project['월매출금액'].quantile(0.75)/1000000:.2f}M원/월
   - 목표 매출 달성 확률 고려 필요
"""

print(INSIGHTS)

# 파일로 저장
with open('output/7_insights_report.txt', 'w', encoding='utf-8') as f:
    f.write(INSIGHTS)

print("\n✓ 인사이트 리포트 저장: output/7_insights_report.txt")
```

---

## 6교시: 발표 자료 준비

### 6.1 발표 구조

```
1. 표지 (30초)
   - 프로젝트 제목
   - 팀명/이름
   - 날짜

2. 문제 정의 (1분)
   - 비즈니스 상황
   - 핵심 질문
   - 분석 목표

3. 데이터 및 방법론 (1분)
   - 데이터 소스
   - 분석 방법
   - 주요 도구

4. 주요 발견사항 (3분)
   - 시각화 3-5개
   - 핵심 인사이트
   - 가설 검증 결과

5. 결론 및 제언 (1분)
   - 비즈니스 제안
   - 실행 방안
   - 한계점 및 향후 과제

총 6-7분
```

### 6.2 핵심 슬라이드 예시

```python
# 슬라이드 1: 핵심 발견 요약

summary_fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 지역별 Top 5
district_top5 = df_project.groupby('행정구역명')['월매출금액'].mean().nlargest(5)
axes[0, 0].barh(range(len(district_top5)), district_top5.values, color='steelblue')
axes[0, 0].set_yticks(range(len(district_top5)))
axes[0, 0].set_yticklabels(district_top5.index)
axes[0, 0].set_title('1. 최고 매출 지역 Top 5', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('평균 월매출 (원)')
axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

# 2. 연령대별 매출
age_avg = df_project[age_columns].mean()
axes[0, 1].bar(age_labels, age_avg.values, color='coral')
axes[0, 1].set_title('2. 주요 고객층', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('평균 매출 (원)')
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
max_idx = age_avg.argmax()
axes[0, 1].patches[max_idx].set_color('red')

# 3. 평일 vs 주말
ww_data = ['평일', '주말']
ww_sales = [weekday_avg, weekend_avg]
axes[1, 0].bar(ww_data, ww_sales, color=['#457B9D', '#E63946'])
axes[1, 0].set_title('3. 평일 vs 주말 일평균 매출', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('일평균 매출 (원)')
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
for i, v in enumerate(ww_sales):
    axes[1, 0].text(i, v, f'{v/1000000:.1f}M', ha='center', va='bottom', fontweight='bold')

# 4. 매출 분포 (박스플롯)
axes[1, 1].boxplot([df_project['월매출금액']], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen'))
axes[1, 1].set_title('4. 전체 매출 분포', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('월매출 (원)')
axes[1, 1].set_xticklabels([''])
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

# 통계값 표시
median = df_project['월매출금액'].median()
q1 = df_project['월매출금액'].quantile(0.25)
q3 = df_project['월매출금액'].quantile(0.75)
axes[1, 1].text(1.15, median, f'중앙값: {median/1000000:.1f}M', fontsize=10)
axes[1, 1].text(1.15, q1, f'Q1: {q1/1000000:.1f}M', fontsize=10)
axes[1, 1].text(1.15, q3, f'Q3: {q3/1000000:.1f}M', fontsize=10)

plt.suptitle(f'{target_industry} 분석 - 핵심 발견사항', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('output/7_presentation_summary.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 6.3 제언 및 액션 아이템

```python
RECOMMENDATIONS = f"""
================================
비즈니스 제언 및 실행 방안
================================

▶ 즉시 실행 가능 (Quick Wins)
  1. 입점 지역: {top_district} 우선 검토
  2. 타겟 마케팅: {main_age} 중심 프로모션
  3. SNS 마케팅: 젊은 층 공략 (Instagram, TikTok)

▶ 중기 전략 (3-6개월)
  1. 메뉴 개발: {main_age} 선호 메뉴 조사 및 출시
  2. 공간 디자인: 인스타그램 감성 인테리어
  3. 제휴 마케팅: 주변 대학, 기업과 협력

▶ 장기 전략 (6-12개월)
  1. 다점포 확장: 검증된 지역 중심으로 확대
  2. 데이터 기반 운영: 일일 매출 추적 시스템 구축
  3. 고객 로열티 프로그램: 재방문율 향상

▶ 리스크 관리
  1. 경쟁 분석: 주기적 모니터링 필요
  2. 임대료 협상: 매출 대비 10-15% 유지
  3. 계절성 고려: 비수기 대비 프로모션

▶ 성과 지표 (KPI)
  1. 월 매출: {df_project['월매출금액'].median()/1000000:.1f}M원 이상
  2. 일평균 방문객: 50명 이상 (추정)
  3. 재방문율: 30% 이상
  4. 고객 만족도: 4.5/5.0 이상

▶ 향후 분석 과제
  1. 시간대별 매출 데이터 추가 분석
  2. 날씨/계절 영향 분석
  3. 경쟁사 가격 전략 벤치마킹
  4. 고객 리뷰 텍스트 분석 (감성 분석)
"""

print(RECOMMENDATIONS)

with open('output/7_recommendations.txt', 'w', encoding='utf-8') as f:
    f.write(RECOMMENDATIONS)

print("\n✓ 제언 리포트 저장: output/7_recommendations.txt")
```

---

## 7교시: 발표 및 피드백

### 7.1 발표 체크리스트

```
□ 시간 준수 (6-7분)
□ 명확한 문제 정의
□ 시각화 가독성
□ 인사이트 구체성
□ 실행 가능한 제언
□ 질의응답 준비
```

### 7.2 예상 질문 및 답변 준비

```python
QA_PREP = """
================================
예상 질문 및 답변 준비
================================

Q1: 데이터의 시간 범위는 어떻게 되나요?
A: 2023년 3월~12월 데이터를 활용했습니다. 최신 트렌드를 반영하기 위해 
   최근 1년 데이터에 집중했습니다.

Q2: 이상치는 어떻게 처리했나요?
A: IQR 방법으로 이상치를 탐지했으나, 제거하지 않고 별도 분석했습니다.
   고매출 이상치는 성공 사례로 벤치마킹했습니다.

Q3: 다른 업종과 비교했을 때 어떤가요?
A: {target_industry}는 평균 {df_project['월매출금액'].mean()/1000000:.1f}M원으로
   서비스업 전체 평균 대비 [높음/낮음/비슷]한 수준입니다.

Q4: 실제 입점 시 고려할 추가 요인은?
A: 임대료, 유동인구, 주차 가능 여부, 경쟁사 거리, 접근성 등을
   현장 실사를 통해 추가로 검토해야 합니다.

Q5: 분석의 한계는 무엇인가요?
A: 1) 시간대별 세부 데이터 부족
   2) 개별 매장의 브랜드/서비스 품질 미반영
   3) 계절성 분석 부족
   → 추가 데이터 확보 시 보완 가능

Q6: ROI는 어떻게 되나요?
A: 초기 투자비 [X]만원 가정 시, 중앙값 매출 기준 약 [Y]개월 회수 예상
   (상세 재무 분석 필요)
"""

print(QA_PREP)
```

### 7.3 피드백 및 개선

**발표 후 자기 평가:**
```
□ 시간 관리가 적절했는가?
□ 청중이 이해하기 쉬웠는가?
□ 시각화가 효과적이었는가?
□ 인사이트가 명확했는가?
□ 질문에 잘 대응했는가?

개선 사항:
- 
- 
- 
```

---

## 프로젝트 완료 체크리스트

```
□ 데이터 전처리 완료
□ EDA 수행 및 패턴 파악
□ 핵심 시각화 3개 이상 제작
□ 가설 검증
□ 인사이트 도출
□ 비즈니스 제언 작성
□ 발표 자료 준비
□ 발표 완료
□ 피드백 반영

선택 사항:
□ GitHub에 프로젝트 업로드
□ 블로그/포트폴리오에 정리
□ 추가 분석 수행
```

---

## 학습 마무리

### 축하합니다! 🎉

3주 과정의 **서울시 상권분석 데이터 시각화 교육**을 완료하셨습니다!

**배운 내용 정리:**
- ✅ Matplotlib: 기본 차트 (라인, 바, 히스토그램, 산점도)
- ✅ Seaborn: 통계적 시각화 (바, 박스, 바이올린, 히트맵)
- ✅ Folium: 지리공간 시각화 (마커, 클러스터, 히트맵)
- ✅ 실무 프로젝트: End-to-end 데이터 분석 경험

**다음 단계:**
1. 자신의 데이터로 프로젝트 수행
2. Kaggle 등에서 다양한 데이터셋 분석
3. 고급 시각화 학습 (Plotly, Dash 등)
4. 머신러닝 모델 결과 시각화
5. 대시보드 구축 (Streamlit, Tableau 등)

**포트폴리오 활용:**
- 이 프로젝트를 GitHub에 업로드
- 블로그에 분석 과정 정리
- 이력서/포트폴리오에 추가

감사합니다! 🙏
```

---

## 추가 리소스

### 추천 학습 자료
- [Matplotlib 공식 문서](https://matplotlib.org/)
- [Seaborn 튜토리얼](https://seaborn.pydata.org/tutorial.html)
- [Folium 문서](https://python-visualization.github.io/folium/)
- [서울 열린데이터광장](https://data.seoul.go.kr/)

### 연습 문제
1. 다른 업종(한식음식점, 주점 등)으로 동일한 분석 수행
2. 계절성 분석: 분기별 매출 변화 탐색
3. 다변량 분석: 지역 + 업종 + 연령대 동시 고려
4. 예측 모델: 매출 예측 회귀 모델 구축
