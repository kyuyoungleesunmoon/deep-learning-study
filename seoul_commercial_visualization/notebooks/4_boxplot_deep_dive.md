# 4교시: Box Plot - 업종/지역별 매출 분포 심화 분석

**학습 목표:**
- Boxplot을 활용한 다중 그룹 분포 비교
- 이상치(Outlier) 탐지 및 분석
- 분포의 통계적 비교 방법
- 실무 의사결정을 위한 분포 분석 기법

**예상 소요 시간:** 60분

---

## 1. Boxplot 심화 이해

### 1.1 Boxplot의 구성 요소 상세

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

# Boxplot 설명을 위한 샘플 데이터
sample_industry = df[df['서비스업종코드명'] == '커피/음료']['월매출금액']

# Boxplot의 통계값 계산
Q1 = sample_industry.quantile(0.25)  # 1사분위수 (25%)
Q2 = sample_industry.quantile(0.50)  # 중앙값 (50%)
Q3 = sample_industry.quantile(0.75)  # 3사분위수 (75%)
IQR = Q3 - Q1  # 사분위범위
lower_whisker = Q1 - 1.5 * IQR  # 하위 수염
upper_whisker = Q3 + 1.5 * IQR  # 상위 수염

print("=" * 60)
print("Boxplot 구성 요소 (커피/음료 업종 월매출)")
print("=" * 60)
print(f"최솟값: {sample_industry.min()/1000000:.2f}M원")
print(f"Q1 (25%): {Q1/1000000:.2f}M원")
print(f"Q2 (중앙값, 50%): {Q2/1000000:.2f}M원")
print(f"Q3 (75%): {Q3/1000000:.2f}M원")
print(f"최댓값: {sample_industry.max()/1000000:.2f}M원")
print(f"\nIQR (사분위범위): {IQR/1000000:.2f}M원")
print(f"하위 경계 (Q1 - 1.5*IQR): {lower_whisker/1000000:.2f}M원")
print(f"상위 경계 (Q3 + 1.5*IQR): {upper_whisker/1000000:.2f}M원")

# 이상치 탐지
outliers = sample_industry[(sample_industry < lower_whisker) | (sample_industry > upper_whisker)]
print(f"\n이상치 개수: {len(outliers)}개 ({len(outliers)/len(sample_industry)*100:.1f}%)")
```

### 1.2 시각화로 이해하기

```python
# Boxplot과 함께 구성 요소 표시
fig, ax = plt.subplots(figsize=(12, 8))

# Boxplot
bp = ax.boxplot([sample_industry], vert=True, widths=0.3, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# 주석 추가
ax.text(1.15, Q1, f'Q1 (25%): {Q1/1000000:.1f}M', fontsize=11, va='center')
ax.text(1.15, Q2, f'중앙값 (50%): {Q2/1000000:.1f}M', fontsize=11, va='center', 
        fontweight='bold', color='red')
ax.text(1.15, Q3, f'Q3 (75%): {Q3/1000000:.1f}M', fontsize=11, va='center')

# 제목 및 레이블
ax.set_title('Boxplot 구성 요소 상세 설명\n(커피/음료 업종 월매출)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.set_xticklabels(['커피/음료'])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/4_boxplot_components.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 2. 다중 그룹 Boxplot 비교

### 2.1 업종별 매출 분포 상세 비교

```python
# 주요 업종 선택
major_industries = ['커피/음료', '한식음식점', '편의점', '치킨전문점', '의류', '화장품']
df_major = df[df['서비스업종코드명'].isin(major_industries)]

# 크기가 큰 Boxplot
fig, ax = plt.subplots(figsize=(14, 8))

# Seaborn boxplot with enhanced styling
sns.boxplot(data=df_major,
            x='서비스업종코드명',
            y='월매출금액',
            order=sorted(major_industries),
            palette='Set3',
            ax=ax,
            linewidth=1.5)

# 평균값 추가 (다이아몬드 마커)
means = df_major.groupby('서비스업종코드명')['월매출금액'].mean()
positions = range(len(sorted(major_industries)))
for pos, industry in enumerate(sorted(major_industries)):
    ax.scatter(pos, means[industry], marker='D', s=100, color='red', 
              zorder=3, label='평균' if pos == 0 else '')

# 그래프 꾸미기
ax.set_title('주요 업종별 월매출 분포 비교\n(빨간 다이아몬드 = 평균)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
plt.xticks(rotation=45, ha='right')
ax.grid(True, axis='y', linestyle='--', alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('output/4_multi_industry_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 각 업종 통계 출력
print("\n" + "=" * 80)
print("업종별 통계 요약")
print("=" * 80)
print(f"{'업종':<15} {'중앙값':>12} {'평균':>12} {'IQR':>12} {'변동계수':>12}")
print("-" * 80)

for industry in sorted(major_industries):
    data = df_major[df_major['서비스업종코드명'] == industry]['월매출금액']
    median = data.median()
    mean = data.mean()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    cv = (data.std() / mean) * 100  # 변동계수 (Coefficient of Variation)
    
    print(f"{industry:<15} {median/1000000:>10.2f}M {mean/1000000:>10.2f}M "
          f"{iqr/1000000:>10.2f}M {cv:>10.1f}%")
```

### 2.2 결과 해석

**실무 인사이트:**

1. **중앙값 vs 평균**:
   - 평균이 중앙값보다 높으면: 고매출 이상치가 존재
   - 평균과 중앙값이 비슷하면: 비교적 균등한 분포

2. **IQR (사분위범위)**:
   - IQR이 클수록 매출의 변동성이 큼
   - 성공과 실패의 격차가 큰 업종

3. **변동계수 (CV)**:
   - CV가 낮을수록 안정적인 업종
   - CV가 높을수록 리스크와 기회가 큰 업종

---

## 3. 지역별 비교 Boxplot

### 3.1 구별 특정 업종 매출 분포

```python
# 커피/음료 업종의 주요 5개 구 비교
coffee_data = df[df['서비스업종코드명'] == '커피/음료']
top5_districts = coffee_data.groupby('행정구역명')['월매출금액'].median().nlargest(5).index

coffee_top5 = coffee_data[coffee_data['행정구역명'].isin(top5_districts)]

# Boxplot
fig, ax = plt.subplots(figsize=(14, 8))

sns.boxplot(data=coffee_top5,
            x='행정구역명',
            y='월매출금액',
            order=top5_districts,
            palette='coolwarm',
            ax=ax)

# 그래프 꾸미기
ax.set_title('커피/음료 업종 - 구별 월매출 분포 Top 5\n(중앙값 기준)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/4_coffee_district_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 구별 통계
print("\n" + "=" * 70)
print("커피/음료 업종 - 구별 통계")
print("=" * 70)
for district in top5_districts:
    data = coffee_top5[coffee_top5['행정구역명'] == district]['월매출금액']
    print(f"\n[{district}]")
    print(f"  중앙값: {data.median()/1000000:.2f}M원")
    print(f"  평균: {data.mean()/1000000:.2f}M원")
    print(f"  IQR: {(data.quantile(0.75) - data.quantile(0.25))/1000000:.2f}M원")
    print(f"  데이터 수: {len(data)}개")
```

---

## 4. 이상치 분석

### 4.1 이상치 탐지 및 시각화

```python
# 한식음식점 업종의 이상치 분석
korean_food = df[df['서비스업종코드명'] == '한식음식점']['월매출금액']

Q1 = korean_food.quantile(0.25)
Q3 = korean_food.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 추출
outliers_data = df[df['서비스업종코드명'] == '한식음식점'].copy()
outliers_data['is_outlier'] = (outliers_data['월매출금액'] < lower_bound) | \
                               (outliers_data['월매출금액'] > upper_bound)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 좌측: Boxplot
sns.boxplot(y=korean_food, ax=ax1, color='lightcoral')
ax1.set_title('한식음식점 매출 분포\n(Boxplot)', fontsize=14, fontweight='bold')
ax1.set_ylabel('월매출금액 (원)', fontsize=12)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

# 우측: 히스토그램 with 이상치 강조
ax2.hist(korean_food[korean_food <= upper_bound], bins=30, 
         color='lightblue', alpha=0.7, label='정상 범위')
outliers_upper = korean_food[korean_food > upper_bound]
if len(outliers_upper) > 0:
    ax2.hist(outliers_upper, bins=10, color='red', alpha=0.7, label='이상치 (상위)')

ax2.axvline(upper_bound, color='red', linestyle='--', linewidth=2, 
            label=f'상위 경계: {upper_bound/1000000:.1f}M')
ax2.axvline(korean_food.median(), color='green', linestyle='--', linewidth=2,
            label=f'중앙값: {korean_food.median()/1000000:.1f}M')

ax2.set_title('한식음식점 매출 분포\n(히스토그램 + 이상치)', fontsize=14, fontweight='bold')
ax2.set_xlabel('월매출금액 (원)', fontsize=12)
ax2.set_ylabel('빈도', fontsize=12)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/4_outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 이상치 상세 정보
outliers = outliers_data[outliers_data['is_outlier'] == True]
print("\n" + "=" * 60)
print(f"이상치 분석 결과 - 한식음식점 업종")
print("=" * 60)
print(f"전체 데이터: {len(outliers_data)}개")
print(f"이상치: {len(outliers)}개 ({len(outliers)/len(outliers_data)*100:.1f}%)")
print(f"\n이상치 매출 범위:")
print(f"  최솟값: {outliers['월매출금액'].min()/1000000:.2f}M원")
print(f"  최댓값: {outliers['월매출금액'].max()/1000000:.2f}M원")

if len(outliers) > 0:
    print(f"\n이상치 상위 5개 상권:")
    top_outliers = outliers.nlargest(5, '월매출금액')[['상권코드명', '행정구역명', '월매출금액']]
    for idx, row in top_outliers.iterrows():
        print(f"  - {row['상권코드명']} ({row['행정구역명']}): {row['월매출금액']/1000000:.2f}M원")
```

### 4.2 이상치 해석

**실무 인사이트:**
- **상위 이상치**: 매우 성공한 매장 → 성공 요인 분석 필요
  - 입지? 브랜드? 서비스 품질?
- **하위 이상치**: 어려움을 겪는 매장 → 문제점 파악
  - 폐업 위험군, 개선 필요
- **이상치 제외 여부**: 
  - 평균 계산 시 이상치 제외 고려
  - 하지만 이상치도 중요한 정보!

---

## 5. 그룹 간 통계적 비교

### 5.1 Notched Boxplot (중앙값 신뢰구간)

```python
# Notched boxplot: 중앙값의 신뢰구간을 V자 홈(notch)으로 표시
# 홈이 겹치지 않으면 중앙값이 통계적으로 유의미하게 다름

industries_compare = ['커피/음료', '한식음식점', '주점']
df_compare = df[df['서비스업종코드명'].isin(industries_compare)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 일반 Boxplot
sns.boxplot(data=df_compare, x='서비스업종코드명', y='월매출금액',
            palette='pastel', ax=ax1)
ax1.set_title('일반 Boxplot', fontsize=14, fontweight='bold')
ax1.set_xlabel('', fontsize=12)
ax1.set_ylabel('월매출금액 (원)', fontsize=12)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax1.grid(True, axis='y', alpha=0.3)

# Notched Boxplot (중앙값 신뢰구간)
bp = ax2.boxplot([df_compare[df_compare['서비스업종코드명'] == ind]['월매출금액'] 
                   for ind in industries_compare],
                  labels=industries_compare,
                  notch=True,  # 홈 표시
                  patch_artist=True,
                  boxprops=dict(facecolor='lightgreen', alpha=0.7))

ax2.set_title('Notched Boxplot\n(V자 홈 = 중앙값 95% 신뢰구간)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('', fontsize=12)
ax2.set_ylabel('월매출금액 (원)', fontsize=12)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('output/4_notched_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n** Notched Boxplot 해석 **")
print("V자 홈(notch)이 겹치지 않으면 중앙값이 통계적으로 유의미하게 다릅니다.")
print("홈이 겹치면 중앙값 차이가 통계적으로 명확하지 않을 수 있습니다.")
```

---

## 6. 시간에 따른 분포 변화

### 6.1 분기별 매출 분포 변화

```python
# 커피/음료 업종의 분기별 매출 분포
coffee_by_quarter = df[df['서비스업종코드명'] == '커피/음료'].copy()

# 기준년월을 분기로 변환
coffee_by_quarter['분기'] = coffee_by_quarter['기준년월코드'].astype(str).str[:4] + 'Q' + \
                            ((coffee_by_quarter['기준년월코드'].astype(str).str[4:6].astype(int) - 1) // 3 + 1).astype(str)

# Boxplot
fig, ax = plt.subplots(figsize=(14, 8))

# 분기 순서대로 정렬
quarter_order = sorted(coffee_by_quarter['분기'].unique())

sns.boxplot(data=coffee_by_quarter,
            x='분기',
            y='월매출금액',
            order=quarter_order,
            palette='viridis',
            ax=ax)

# 평균값 연결선 추가
means = coffee_by_quarter.groupby('분기')['월매출금액'].mean()
positions = range(len(quarter_order))
ax.plot(positions, [means[q] for q in quarter_order], 
        'r--o', linewidth=2, markersize=8, label='평균 추세')

# 그래프 꾸미기
ax.set_title('커피/음료 업종 - 분기별 월매출 분포 변화', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('분기', fontsize=12)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax.legend(fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/4_quarterly_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 7. 실무 적용 예시

### 시나리오: 신규 카페 입점 지역 및 예상 매출 분석

```python
# 3개 후보 지역의 커피/음료 매출 분포 비교
candidate_districts = ['강남구', '마포구', '송파구']
coffee_candidates = df[(df['서비스업종코드명'] == '커피/음료') & 
                      (df['행정구역명'].isin(candidate_districts))]

fig, ax = plt.subplots(figsize=(14, 8))

# Boxplot with stripplot overlay (실제 데이터 포인트 표시)
sns.boxplot(data=coffee_candidates,
            x='행정구역명',
            y='월매출금액',
            palette='Set2',
            ax=ax)

sns.stripplot(data=coffee_candidates,
              x='행정구역명',
              y='월매출금액',
              color='black',
              alpha=0.3,
              size=3,
              ax=ax)

# 목표 매출선 추가 (예: 5M원)
target_revenue = 5000000
ax.axhline(target_revenue, color='red', linestyle='--', linewidth=2,
           label=f'목표 매출: {target_revenue/1000000}M원')

# 그래프 꾸미기
ax.set_title('신규 카페 입점 후보지 매출 분포 비교\n(검은 점 = 실제 매장 매출)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax.legend(fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/4_cafe_location_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 의사결정 리포트
print("\n" + "=" * 70)
print("신규 카페 입점 의사결정 리포트")
print("=" * 70)

for district in candidate_districts:
    data = coffee_candidates[coffee_candidates['행정구역명'] == district]['월매출금액']
    
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    success_rate = (data >= target_revenue).sum() / len(data) * 100
    
    print(f"\n[{district}]")
    print(f"  중앙값 (예상 매출): {median/1000000:.2f}M원")
    print(f"  25% 매장: {q1/1000000:.2f}M원 이하")
    print(f"  75% 매장: {q3/1000000:.2f}M원 이하")
    print(f"  목표 달성 비율: {success_rate:.1f}%")
    print(f"  종합 평가: ", end='')
    
    if median >= target_revenue and success_rate >= 50:
        print("✓ 우수 (목표 달성 가능성 높음)")
    elif median >= target_revenue * 0.8:
        print("△ 보통 (목표 달성 가능, 노력 필요)")
    else:
        print("✗ 신중 (목표 달성 어려움)")
```

---

## 8. 학습 정리

### 8.1 핵심 개념

1. **Boxplot 구성**: Q1, Q2(중앙값), Q3, IQR, 수염, 이상치
2. **통계적 비교**: 중앙값, 평균, IQR, 변동계수
3. **이상치 분석**: 특별한 케이스 탐지 및 원인 분석
4. **Notched Boxplot**: 중앙값의 통계적 유의성 판단
5. **실무 활용**: 목표 설정, 위험도 평가, 의사결정 지원

### 8.2 실무 활용 팁

- **중앙값 우선**: 평균보다 중앙값이 실제 상황을 더 잘 반영
- **IQR 확인**: 변동성이 큰 시장인지 안정적인 시장인지 판단
- **이상치 분석**: 성공 사례와 실패 사례의 특징 파악
- **목표선 추가**: 내 목표 매출이 현실적인지 시각적으로 확인

---

## 9. 학습 확인 퀴즈

### 문제 1
Boxplot에서 상자의 높이(IQR)가 크다는 것은 무엇을 의미하는가?
1. 평균이 높다
2. 데이터의 변동성이 크다
3. 이상치가 많다
4. 중앙값이 높다

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 데이터의 변동성이 크다**

**해설:** 
IQR (Interquartile Range, 사분위범위)는 Q3 - Q1로 계산되며, 데이터의 중간 50%가 분포하는 범위를 나타냅니다. IQR이 클수록 데이터가 넓게 퍼져 있어 변동성이 크다는 의미입니다. 이는 성공과 실패의 격차가 큰 시장을 나타낼 수 있습니다.
</details>

---

### 문제 2
이상치(outlier)를 판단하는 기준은?
1. 평균 ± 표준편차
2. Q1 - 1.5*IQR 또는 Q3 + 1.5*IQR
3. 중앙값 ± 2*표준편차
4. 최솟값과 최댓값

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. Q1 - 1.5*IQR 또는 Q3 + 1.5*IQR**

**해설:** 
일반적으로 이상치는 다음 범위를 벗어나는 값으로 정의됩니다:
- 하위 이상치: Q1 - 1.5 × IQR보다 작은 값
- 상위 이상치: Q3 + 1.5 × IQR보다 큰 값
이는 Boxplot에서 수염 밖에 위치한 점들로 표시됩니다.
</details>

---

### 문제 3
Notched Boxplot에서 두 그룹의 홈(notch)이 겹치지 않으면?
1. 평균이 같다
2. 중앙값이 통계적으로 유의미하게 다르다
3. 분산이 같다
4. 이상치가 없다

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 중앙값이 통계적으로 유의미하게 다르다**

**해설:** 
Notched Boxplot의 홈은 중앙값의 95% 신뢰구간을 나타냅니다. 두 그룹의 홈이 겹치지 않으면, 두 그룹의 중앙값이 통계적으로 유의미하게 다르다고 판단할 수 있습니다 (약 95% 신뢰수준). 이는 실무에서 두 그룹을 비교할 때 유용한 정보입니다.
</details>

---

### 문제 4
Boxplot에서 빨간 다이아몬드로 평균을 표시할 때, 평균이 상자 위쪽에 위치하면?
1. 정규분포를 따른다
2. 오른쪽으로 치우친 분포 (상위 이상치 존재)
3. 왼쪽으로 치우친 분포
4. 균등 분포

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 오른쪽으로 치우친 분포 (상위 이상치 존재)**

**해설:** 
상자의 중앙에는 중앙값(median)이 표시됩니다. 평균이 중앙값보다 위에 있다면, 이는 상위에 큰 값들이 있어 평균을 끌어올린다는 의미입니다. 이는 오른쪽으로 치우친 분포(right-skewed)의 특징이며, 매출 데이터에서 흔히 나타납니다.
</details>

---

### 문제 5
실무에서 신규 입점 지역을 선택할 때, Boxplot을 어떻게 활용하는가?
1. 최댓값이 가장 높은 지역 선택
2. 중앙값과 IQR을 고려하여 안정적이고 기대 매출이 높은 지역 선택
3. 이상치가 가장 많은 지역 선택
4. 최솟값이 가장 높은 지역 선택

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 중앙값과 IQR을 고려하여 안정적이고 기대 매출이 높은 지역 선택**

**해설:** 
실무 의사결정에서는 다음을 종합적으로 고려해야 합니다:
- **중앙값**: 예상 매출 (평균보다 현실적)
- **IQR**: 변동성/위험도 (IQR이 작을수록 안정적)
- **목표 달성 비율**: 내 목표 매출을 달성한 매장 비율

단순히 최댓값이나 최솟값만 보는 것은 위험합니다. 전체적인 분포를 파악하는 것이 중요합니다.
</details>

---

## 다음 교시 예고

**5교시: Heatmap - 변수 간 상관관계 인사이트 도출**
- 상관행렬(Correlation Matrix) 생성 및 해석
- Heatmap으로 다차원 데이터 시각화
- 업종/지역/연령대 간 패턴 발견
- 실무 의사결정을 위한 상관관계 분석
