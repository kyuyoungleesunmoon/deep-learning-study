# 3교시: Seaborn - 미적으로 우수한 시각화

**학습 목표:**
- Seaborn의 개념과 Matplotlib과의 차이점 이해
- Seaborn으로 카테고리별 비교 시각화
- 통계적 정보가 포함된 그래프 작성
- 실무에서 바로 사용 가능한 보고서용 그래프 제작

**예상 소요 시간:** 60분

---

## 1. Seaborn 소개

### 1.1 Seaborn이란?

**Seaborn**은 Matplotlib을 기반으로 만들어진 고급 시각화 라이브러리입니다.

**Matplotlib vs Seaborn:**

| 특징 | Matplotlib | Seaborn |
|------|-----------|---------|
| 난이도 | 중간 (세밀한 제어) | 쉬움 (간결한 코드) |
| 기본 스타일 | 기본적 | 현대적, 아름다움 |
| 통계 기능 | 수동 계산 필요 | 자동 계산 (평균, 신뢰구간 등) |
| Pandas 연동 | 수동 변환 | 자동 인식 |
| 용도 | 세밀한 커스터마이징 | 빠른 탐색, 보고서 |

**비유:** 
- Matplotlib = 수동 변속기 자동차 (자유롭지만 복잡)
- Seaborn = 자동 변속기 자동차 (쉽고 편리)

### 1.2 기본 설정

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일 설정
sns.set_style("whitegrid")  # 스타일: darkgrid, whitegrid, dark, white, ticks
sns.set_palette("husl")  # 색상 팔레트

# 데이터 불러오기
df = pd.read_csv("data/서울시_상권_추정매출.csv", encoding="cp949")
```

---

## 2. Barplot: 카테고리별 평균 비교

### 2.1 개념 설명

**sns.barplot()**은 카테고리별 평균을 자동으로 계산하고, 신뢰구간까지 표시합니다.
- **자동 계산**: groupby()와 mean()을 자동으로 수행
- **신뢰구간**: 평균의 불확실성을 검은 선으로 표시
- **용도**: "어느 업종의 평균 매출이 높은가?" (통계적으로)

### 2.2 실습: 업종별 평균 월매출 비교

```python
# 상위 8개 업종 선택
top_industries = df.groupby('서비스업종코드명')['월매출금액'].mean().nlargest(8).index
df_top = df[df['서비스업종코드명'].isin(top_industries)]

# Seaborn barplot
fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(data=df_top, 
            x='월매출금액', 
            y='서비스업종코드명',
            order=top_industries,  # 정렬 순서
            palette='rocket',  # 색상 팔레트
            ax=ax)

# 그래프 꾸미기
ax.set_title('업종별 평균 월매출 Top 8\n(95% 신뢰구간 포함)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('평균 월매출금액 (원)', fontsize=12)
ax.set_ylabel('', fontsize=12)

# x축 포맷팅
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

plt.tight_layout()
plt.savefig('output/3_seaborn_barplot_top8.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2.3 결과 해석

**실무 인사이트:**
- **검은 선 (신뢰구간)**: 짧을수록 데이터가 일관적, 길수록 변동성이 큼
- **비교 방법**: 
  - 신뢰구간이 겹치지 않으면 통계적으로 유의미한 차이
  - 신뢰구간이 겹치면 차이가 명확하지 않을 수 있음

---

## 3. Countplot: 빈도 비교

### 3.1 개념 설명

**sns.countplot()**은 각 카테고리의 빈도(개수)를 자동으로 계산하고 시각화합니다.
- **용도**: "어느 업종이 가장 많은가?", "어느 구에 매장이 많은가?"

### 3.2 실습: 구별 상권 개수

```python
# 주요 10개 구 선택
top_districts = df['행정구역명'].value_counts().head(10).index
df_districts = df[df['행정구역명'].isin(top_districts)]

# Countplot
fig, ax = plt.subplots(figsize=(12, 8))

sns.countplot(data=df_districts,
              y='행정구역명',
              order=top_districts,
              palette='viridis',
              ax=ax)

# 꾸미기
ax.set_title('구별 상권 데이터 개수 Top 10', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('상권 개수', fontsize=12)
ax.set_ylabel('', fontsize=12)

# 각 막대에 정확한 수치 표시
for container in ax.containers:
    ax.bar_label(container, fmt='%d개', padding=5)

plt.tight_layout()
plt.savefig('output/3_seaborn_countplot_districts.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 4. Boxplot과 Violinplot: 분포 비교

### 4.1 개념 설명

**Boxplot (상자 그림)**:
- 데이터의 분포를 요약 (최솟값, 25%, 중앙값, 75%, 최댓값)
- 이상치(outlier)를 점으로 표시

**Violinplot (바이올린 그림)**:
- Boxplot + 분포 형태 (히스토그램의 회전)
- 데이터의 밀도를 시각화

### 4.2 실습: 업종별 매출 분포 비교 (Boxplot)

```python
# 주요 5개 업종
industries_for_box = ['커피/음료', '한식음식점', '편의점', '치킨전문점', '의류']
df_box = df[df['서비스업종코드명'].isin(industries_for_box)]

# Boxplot
fig, ax = plt.subplots(figsize=(12, 8))

sns.boxplot(data=df_box,
            x='서비스업종코드명',
            y='월매출금액',
            palette='Set2',
            ax=ax)

# 꾸미기
ax.set_title('업종별 월매출금액 분포 비교 (Box Plot)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

# x축 레이블 회전
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('output/3_seaborn_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4.3 Boxplot 해석 방법

```
      최댓값 (위 수염)
         |
    ┌────┴────┐
    │         │  75% (3사분위수)
    │    ─    │  중앙값 (50%)
    │         │  25% (1사분위수)
    └────┬────┘
         |
      최솟값 (아래 수염)
    
    ● ●  이상치 (outliers)
```

**실무 인사이트:**
- 상자의 높이 (IQR): 데이터의 변동성
- 중앙선 위치: 분포의 치우침
- 이상치: 특별히 높거나 낮은 매출 → 추가 분석 필요

### 4.4 실습: 업종별 매출 분포 비교 (Violinplot)

```python
# Violinplot
fig, ax = plt.subplots(figsize=(12, 8))

sns.violinplot(data=df_box,
               x='서비스업종코드명',
               y='월매출금액',
               palette='muted',
               ax=ax)

# 꾸미기
ax.set_title('업종별 월매출금액 분포 비교 (Violin Plot)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/3_seaborn_violinplot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4.5 결과 해석

**Violinplot의 장점:**
- 분포의 형태를 직관적으로 파악 (단봉, 쌍봉 등)
- 넓은 부분: 데이터가 많이 분포
- 좁은 부분: 데이터가 적게 분포

---

## 5. Pointplot: 변화 추세 비교

### 5.1 개념 설명

**sns.pointplot()**은 카테고리별 평균을 점으로 연결하여 추세를 시각화합니다.
- **용도**: 시간/조건에 따른 변화 비교
- **특징**: 신뢰구간을 막대로 표시

### 5.2 실습: 연령대별 업종 선호도

```python
# 연령대 데이터 재구조화
age_columns = ['10대매출금액', '20대매출금액', '30대매출금액', 
               '40대매출금액', '50대매출금액', '60대이상매출금액']

# 3개 업종 선택
industries_for_age = ['커피/음료', '한식음식점', '주점']

# 데이터 재구조화
age_data = []
for industry in industries_for_age:
    industry_df = df[df['서비스업종코드명'] == industry]
    for age_col in age_columns:
        age_group = age_col.replace('매출금액', '')
        avg_sales = industry_df[age_col].mean()
        age_data.append({
            '업종': industry,
            '연령대': age_group,
            '평균매출': avg_sales
        })

age_df = pd.DataFrame(age_data)

# Pointplot
fig, ax = plt.subplots(figsize=(12, 7))

sns.pointplot(data=age_df,
              x='연령대',
              y='평균매출',
              hue='업종',
              markers=['o', 's', '^'],  # 마커 모양
              linestyles=['-', '--', '-.'],  # 선 스타일
              palette='tab10',
              ax=ax)

# 꾸미기
ax.set_title('업종별 연령대 평균 매출 비교', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('연령대', fontsize=12)
ax.set_ylabel('평균 매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
ax.legend(title='업종', fontsize=11, title_fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/3_seaborn_pointplot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.3 결과 해석

**실무 인사이트:**
- 커피/음료: 20-30대에서 피크
- 주점: 30-40대에서 높은 매출
- 한식음식점: 전 연령대에서 고른 분포

---

## 6. 스타일과 테마

### 6.1 Seaborn 스타일

```python
# 5가지 스타일 비교
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for idx, style in enumerate(styles):
    sns.set_style(style)
    
    # 샘플 플롯
    sample_data = df.groupby('서비스업종코드명')['월매출금액'].mean().nlargest(5)
    
    axes[idx].bar(range(len(sample_data)), sample_data.values, color='steelblue')
    axes[idx].set_title(f'스타일: {style}', fontsize=12, fontweight='bold')
    axes[idx].set_xticks([])
    axes[idx].set_ylabel('매출' if idx == 0 else '')

plt.tight_layout()
plt.savefig('output/3_seaborn_styles.png', dpi=300, bbox_inches='tight')
plt.show()

# 원래 스타일로 복구
sns.set_style("whitegrid")
```

### 6.2 색상 팔레트

```python
# 다양한 색상 팔레트
palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, palette in enumerate(palettes):
    sample_data = df.groupby('서비스업종코드명')['월매출금액'].mean().nlargest(6)
    
    sns.barplot(x=list(range(len(sample_data))), 
                y=sample_data.values,
                palette=palette,
                ax=axes[idx])
    
    axes[idx].set_title(f'팔레트: {palette}', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('평균 매출' if idx % 3 == 0 else '')
    axes[idx].set_xticklabels([])

plt.tight_layout()
plt.savefig('output/3_seaborn_palettes.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 7. 실무 적용 예시

### 시나리오: 업종 및 지역별 경쟁 분석 보고서

```python
# 강남구, 마포구, 종로구의 커피/음료, 한식음식점 매출 비교
selected_districts = ['강남구', '마포구', '종로구']
selected_industries = ['커피/음료', '한식음식점']

df_analysis = df[(df['행정구역명'].isin(selected_districts)) & 
                 (df['서비스업종코드명'].isin(selected_industries))]

# Catplot (여러 카테고리 비교에 최적)
g = sns.catplot(data=df_analysis,
                x='행정구역명',
                y='월매출금액',
                hue='서비스업종코드명',
                kind='bar',  # bar, box, violin, point 등 선택 가능
                height=6,
                aspect=1.5,
                palette='Set2',
                legend_out=False)

# 꾸미기
g.set_axis_labels('행정구역', '평균 월매출금액 (원)', fontsize=12)
g.set_xticklabels(fontsize=11)
g.legend.set_title('업종')
g.fig.suptitle('지역별/업종별 평균 월매출 비교\n(신규 입점 의사결정 자료)', 
               fontsize=16, fontweight='bold', y=1.02)

# y축 포맷팅
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/3_business_analysis_report.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n** 의사결정 인사이트 **")
for district in selected_districts:
    for industry in selected_industries:
        subset = df_analysis[(df_analysis['행정구역명'] == district) & 
                            (df_analysis['서비스업종코드명'] == industry)]
        avg_sales = subset['월매출금액'].mean()
        print(f"{district} - {industry}: 평균 {avg_sales/1000000:.1f}M원")
```

---

## 8. 학습 정리

### 8.1 핵심 개념

1. **Seaborn의 장점**: 간결한 코드, 아름다운 디자인, 자동 통계 계산
2. **Barplot**: 카테고리별 평균 + 신뢰구간
3. **Countplot**: 카테고리별 빈도
4. **Boxplot/Violinplot**: 분포 비교
5. **Pointplot**: 변화 추세 비교
6. **스타일과 팔레트**: 그래프의 전반적인 디자인 조정

### 8.2 실무 활용 팁

- **보고서용**: whitegrid 스타일 + muted/pastel 팔레트 (깔끔)
- **발표용**: white 스타일 + bright/deep 팔레트 (강렬)
- **신뢰구간**: 데이터의 신뢰성을 보여주는 중요한 지표
- **Catplot**: 복잡한 다중 비교를 한 번에 (FacetGrid 기능)

---

## 9. 학습 확인 퀴즈

### 문제 1
Seaborn의 가장 큰 장점은 무엇인가?
1. Matplotlib보다 빠르다
2. 통계 계산을 자동으로 수행하고 간결한 코드로 아름다운 그래프를 만든다
3. 3D 그래프를 쉽게 그릴 수 있다
4. 애니메이션을 만들 수 있다

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 통계 계산을 자동으로 수행하고 간결한 코드로 아름다운 그래프를 만든다**

**해설:** 
Seaborn은 Matplotlib을 기반으로 하며, 평균, 신뢰구간 등의 통계 계산을 자동으로 수행합니다. 또한 현대적이고 아름다운 기본 스타일을 제공하여 보고서나 발표 자료에 바로 사용할 수 있습니다.
</details>

---

### 문제 2
Barplot에서 검은 선은 무엇을 의미하는가?
1. 표준편차
2. 최댓값과 최솟값
3. 95% 신뢰구간
4. 중앙값

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 3. 95% 신뢰구간**

**해설:** 
Seaborn의 barplot에서 막대 위의 검은 선은 기본적으로 95% 신뢰구간(confidence interval)을 나타냅니다. 이는 평균값의 불확실성을 보여주며, 선이 짧을수록 데이터가 일관적이고 신뢰할 만합니다.
</details>

---

### 문제 3
Boxplot에서 상자 안의 선은 무엇을 나타내는가?
1. 평균
2. 중앙값 (50%)
3. 최빈값
4. 표준편차

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 중앙값 (50%)**

**해설:** 
Boxplot의 상자 안에 있는 선은 중앙값(median)을 나타냅니다.
- 상자의 아래: 25% (1사분위수)
- 상자의 선: 50% (중앙값)
- 상자의 위: 75% (3사분위수)
- 평균은 별도로 표시되지 않습니다.
</details>

---

### 문제 4
Violinplot이 Boxplot보다 더 많은 정보를 제공하는 이유는?
1. 색상이 더 다양하다
2. 데이터의 분포 형태(밀도)를 시각화한다
3. 더 많은 이상치를 표시한다
4. 평균과 중앙값을 동시에 표시한다

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 데이터의 분포 형태(밀도)를 시각화한다**

**해설:** 
Violinplot은 Boxplot에 분포의 밀도(density)를 추가로 보여줍니다. 바이올린의 넓은 부분은 해당 값에 데이터가 많이 분포함을 의미하며, 좁은 부분은 데이터가 적음을 의미합니다. 이를 통해 단봉 분포인지, 쌍봉 분포인지 등 분포의 형태를 파악할 수 있습니다.
</details>

---

### 문제 5
Seaborn에서 그래프의 전반적인 디자인을 변경하려면 어떤 함수를 사용하는가?
1. `sns.set_style()`
2. `sns.change_design()`
3. `sns.set_config()`
4. `sns.configure()`

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 1. `sns.set_style()`**

**해설:** 
`sns.set_style()`을 사용하여 그래프의 전반적인 스타일을 변경할 수 있습니다.
- darkgrid: 어두운 배경 + 그리드
- whitegrid: 밝은 배경 + 그리드
- dark: 어두운 배경, 그리드 없음
- white: 밝은 배경, 그리드 없음
- ticks: 축의 눈금만 표시

추가로 `sns.set_palette()`로 색상 팔레트를 변경할 수 있습니다.
</details>

---

## 다음 교시 예고

**4교시: Box Plot - 업종/지역별 매출 분포 비교** (11/6 목)
- Boxplot의 심화 활용
- 다중 그룹 비교
- 이상치 탐지 및 분석
- 분포의 통계적 비교
