# 2교시: 히스토그램과 산점도 - 데이터 분포와 관계 분석

**학습 목표:**
- 히스토그램으로 데이터 분포 파악
- 산점도로 변수 간 관계 탐색
- 실무 의사결정에 활용 가능한 인사이트 도출
- 연령대별 소비 패턴과 매출-점포수 관계 분석

**예상 소요 시간:** 60분

---

## 1. 히스토그램 (Histogram): 데이터 분포 파악

### 1.1 개념 설명

**히스토그램**은 연속형 데이터의 분포를 시각화하는 그래프입니다.
- **용도**: 매출 분포, 나이 분포, 가격대 분포 등
- **실무 활용**: "대부분의 매장이 어느 정도 매출을 올리는가?"
- **비유**: 학생들의 시험 점수 분포를 구간별로 나눈 것과 같습니다.

**히스토그램 vs 바 차트:**
- 히스토그램: 연속형 데이터 (매출, 나이, 가격 등)
- 바 차트: 범주형 데이터 (업종, 지역, 상품명 등)

### 1.2 실습: 월매출 분포 확인

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("data/서울시_상권_추정매출.csv", encoding="cp949")

# 월매출금액 분포 히스토그램
fig, ax = plt.subplots(figsize=(12, 6))

# 히스토그램 그리기
n, bins, patches = ax.hist(df['월매출금액'], 
                           bins=30,  # 구간 개수
                           color='#457B9D', 
                           alpha=0.7,
                           edgecolor='black')

# 평균선 추가
mean_value = df['월매출금액'].mean()
ax.axvline(mean_value, color='red', linestyle='--', linewidth=2, 
           label=f'평균: {mean_value/1000000:.1f}M원')

# 중앙값 추가
median_value = df['월매출금액'].median()
ax.axvline(median_value, color='green', linestyle='--', linewidth=2,
           label=f'중앙값: {median_value/1000000:.1f}M원')

# 그래프 꾸미기
ax.set_title('서울시 상권 월매출금액 분포', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('월매출금액 (원)', fontsize=12)
ax.set_ylabel('빈도 (개수)', fontsize=12)
ax.legend(fontsize=11)

# x축 포맷팅
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

# 그리드
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/2_revenue_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 기술통계 출력
print("=" * 60)
print("월매출금액 기술통계")
print("=" * 60)
print(f"평균: {df['월매출금액'].mean()/1000000:.2f}M원")
print(f"중앙값: {df['월매출금액'].median()/1000000:.2f}M원")
print(f"최솟값: {df['월매출금액'].min()/1000000:.2f}M원")
print(f"최댓값: {df['월매출금액'].max()/1000000:.2f}M원")
print(f"표준편차: {df['월매출금액'].std()/1000000:.2f}M원")
```

### 1.3 결과 해석

**실무 인사이트:**
- **분포 형태**: 
  - 오른쪽으로 치우침 (right-skewed): 대부분의 매장은 낮은 매출, 소수가 매우 높은 매출
  - 이는 상권의 양극화를 의미할 수 있음
  
- **평균 vs 중앙값**:
  - 평균 > 중앙값: 소수의 고매출 매장이 평균을 끌어올림
  - 중앙값이 실제 "전형적인" 매출 수준을 더 잘 반영
  
- **비즈니스 활용**:
  - 신규 입점 시 현실적인 매출 목표 설정
  - 상위 25% 매장의 특징 분석 필요

---

## 2. 업종별 매출 분포 비교

### 2.1 실습: 여러 업종의 분포를 한눈에

```python
# 주요 업종 3개 선택
industries = ['커피/음료', '한식음식점', '편의점']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, industry in enumerate(industries):
    industry_data = df[df['서비스업종코드명'] == industry]['월매출금액']
    
    # 히스토그램
    axes[idx].hist(industry_data, bins=20, color='#A8DADC', 
                  alpha=0.7, edgecolor='black')
    
    # 평균선
    mean_val = industry_data.mean()
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                     label=f'평균: {mean_val/1000000:.1f}M')
    
    # 꾸미기
    axes[idx].set_title(f'{industry}\n월매출 분포', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('월매출금액 (원)', fontsize=11)
    axes[idx].set_ylabel('빈도', fontsize=11)
    axes[idx].legend()
    axes[idx].grid(True, axis='y', alpha=0.3)
    axes[idx].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

plt.tight_layout()
plt.savefig('output/2_industry_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 각 업종 통계 비교
print("\n" + "=" * 60)
print("업종별 매출 통계 비교")
print("=" * 60)
for industry in industries:
    industry_data = df[df['서비스업종코드명'] == industry]['월매출금액']
    print(f"\n[{industry}]")
    print(f"  평균: {industry_data.mean()/1000000:.2f}M원")
    print(f"  중앙값: {industry_data.median()/1000000:.2f}M원")
    print(f"  표준편차: {industry_data.std()/1000000:.2f}M원")
```

### 2.2 결과 해석

**실무 인사이트:**
- 업종마다 매출 분포의 형태가 다름
- 표준편차가 큰 업종: 성공 여부의 변동성이 크다 (high risk, high return)
- 표준편차가 작은 업종: 안정적이지만 대박 가능성도 낮음

---

## 3. 산점도 (Scatter Plot): 변수 간 관계 탐색

### 3.1 개념 설명

**산점도**는 두 연속형 변수 간의 관계를 시각화합니다.
- **용도**: 매출과 점포수, 나이와 소득, 광고비와 매출 등
- **실무 활용**: "점포가 많을수록 매출이 높은가?"
- **관계 유형**:
  - 양의 상관: 한 변수가 증가하면 다른 변수도 증가
  - 음의 상관: 한 변수가 증가하면 다른 변수는 감소
  - 무상관: 두 변수 간 관계 없음

### 3.2 실습: 점포수와 월매출의 관계

```python
# 상권별로 집계 (점포수와 평균 매출)
commercial_zone_data = df.groupby('상권코드명').agg({
    '점포수': 'first',  # 상권의 점포수
    '월매출금액': 'mean'  # 평균 월매출
}).reset_index()

# 산점도 그리기
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(commercial_zone_data['점포수'], 
                    commercial_zone_data['월매출금액'],
                    alpha=0.6,  # 투명도
                    s=100,  # 점 크기
                    c=commercial_zone_data['월매출금액'],  # 색상 매핑
                    cmap='YlOrRd',  # 색상 맵
                    edgecolors='black',
                    linewidths=0.5)

# 추세선 추가 (선형 회귀)
from numpy.polynomial import Polynomial
x = commercial_zone_data['점포수']
y = commercial_zone_data['월매출금액']
p = Polynomial.fit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = p(x_line)
ax.plot(x_line, y_line, 'r--', linewidth=2, label='추세선')

# 컬러바 추가
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('월매출금액 (원)', fontsize=11)

# 그래프 꾸미기
ax.set_title('상권 점포수와 평균 월매출의 관계', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('점포수 (개)', fontsize=12)
ax.set_ylabel('평균 월매출금액 (원)', fontsize=12)
ax.legend(fontsize=11)

# y축 포맷팅
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))

# 그리드
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/2_stores_vs_revenue.png', dpi=300, bbox_inches='tight')
plt.show()

# 상관계수 계산
correlation = commercial_zone_data['점포수'].corr(commercial_zone_data['월매출금액'])
print(f"\n점포수와 월매출의 상관계수: {correlation:.3f}")
```

### 3.3 결과 해석

**실무 인사이트:**
- **상관계수**:
  - 0.7 ~ 1.0: 강한 양의 상관
  - 0.3 ~ 0.7: 중간 양의 상관
  - -0.3 ~ 0.3: 약한 상관 또는 무상관
  
- **해석 예시**:
  - 점포수가 많은 상권 = 유동인구가 많고 매출도 높음
  - 하지만 점포수가 너무 많으면 경쟁 심화로 개별 매출은 감소할 수도
  
- **비즈니스 활용**:
  - 적정 점포수 구간 파악
  - "이미 포화된 상권인가?" 판단

---

## 4. 연령대별 매출 비중 분석

### 4.1 실습: 업종별 주요 고객층 파악

```python
# 연령대 컬럼
age_columns = ['10대매출금액', '20대매출금액', '30대매출금액', 
               '40대매출금액', '50대매출금액', '60대이상매출금액']

# 업종 선택
industries_for_age = ['커피/음료', '주점', '학원']

# 각 업종별 연령대 매출 비중 계산
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, industry in enumerate(industries_for_age):
    industry_data = df[df['서비스업종코드명'] == industry]
    
    # 연령대별 평균 매출
    age_sales = [industry_data[col].mean() for col in age_columns]
    age_labels = ['10대', '20대', '30대', '40대', '50대', '60대+']
    
    # 바 차트
    bars = axes[idx].bar(age_labels, age_sales, 
                        color=['#E76F51', '#F4A261', '#E9C46A', 
                               '#2A9D8F', '#264653', '#8AB17D'])
    
    # 꾸미기
    axes[idx].set_title(f'{industry}\n연령대별 평균 매출', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('평균 매출금액 (원)', fontsize=11)
    axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
    axes[idx].grid(True, axis='y', alpha=0.3)
    
    # 최고 매출 연령대 강조
    max_idx = np.argmax(age_sales)
    bars[max_idx].set_color('#E63946')
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(2)

plt.tight_layout()
plt.savefig('output/2_age_revenue_by_industry.png', dpi=300, bbox_inches='tight')
plt.show()

# 주요 고객층 출력
print("\n" + "=" * 60)
print("업종별 주요 고객층 (연령대)")
print("=" * 60)
for industry in industries_for_age:
    industry_data = df[df['서비스업종코드명'] == industry]
    age_sales = [industry_data[col].mean() for col in age_columns]
    age_labels = ['10대', '20대', '30대', '40대', '50대', '60대+']
    max_idx = np.argmax(age_sales)
    print(f"{industry}: {age_labels[max_idx]} (평균 {age_sales[max_idx]/1000000:.1f}M원)")
```

### 4.2 결과 해석

**실무 인사이트:**
- 커피/음료: 20-30대 주 고객 → 젊은 층 타겟 마케팅
- 주점: 30-40대 주 고객 → 직장인 대상 프로모션
- 학원: 10-20대 주 고객 → 학생 및 부모 타겟

---

## 5. 고급: 지역과 매출의 관계 (다중 산점도)

### 5.1 실습: 구별 매출과 점포수 관계

```python
# 주요 구 선택
selected_districts = ['강남구', '마포구', '종로구', '송파구', '영등포구']

# 구별 데이터 집계
district_summary = df[df['행정구역명'].isin(selected_districts)].groupby(['행정구역명', '상권코드명']).agg({
    '점포수': 'first',
    '월매출금액': 'mean'
}).reset_index()

# 산점도 (구별로 색상 구분)
fig, ax = plt.subplots(figsize=(12, 8))

colors_map = {'강남구': '#E63946', '마포구': '#F1FAEE', '종로구': '#A8DADC',
              '송파구': '#457B9D', '영등포구': '#1D3557'}

for district in selected_districts:
    district_data = district_summary[district_summary['행정구역명'] == district]
    ax.scatter(district_data['점포수'], 
              district_data['월매출금액'],
              label=district,
              alpha=0.6,
              s=100,
              color=colors_map[district],
              edgecolors='black',
              linewidths=0.5)

# 그래프 꾸미기
ax.set_title('구별 점포수와 평균 월매출 관계', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('점포수 (개)', fontsize=12)
ax.set_ylabel('평균 월매출금액 (원)', fontsize=12)
ax.legend(title='행정구역', fontsize=11, title_fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000)}M'))
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/2_district_stores_revenue.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.2 결과 해석

**실무 인사이트:**
- 구마다 점포수-매출 관계가 다를 수 있음
- 강남구: 점포수 대비 높은 매출 (프리미엄 상권)
- 기타 지역: 점포수와 매출이 비례

---

## 6. 실무 적용 예시

### 시나리오: 신규 카페 타겟 고객층 분석

```python
# 커피/음료 업종의 연령대별 매출 vs 지역
coffee_data = df[df['서비스업종코드명'] == '커피/음료']

# 주요 3개 구 선택
top_districts = coffee_data.groupby('행정구역명')['월매출금액'].mean().nlargest(3).index

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, district in enumerate(top_districts):
    district_coffee = coffee_data[coffee_data['행정구역명'] == district]
    
    age_sales = [district_coffee[col].mean() for col in age_columns]
    age_labels = ['10대', '20대', '30대', '40대', '50대', '60대+']
    
    axes[idx].bar(age_labels, age_sales, color='#457B9D', alpha=0.7)
    axes[idx].set_title(f'{district} 커피/음료\n연령대별 평균 매출', fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('평균 매출금액 (원)', fontsize=11)
    axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
    axes[idx].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/2_cafe_target_by_district.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n** 타겟 고객층 분석 결과 **")
for district in top_districts:
    district_coffee = coffee_data[coffee_data['행정구역명'] == district]
    age_sales = [district_coffee[col].mean() for col in age_columns]
    age_labels = ['10대', '20대', '30대', '40대', '50대', '60대+']
    max_idx = np.argmax(age_sales)
    print(f"{district}: 주 고객층 {age_labels[max_idx]} (비중 {age_sales[max_idx]/sum(age_sales)*100:.1f}%)")
```

---

## 7. 학습 정리

### 7.1 핵심 개념

1. **히스토그램**: 데이터 분포 파악 (평균, 중앙값, 분산)
2. **산점도**: 두 변수 간 관계 탐색 (상관관계)
3. **추세선**: 관계의 방향과 강도 시각화
4. **다중 비교**: 여러 그룹의 분포나 관계를 동시에 비교

### 7.2 실무 활용 팁

- **히스토그램**: 평균과 중앙값을 함께 표시하여 분포의 치우침 파악
- **산점도**: 색상, 크기를 추가 변수로 활용 (3-4차원 정보 표현)
- **이상치**: 산점도에서 멀리 떨어진 점들은 특별한 케이스 → 별도 분석 필요

---

## 8. 학습 확인 퀴즈

### 문제 1
히스토그램에서 평균이 중앙값보다 훨씬 크다면, 이는 무엇을 의미하는가?
1. 데이터가 정규분포를 따른다
2. 데이터가 왼쪽으로 치우쳐 있다
3. 데이터가 오른쪽으로 치우쳐 있다
4. 이상치가 없다

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 3. 데이터가 오른쪽으로 치우쳐 있다**

**해설:** 
평균 > 중앙값인 경우, 오른쪽 꼬리가 긴 분포(right-skewed)를 의미합니다. 즉, 대부분의 데이터는 낮은 값에 몰려 있고, 소수의 매우 큰 값들이 평균을 끌어올립니다. 매출 데이터에서 흔히 나타나는 패턴입니다.
</details>

---

### 문제 2
산점도를 사용하기에 가장 적합한 경우는?
1. 업종별 매출 비교
2. 월별 매출 추세 분석
3. 광고비와 매출의 관계 분석
4. 연령대별 매출 분포

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 3. 광고비와 매출의 관계 분석**

**해설:** 
산점도는 두 연속형 변수 간의 관계를 탐색할 때 사용합니다. 광고비와 매출 모두 연속형 변수이며, 둘 사이의 상관관계를 파악하기에 적합합니다.
- 1번: 바 차트
- 2번: 라인 차트
- 4번: 히스토그램 또는 박스 플롯
</details>

---

### 문제 3
상관계수가 -0.8이라면, 이는 무엇을 의미하는가?
1. 두 변수 간에 관계가 없다
2. 한 변수가 증가하면 다른 변수도 증가한다
3. 한 변수가 증가하면 다른 변수는 감소한다
4. 두 변수가 완전히 같다

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 3. 한 변수가 증가하면 다른 변수는 감소한다**

**해설:** 
상관계수가 음수(-)이면 음의 상관관계를 의미합니다. -0.8은 강한 음의 상관으로, 한 변수가 증가할 때 다른 변수는 감소하는 경향이 뚜렷합니다.
예: 가격이 올라갈수록 판매량이 줄어드는 경우
</details>

---

### 문제 4
히스토그램의 `bins` 매개변수는 무엇을 의미하는가?
1. 데이터의 개수
2. 구간(bin)의 개수
3. 그래프의 색상
4. 막대의 두께

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 구간(bin)의 개수**

**해설:** 
`bins`는 데이터를 몇 개의 구간으로 나눌지를 결정합니다. 
- bins=10: 데이터를 10개 구간으로 나눔 (넓은 구간, 전체적인 패턴 파악)
- bins=50: 데이터를 50개 구간으로 나눔 (좁은 구간, 세밀한 분포 파악)
적절한 bins 수를 선택하는 것이 중요합니다.
</details>

---

### 문제 5
산점도에서 추세선을 추가하는 이유는?
1. 그래프를 예쁘게 만들기 위해
2. 두 변수 간 관계의 방향과 강도를 시각적으로 보여주기 위해
3. 데이터 포인트를 연결하기 위해
4. 이상치를 제거하기 위해

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 두 변수 간 관계의 방향과 강도를 시각적으로 보여주기 위해**

**해설:** 
추세선(trendline)은 데이터의 전반적인 패턴을 나타냅니다. 
- 양의 기울기: 양의 상관관계
- 음의 기울기: 음의 상관관계
- 수평선: 상관관계 없음
추세선을 통해 복잡한 산점도에서도 관계를 쉽게 파악할 수 있습니다.
</details>

---

## 다음 교시 예고

**3교시: Seaborn - 미적으로 우수한 시각화**
- Seaborn의 기본 개념과 장점
- 카테고리별 비교 (barplot, countplot)
- 분포 비교 (violinplot, stripplot)
- 스타일과 테마 설정
