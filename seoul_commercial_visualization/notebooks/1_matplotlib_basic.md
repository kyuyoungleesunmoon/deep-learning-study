# 1교시: Matplotlib 기본 - 라인 차트와 바 차트

**학습 목표:**
- Matplotlib의 기본 구조 이해
- 라인 차트로 시계열 데이터 시각화
- 바 차트로 카테고리별 비교 수행
- 실무 의사결정에 활용 가능한 매출 추세 및 업종 비교 그래프 작성

**예상 소요 시간:** 60분

---

## 1. Matplotlib 소개

### 1.1 Matplotlib란?

**Matplotlib**은 Python에서 가장 널리 사용되는 시각화 라이브러리입니다. 
- **장점**: 세밀한 제어 가능, 출판 품질의 그래프 생성
- **단점**: 초기 학습 곡선이 다소 있음 (하지만 기본만 알면 충분!)

### 1.2 기본 구조

```python
import matplotlib.pyplot as plt

# Figure: 전체 그림 영역
# Axes: 실제 그래프가 그려지는 영역

fig, ax = plt.subplots()  # Figure와 Axes 생성
ax.plot([1, 2, 3], [1, 4, 9])  # 그래프 그리기
plt.show()  # 화면에 표시
```

**비유**: Figure는 액자, Axes는 그림이 그려지는 캔버스입니다.

---

## 2. 데이터 준비

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 서울시 상권 데이터 불러오기
df = pd.read_csv("data/서울시_상권_추정매출.csv", encoding="cp949")

# 데이터 확인
print(df.head())
print(f"\n데이터 크기: {df.shape}")
print(f"\n컬럼 목록:\n{df.columns.tolist()}")
```

---

## 3. 라인 차트 (Line Chart): 월별 매출 추세 분석

### 3.1 개념 설명

**라인 차트**는 시간에 따른 변화를 시각화할 때 가장 효과적입니다.
- **용도**: 매출 추세, 방문자 수 변화, 재고 변동 등
- **실무 활용**: "우리 매장의 매출이 증가하는가, 감소하는가?"

### 3.2 실습: 특정 업종의 월별 매출 추세

```python
# 특정 업종 (예: 커피/음료) 선택
coffee_data = df[df['서비스업종코드명'] == '커피/음료'].copy()

# 기준년월코드별로 평균 월매출 계산
monthly_sales = coffee_data.groupby('기준년월코드')['월매출금액'].mean().sort_index()

# 라인 차트 그리기
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(monthly_sales.index, monthly_sales.values, 
        marker='o',  # 데이터 포인트 표시
        linewidth=2,  # 선 두께
        color='#2E86AB',  # 선 색상
        markersize=8)  # 마커 크기

# 그래프 꾸미기
ax.set_title('커피/음료 업종 월별 평균 매출 추세', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('기준년월', fontsize=12)
ax.set_ylabel('평균 매출금액 (원)', fontsize=12)

# y축 포맷팅 (천 단위 구분)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# 그리드 추가 (가독성 향상)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/1_coffee_monthly_trend.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.3 결과 해석

**실무 인사이트:**
- 그래프에서 상승/하락 추세를 확인
- 특정 월에 급증/급락이 있다면 원인 분석 필요 (프로모션? 경쟁사 진입?)
- 계절성 패턴이 있는지 확인

---

## 4. 바 차트 (Bar Chart): 업종별 매출 비교

### 4.1 개념 설명

**바 차트**는 카테고리 간 비교에 가장 적합합니다.
- **용도**: 지역별 매출, 상품별 판매량, 부서별 성과 등
- **실무 활용**: "어떤 업종이 가장 매출이 높은가?"

### 4.2 실습: 상위 10개 업종 매출 비교

```python
# 업종별 평균 월매출 계산
industry_sales = df.groupby('서비스업종코드명')['월매출금액'].mean().sort_values(ascending=False)

# 상위 10개 업종 선택
top10_industries = industry_sales.head(10)

# 수평 바 차트 그리기 (긴 이름은 수평이 더 읽기 좋음)
fig, ax = plt.subplots(figsize=(10, 8))

# 색상 그라데이션 (가장 높은 것을 강조)
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top10_industries)))

bars = ax.barh(range(len(top10_industries)), top10_industries.values, color=colors)

# y축 레이블 설정
ax.set_yticks(range(len(top10_industries)))
ax.set_yticklabels(top10_industries.index)

# 그래프 꾸미기
ax.set_title('업종별 평균 월매출 Top 10', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('평균 월매출금액 (원)', fontsize=12)

# x축 포맷팅
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))

# 각 막대 끝에 정확한 값 표시
for i, (bar, value) in enumerate(zip(bars, top10_industries.values)):
    ax.text(value, i, f' {value/1000000:.1f}M원', 
            va='center', fontsize=10)

# 그리드 추가
ax.grid(True, axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/1_top10_industries.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4.3 결과 해석

**실무 인사이트:**
- 어떤 업종이 높은 매출을 기록하는가?
- 내가 진출하려는 업종의 평균 매출 수준은?
- 경쟁이 심한 업종 vs 블루오션 파악

---

## 5. 고급: 서브플롯 (여러 그래프를 한 번에)

### 5.1 개념

**서브플롯**을 사용하면 여러 그래프를 한 화면에 배치하여 비교할 수 있습니다.

### 5.2 실습: 요일별 매출 패턴 비교

```python
# 요일별 매출 컬럼
weekday_columns = ['월요일매출금액', '화요일매출금액', '수요일매출금액', 
                   '목요일매출금액', '금요일매출금액', '토요일매출금액', '일요일매출금액']

# 두 개 업종 선택
industries_to_compare = ['커피/음료', '주점']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for idx, industry in enumerate(industries_to_compare):
    # 해당 업종 데이터 필터링
    industry_data = df[df['서비스업종코드명'] == industry]
    
    # 요일별 평균 매출 계산
    weekday_sales = [industry_data[col].mean() for col in weekday_columns]
    weekdays = ['월', '화', '수', '목', '금', '토', '일']
    
    # 바 차트 그리기
    bars = axes[idx].bar(weekdays, weekday_sales, 
                        color=['#A8DADC' if i < 5 else '#457B9D' for i in range(7)])
    
    # 그래프 꾸미기
    axes[idx].set_title(f'{industry} 요일별 평균 매출', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('평균 매출금액 (원)', fontsize=11)
    axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))
    
    # 그리드
    axes[idx].grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 각 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height/1000000:.1f}M',
                      ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('output/1_weekday_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.3 결과 해석

**실무 인사이트:**
- 커피/음료: 평일과 주말의 매출 차이가 적음 (일상 소비)
- 주점: 주말(금,토,일)에 매출이 급증 (여가 소비)
- 이를 통해 인력 배치, 재고 관리 전략 수립 가능

---

## 6. 실무 적용 예시

### 시나리오: 신규 카페 입점 지역 선정

```python
# 강남구, 마포구, 종로구 3곳의 커피/음료 매출 비교
selected_districts = ['강남구', '마포구', '종로구']
coffee_data = df[df['서비스업종코드명'] == '커피/음료']

district_sales = coffee_data[coffee_data['행정구역명'].isin(selected_districts)].groupby('행정구역명')['월매출금액'].mean().sort_values(ascending=False)

# 바 차트
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(district_sales.index, district_sales.values, 
              color=['#E63946', '#F1FAEE', '#A8DADC'])

# 꾸미기
ax.set_title('지역별 커피/음료 평균 월매출 비교\n(신규 입점 후보지 분석)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('평균 월매출금액 (원)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000000):.1f}M'))

# 값 표시
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height/1000000:.1f}M원',
           ha='center', va='bottom', fontsize=12, fontweight='bold')

# 그리드
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('output/1_district_cafe_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n** 의사결정 인사이트 **")
print(f"가장 높은 매출 지역: {district_sales.index[0]} ({district_sales.values[0]/1000000:.1f}M원)")
print(f"→ 높은 매출 지역은 경쟁도 치열할 수 있으므로 임대료, 유동인구 등 추가 분석 필요")
```

---

## 7. 학습 정리

### 7.1 핵심 개념

1. **라인 차트**: 시간에 따른 변화 추세 파악 (월별 매출, 성장률 등)
2. **바 차트**: 카테고리 간 크기 비교 (업종별, 지역별 비교)
3. **Figure와 Axes**: Matplotlib의 기본 구조
4. **서브플롯**: 여러 그래프를 한 화면에 배치

### 7.2 실무 활용 팁

- **색상**: 강조하고 싶은 부분을 다른 색으로
- **레이블**: 정확한 수치를 막대/점 위에 표시
- **제목**: 그래프의 목적을 명확히 (예: "신규 입점 후보지 분석")
- **저장**: `plt.savefig()`로 보고서에 바로 사용

---

## 8. 학습 확인 퀴즈

### 문제 1
Matplotlib에서 그래프를 그리기 위한 기본 영역을 무엇이라고 하는가?
1. Figure
2. Axes
3. Canvas
4. Plot

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. Axes**

**해설:** 
- Figure는 전체 그림 영역 (액자)
- Axes는 실제 그래프가 그려지는 영역 (캔버스)
- `fig, ax = plt.subplots()`에서 ax가 Axes 객체입니다.
</details>

---

### 문제 2
시간에 따른 매출 추세를 시각화하기에 가장 적합한 그래프는?
1. 바 차트 (Bar Chart)
2. 라인 차트 (Line Chart)
3. 파이 차트 (Pie Chart)
4. 박스 플롯 (Box Plot)

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 라인 차트 (Line Chart)**

**해설:** 
라인 차트는 시계열 데이터의 변화를 시각화하는 데 가장 효과적입니다. 선의 기울기를 통해 증가/감소 추세를 직관적으로 파악할 수 있습니다.
</details>

---

### 문제 3
다음 중 바 차트를 사용하기 적합한 경우는?
1. 월별 매출 추세 분석
2. 업종별 매출 크기 비교
3. 변수 간 상관관계 분석
4. 매출 분포 확인

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. 업종별 매출 크기 비교**

**해설:** 
바 차트는 카테고리(업종, 지역, 상품 등) 간의 크기를 비교할 때 가장 효과적입니다. 막대의 길이를 통해 쉽게 비교할 수 있습니다.
- 1번은 라인 차트
- 3번은 산점도 또는 히트맵
- 4번은 히스토그램 또는 박스 플롯
</details>

---

### 문제 4
여러 그래프를 한 화면에 배치하려면 어떤 기능을 사용하는가?
1. subplot()
2. subplots()
3. plot_multi()
4. grid_plot()

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. subplots()**

**해설:** 
`plt.subplots(nrows, ncols)`를 사용하면 여러 개의 Axes를 한 번에 생성할 수 있습니다.
예: `fig, axes = plt.subplots(2, 2)`는 2x2 그리드의 4개 그래프를 생성합니다.
</details>

---

### 문제 5
한글이 깨지는 문제를 해결하기 위한 설정은?
1. `plt.rcParams['font.size'] = 12`
2. `plt.rcParams['font.family'] = 'Malgun Gothic'`
3. `plt.rcParams['figure.dpi'] = 300`
4. `plt.rcParams['savefig.format'] = 'png'`

<details>
<summary>정답 및 해설 (클릭하여 보기)</summary>

**정답: 2. `plt.rcParams['font.family'] = 'Malgun Gothic'`**

**해설:** 
Matplotlib은 기본적으로 한글을 지원하지 않으므로, 한글 폰트를 지정해야 합니다.
- Windows: 'Malgun Gothic' (맑은 고딕)
- Mac: 'AppleGothic'
- Linux: 'NanumGothic'

추가로 `plt.rcParams['axes.unicode_minus'] = False`를 설정하여 마이너스 기호 깨짐도 방지합니다.
</details>

---

## 다음 교시 예고

**2교시: 히스토그램과 산점도 - 연령/상권/지표 간 관계 시각화**
- 데이터의 분포를 파악하는 히스토그램
- 변수 간 관계를 탐색하는 산점도
- 실무 예시: 연령대별 소비 패턴, 매출과 점포수의 관계 등
