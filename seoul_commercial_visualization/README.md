# 서울시 상권분석 데이터 시각화 실무 교육

**재직자를 위한 3주 완성 실전 데이터 시각화 과정**

실제 서울시 상권분석 데이터를 활용하여 Python 데이터 시각화 기법을 배우고, 실무에 바로 적용 가능한 인사이트 도출 능력을 키우는 교육 과정입니다.

---

## 📋 과정 개요

### 교육 대상
- Python 기초는 있지만 실무 데이터 분석 경험이 없는 재직자
- 현업에서 즉시 그래프와 보고서를 만들고 싶은 직장인
- 데이터 기반 의사결정 역량을 키우고 싶은 실무자

### 학습 목표
- Matplotlib, Seaborn, Folium을 활용한 다양한 시각화 기법 습득
- 실무 데이터 분석 프로세스 이해 (가설 수립 → EDA → 시각화 → 인사이트 도출)
- 비즈니스 의사결정을 지원하는 보고서 작성 능력 향상
- 지리공간 데이터 분석 및 시각화 역량 확보

### 데이터 소스
- **서울시 상권분석서비스 (추정매출-상권) 데이터**
- 출처: [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do)
- 포함 정보: 지역별/업종별 매출, 요일별 매출, 연령대별 매출, 점포수, 위치 정보 등

---

## 📅 커리큘럼

### Week 1, Day 1 (11/4 화) - Matplotlib 기초
| 교시 | 주제 | 내용 |
|------|------|------|
| 1교시 | [Matplotlib 기본](./notebooks/1_matplotlib_basic.md) | 라인 차트, 바 차트로 월별 매출 추세 및 업종 비교 |
| 2교시 | [히스토그램과 산점도](./notebooks/2_histogram_scatter.md) | 데이터 분포 파악, 변수 간 관계 분석 |
| 3교시 | [Seaborn 소개](./notebooks/3_seaborn_intro.md) | 카테고리별 비교, 통계적 시각화 |

### Week 1, Day 2 (11/6 목) - Seaborn & Folium
| 교시 | 주제 | 내용 |
|------|------|------|
| 4교시 | [Box Plot 심화](./notebooks/4_boxplot_deep_dive.md) | 업종/지역별 매출 분포 비교, 이상치 분석 |
| 5교시 | [Heatmap](./notebooks/5_heatmap_correlation.md) | 변수 간 상관관계, 다차원 데이터 시각화 |
| 6교시 | [Folium 지도 시각화](./notebooks/6_folium_mapping.md) | 지리공간 데이터, 마커 클러스터링, 히트맵 |

### Week 1, Day 3 (11/8 토) - 종합 프로젝트
| 교시 | 주제 | 내용 |
|------|------|------|
| 1~2교시 | [프로젝트 기획](./notebooks/7_final_project.md) | 가설 수립, 데이터 전처리 |
| 3~4교시 | EDA 및 시각화 | 탐색적 데이터 분석, 핵심 그래프 제작 |
| 5~6교시 | 보고서 작성 | 인사이트 도출, 발표 자료 준비 |
| 7교시 | 발표 | 프로젝트 발표 및 피드백 |

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/kyuyoungleesunmoon/deep-learning-study.git
cd deep-learning-study/seoul_commercial_visualization

# 필요한 패키지 설치
pip install -r ../requirements.txt

# 또는 개별 설치
pip install pandas numpy matplotlib seaborn folium
```

### 2. 샘플 데이터 생성

```bash
# 샘플 데이터 자동 생성
python prepare_data.py
```

실행 후 `data/서울시_상권_추정매출.csv` 파일이 생성됩니다.

**실제 데이터 사용:**
- [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do)에서 최신 데이터 다운로드
- `data/` 폴더에 저장 후 코드에서 파일명 수정

### 3. 학습 시작

교재(Markdown 파일)를 순서대로 읽으며 학습합니다:
1. `notebooks/1_matplotlib_basic.md` 부터 시작
2. 각 장의 코드를 Python 스크립트나 Jupyter Notebook에서 실행
3. 퀴즈로 학습 내용 확인

---

## 📂 프로젝트 구조

```
seoul_commercial_visualization/
│
├── README.md                 # 이 파일
├── prepare_data.py           # 샘플 데이터 생성 스크립트
│
├── data/                     # 데이터 폴더
│   └── 서울시_상권_추정매출.csv
│
├── notebooks/                # 교육 자료 (Markdown)
│   ├── 1_matplotlib_basic.md
│   ├── 2_histogram_scatter.md
│   ├── 3_seaborn_intro.md
│   ├── 4_boxplot_deep_dive.md
│   ├── 5_heatmap_correlation.md
│   ├── 6_folium_mapping.md
│   └── 7_final_project.md
│
└── output/                   # 생성된 그래프 저장 폴더
    ├── 1_coffee_monthly_trend.png
    ├── 2_revenue_distribution.png
    └── ...
```

---

## 💡 주요 학습 내용

### 1. Matplotlib 기초
- Figure와 Axes 구조 이해
- 라인 차트: 시계열 데이터 추세 분석
- 바 차트: 카테고리별 크기 비교
- 히스토그램: 데이터 분포 파악
- 산점도: 변수 간 관계 탐색

### 2. Seaborn 고급
- 통계적 시각화: barplot, boxplot, violinplot
- Heatmap: 상관관계 분석
- Pairplot, Jointplot: 다변량 분석
- 스타일과 팔레트 커스터마이징

### 3. Folium 지도 시각화
- 인터랙티브 지도 생성
- 마커와 팝업으로 정보 표시
- 마커 클러스터링으로 밀집도 시각화
- 히트맵으로 패턴 표현
- CircleMarker로 값의 크기 표현

### 4. 실무 프로젝트
- 비즈니스 문제 정의 및 가설 수립
- 데이터 전처리 및 EDA
- 핵심 시각화 제작
- 인사이트 도출 및 의사결정 지원
- 발표 자료 작성 및 발표

---

## 📊 실습 예제

각 교재에는 다음과 같은 실습 예제가 포함되어 있습니다:

1. **월별 매출 추세 분석**: 라인 차트로 시계열 패턴 파악
2. **업종별 매출 비교**: 바 차트로 상위 업종 시각화
3. **매출 분포 분석**: 히스토그램과 통계치로 전체 분포 이해
4. **점포수와 매출 관계**: 산점도로 상관관계 탐색
5. **연령대별 소비 패턴**: 바 차트로 주요 고객층 파악
6. **지역별 매출 비교**: Boxplot으로 분포 차이 시각화
7. **변수 간 상관관계**: Heatmap으로 다차원 관계 파악
8. **지리적 패턴**: Folium으로 지도 위에 데이터 시각화

---

## 🎯 학습 팁

### 효과적인 학습 방법
1. **이론 → 실습 → 응용** 순서로 진행
2. 각 교재의 코드를 **직접 타이핑**하며 실행
3. 퀴즈로 **이해도 점검**
4. 자신의 데이터로 **변형 실습**
5. 학습 내용을 **블로그나 노트에 정리**

### 시각화 원칙
- **명확성**: 그래프의 목적이 분명해야 함
- **간결성**: 불필요한 요소 제거
- **정확성**: 데이터를 왜곡하지 않음
- **미적 감각**: 색상, 레이아웃 조화
- **스토리텔링**: 인사이트를 전달

---

## 📚 추가 학습 자료

### 공식 문서
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Folium Documentation](https://python-visualization.github.io/folium/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### 추천 도서
- "Python 데이터 분석 입문" (한빛미디어)
- "파이썬 라이브러리를 활용한 데이터 분석" (한빛미디어)
- "데이터 시각화 교과서" (한빛미디어)

### 온라인 리소스
- [Kaggle Datasets](https://www.kaggle.com/datasets): 다양한 실습 데이터
- [Data to Viz](https://www.data-to-viz.com/): 시각화 유형 선택 가이드
- [ColorBrewer](https://colorbrewer2.org/): 색상 팔레트 추천

---

## 🤝 기여 및 피드백

이 교육 자료에 대한 피드백이나 개선 제안은 언제든 환영합니다!

- **이슈**: 오타, 코드 오류, 개선 제안 등
- **풀 리퀘스트**: 새로운 예제, 설명 보완 등

---

## 📄 라이선스

이 교육 자료는 학습 목적으로 자유롭게 사용 가능합니다.

---

## 👨‍🏫 문의

질문이나 도움이 필요하시면 GitHub Issues를 통해 문의해주세요.

**Happy Learning! 📊📈📉**
