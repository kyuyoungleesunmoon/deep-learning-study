# 🎓 Python 데이터 분석 실무 종합 과정 (Final Session)

> **대상**: 재직자 대상 실무형 Python 데이터 분석 마스터 과정  
> **환경**: Python 3.8+ + Pandas, Matplotlib, Seaborn, Scikit-learn  
> **총 소요 시간**: 7시간 (마지막 수업일 종합 정리)

---

## 📚 과정 개요

이 과정은 Python 데이터 분석의 전체 워크플로우를 7시간 동안 집중적으로 학습하는 종합 실습 과정입니다. 
실무에서 바로 사용 가능한 데이터 분석 기법과 시각화 방법을 배우고, 실제 프로젝트를 완성합니다.

### 🎯 학습 목표

1. **Python 기초 복습**: 데이터 분석에 필요한 핵심 Python 문법
2. **Pandas 데이터 처리**: 데이터 정제, 변환, 집계, 병합
3. **데이터 시각화**: Matplotlib, Seaborn을 활용한 효과적인 시각화
4. **머신러닝 기초**: Scikit-learn을 활용한 예측 모델 구축
5. **실무 프로젝트**: End-to-end 데이터 분석 프로젝트 완성

---

## 📅 7시간 커리큘럼

### 1교시 (1시간): Python 핵심 복습 & 데이터 구조
**파일**: `notebooks/1_python_essentials.md`

**학습 내용**:
- Python 기본 자료형 (List, Dict, Tuple, Set)
- 조건문, 반복문, 함수
- List Comprehension (리스트 컴프리헨션)
- Lambda 함수
- 실무 예제: 매출 데이터 처리 함수 작성

**실습**:
- 직원 정보 관리 프로그램
- 매출 데이터 필터링 함수
- 데이터 변환 및 집계

---

### 2교시 (1시간): Pandas 기초 - 데이터 불러오기 & 탐색
**파일**: `notebooks/2_pandas_basics.md`

**학습 내용**:
- DataFrame 생성 및 구조 이해
- 데이터 불러오기 (CSV, Excel)
- 기본 탐색 (head, tail, info, describe)
- 행/열 선택 (loc, iloc)
- 조건 필터링
- 정렬 (sort_values)

**실습 데이터**: 
- 전자상거래 판매 데이터 (E-commerce Sales Dataset)
- 고객 구매 이력 데이터

---

### 3교시 (1시간): Pandas 고급 - 데이터 정제 & 변환
**파일**: `notebooks/3_data_cleaning.md`

**학습 내용**:
- 결측치 처리 (fillna, dropna, interpolate)
- 중복 데이터 제거 (drop_duplicates)
- 데이터 타입 변환 (astype)
- 문자열 처리 (str 접근자)
- 날짜/시간 처리 (datetime)
- apply 함수로 사용자 정의 변환

**실습**:
- 실제 데이터셋 정제하기
- 불완전한 데이터 처리 전략

---

### 4교시 (1시간): Pandas 심화 - 집계 & 병합
**파일**: `notebooks/4_aggregation_merge.md`

**학습 내용**:
- GroupBy 집계 (평균, 합계, 개수)
- 다중 컬럼 그룹화
- pivot_table 활용
- 데이터 병합 (merge, join, concat)
- 실무 예제: 부서별/월별 매출 분석

**실습**:
- 지역별 상품 판매량 분석
- 고객 세그먼트별 구매 패턴 분석

---

### 5교시 (1시간): 데이터 시각화 - Matplotlib & Seaborn
**파일**: `notebooks/5_visualization.md`

**학습 내용**:
- Matplotlib 기초 (라인, 바, 산점도, 히스토그램)
- Seaborn 활용 (barplot, boxplot, heatmap, pairplot)
- 서브플롯 구성
- 그래프 커스터마이징 (색상, 레이블, 범례)
- 실무 보고서용 시각화 팁

**실습**:
- 월별 매출 추이 그래프
- 상품 카테고리별 판매 비교
- 상관관계 히트맵
- 고객 연령대별 구매액 분포

---

### 6교시 (1시간): 머신러닝 기초 - 예측 모델 만들기
**파일**: `notebooks/6_machine_learning.md`

**학습 내용**:
- 머신러닝 개념 이해
- 데이터 전처리 (StandardScaler, LabelEncoder)
- Train/Test 분리
- 선형 회귀 (Linear Regression)
- 결정 트리 (Decision Tree)
- 랜덤 포레스트 (Random Forest)
- 모델 평가 (MSE, RMSE, R², 정확도)

**실습**:
- 매출 예측 모델 구축
- 고객 이탈 예측 (분류)
- 특성 중요도 분석

---

### 7교시 (1시간): 종합 프로젝트 - 실무 데이터 분석
**파일**: `notebooks/7_final_project.md`

**프로젝트 주제**: **온라인 쇼핑몰 매출 분석 및 예측**

**프로젝트 단계**:
1. **문제 정의**: 비즈니스 목표 설정
2. **데이터 수집**: 실습 데이터 로드
3. **데이터 탐색 (EDA)**: 기본 통계 및 시각화
4. **데이터 정제**: 결측치, 이상치 처리
5. **특성 엔지니어링**: 새로운 변수 생성
6. **시각화**: 핵심 인사이트 도출
7. **예측 모델**: 매출 예측 모델 구축
8. **결과 해석**: 비즈니스 의사결정 제안
9. **보고서 작성**: 최종 결과물 정리

**실습 결과물**:
- 데이터 분석 보고서 (Markdown/PDF)
- 시각화 그래프 (PNG 파일)
- 예측 모델 및 평가 결과
- 최종 정제된 데이터 (CSV)

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# 필요한 라이브러리 설치
pip install pandas numpy matplotlib seaborn scikit-learn jupyter openpyxl
```

또는 requirements.txt 사용:

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# final 폴더로 이동
cd final

# 실습 데이터 자동 생성
python data/generate_datasets.py
```

실행 후 다음 데이터셋이 생성됩니다:
- `data/sales_data.csv` - 전자상거래 판매 데이터
- `data/customer_data.csv` - 고객 정보 데이터
- `data/product_data.csv` - 상품 정보 데이터
- `data/transaction_data.csv` - 거래 내역 데이터

### 3. 학습 시작

**방법 1: Jupyter Notebook 사용 (권장)**

```bash
jupyter notebook
```

브라우저에서 `notebooks/` 폴더의 파일을 순서대로 엽니다.

**방법 2: Markdown 읽고 Python 스크립트 실행**

1. `notebooks/1_python_essentials.md`부터 순서대로 읽기
2. 코드 블록을 복사하여 Python 파일 또는 REPL에서 실행
3. `code_examples/` 폴더의 완성된 예제 코드 참고

---

## 📂 프로젝트 구조

```
final/
│
├── README.md                          # 이 파일 (과정 안내)
│
├── data/                              # 데이터 폴더
│   ├── generate_datasets.py          # 데이터 자동 생성 스크립트
│   ├── sales_data.csv                # 판매 데이터
│   ├── customer_data.csv             # 고객 데이터
│   ├── product_data.csv              # 상품 데이터
│   └── transaction_data.csv          # 거래 데이터
│
├── notebooks/                         # 교육 자료 (실습 노트)
│   ├── 1_python_essentials.md        # 1교시: Python 핵심
│   ├── 2_pandas_basics.md            # 2교시: Pandas 기초
│   ├── 3_data_cleaning.md            # 3교시: 데이터 정제
│   ├── 4_aggregation_merge.md        # 4교시: 집계 & 병합
│   ├── 5_visualization.md            # 5교시: 시각화
│   ├── 6_machine_learning.md         # 6교시: 머신러닝
│   └── 7_final_project.md            # 7교시: 종합 프로젝트
│
├── code_examples/                     # 완성된 코드 예제
│   ├── example_1_python.py           # 1교시 예제
│   ├── example_2_pandas.py           # 2교시 예제
│   ├── example_3_cleaning.py         # 3교시 예제
│   ├── example_4_groupby.py          # 4교시 예제
│   ├── example_5_viz.py              # 5교시 예제
│   ├── example_6_ml.py               # 6교시 예제
│   └── final_project.py              # 7교시 프로젝트
│
└── output/                            # 결과물 저장 폴더
    ├── figures/                       # 시각화 그래프
    ├── models/                        # 저장된 모델
    └── reports/                       # 분석 보고서
```

---

## 💡 학습 팁

1. **순서대로 학습**: 1교시부터 순차적으로 진행하세요
2. **코드 직접 실행**: 복사-붙여넣기만 하지 말고 직접 타이핑하며 실행
3. **데이터 탐색**: 예제 코드를 수정하며 다양한 분석 시도
4. **에러 해결**: 에러가 발생하면 메시지를 읽고 스스로 해결해보기
5. **실무 적용**: 배운 내용을 자신의 업무 데이터에 적용해보기

---

## 📖 추가 학습 자료

### 공개 데이터셋 다운로드 사이트
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [공공데이터포털](https://www.data.go.kr/)
- [서울 열린데이터광장](https://data.seoul.go.kr/)

### 온라인 학습 자료
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Scikit-learn 공식 문서](https://scikit-learn.org/stable/)
- [Matplotlib 공식 문서](https://matplotlib.org/)
- [Seaborn 공식 문서](https://seaborn.pydata.org/)

---

## 🎯 학습 후 체크리스트

이 과정을 완료하면 다음을 할 수 있어야 합니다:

- [ ] Python으로 데이터 구조를 자유롭게 다룰 수 있다
- [ ] CSV/Excel 파일을 Pandas로 불러와 탐색할 수 있다
- [ ] 결측치와 이상치를 적절히 처리할 수 있다
- [ ] GroupBy로 다양한 집계 분석을 수행할 수 있다
- [ ] 여러 데이터프레임을 병합하여 분석할 수 있다
- [ ] Matplotlib/Seaborn으로 효과적인 시각화를 만들 수 있다
- [ ] 간단한 머신러닝 모델을 구축하고 평가할 수 있다
- [ ] 실무 데이터 분석 프로젝트를 처음부터 끝까지 수행할 수 있다

---

## ⚠️ 주의사항

1. **데이터 크기**: 실습 데이터는 학습용으로 크기가 제한되어 있습니다
2. **실행 환경**: Python 3.8 이상 권장
3. **라이브러리 버전**: 최신 버전 사용 권장
4. **저작권**: 실습 데이터는 교육 목적으로만 사용하세요

---

## 🤝 문의 및 지원

- 코드 오류 발견 시: Issue 등록
- 추가 질문: Discussion 활용
- 개선 제안: Pull Request 환영

---

**Good Luck! 🚀**

이 과정을 통해 데이터 분석 실무 역량을 한 단계 업그레이드하세요!
