# 🚀 빠른 시작 가이드

Python 데이터 분석 실무 종합 과정 (Final Session)을 시작하는 방법입니다.

---

## ⚡ 5분 만에 시작하기

### 1단계: 필수 패키지 설치

```bash
# 프로젝트 폴더로 이동
cd final

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2단계: 실습 데이터 생성

```bash
# data 폴더로 이동
cd data

# 데이터 생성 스크립트 실행
python generate_datasets.py
```

실행 결과:
```
✅ 판매 데이터 저장 완료 (1,000건)
✅ 고객 데이터 저장 완료 (500건)
✅ 상품 데이터 저장 완료 (200건)
✅ 거래 내역 데이터 저장 완료 (2,000건)
```

### 3단계: 예제 코드 실행

```bash
# code_examples 폴더로 이동
cd ../code_examples

# Python 기초 예제
python example_1_python.py

# Pandas 기초 예제
python example_2_pandas.py

# 종합 프로젝트
python final_project.py
```

---

## 📚 학습 순서

### 방법 1: Markdown 읽으며 학습 (권장)

1. `notebooks/1_python_essentials.md`를 엽니다
2. 이론을 읽고 코드 블록을 복사하여 실행합니다
3. 실습 문제를 풀어봅니다
4. 다음 노트북으로 진행합니다

```bash
# VSCode나 텍스트 에디터로 열기
code notebooks/1_python_essentials.md

# 또는 브라우저에서 보기 (GitHub)
```

### 방법 2: Jupyter Notebook 사용

```bash
# Jupyter 설치 (아직 설치하지 않았다면)
pip install jupyter

# Jupyter Notebook 실행
jupyter notebook
```

브라우저에서 `notebooks/` 폴더의 `.md` 파일을 엽니다.

### 방법 3: Python REPL에서 직접 실행

```bash
# Python 대화형 모드 실행
python

# 또는 IPython
ipython
```

노트북의 코드를 복사하여 붙여넣기합니다.

---

## 📖 커리큘럼 (7시간)

| 교시 | 시간 | 주제 | 노트북 |
|------|------|------|--------|
| 1교시 | 1시간 | Python 핵심 복습 | `1_python_essentials.md` |
| 2교시 | 1시간 | Pandas 기초 | `2_pandas_basics.md` |
| 3교시 | 1시간 | 데이터 정제 | `3_data_cleaning.md` |
| 4교시 | 1시간 | 집계 & 병합 | `4_aggregation_merge.md` |
| 5교시 | 1시간 | 데이터 시각화 | `5_visualization.md` |
| 6교시 | 1시간 | 머신러닝 기초 | `6_machine_learning.md` |
| 7교시 | 1시간 | 종합 프로젝트 | `7_final_project.md` |

---

## 💡 학습 팁

### ✅ DO (추천)
- 코드를 직접 타이핑하며 실행하기
- 에러 메시지를 읽고 이해하기
- 예제 데이터를 변경하며 실험하기
- 실습 문제를 꼭 풀어보기
- 배운 내용을 자신의 데이터에 적용하기

### ❌ DON'T (비추천)
- 코드만 복사-붙여넣기 하기
- 에러 무시하고 넘어가기
- 이론만 읽고 실습 안 하기
- 실습 문제 건너뛰기

---

## 🛠️ 문제 해결

### 문제 1: 패키지 설치 오류

```bash
# Python 버전 확인 (3.8 이상 권장)
python --version

# pip 업그레이드
pip install --upgrade pip

# 재설치
pip install -r requirements.txt --upgrade
```

### 문제 2: 데이터 파일이 없습니다

```bash
# data 폴더로 이동
cd data

# 데이터 재생성
python generate_datasets.py
```

### 문제 3: 한글이 깨집니다

```python
# 파일 읽을 때 인코딩 지정
df = pd.read_csv('data.csv', encoding='utf-8-sig')

# 또는
df = pd.read_csv('data.csv', encoding='cp949')
```

### 문제 4: 그래프에 한글이 안 나옵니다

```python
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
```

---

## 📁 프로젝트 구조

```
final/
│
├── README.md                    # 전체 과정 소개
├── QUICKSTART.md               # 이 파일 (빠른 시작)
├── requirements.txt            # 필수 패키지 목록
│
├── data/                       # 데이터 폴더
│   ├── generate_datasets.py   # 데이터 생성 스크립트 ⭐
│   ├── sales_data.csv         # 판매 데이터 (자동 생성됨)
│   ├── customer_data.csv      # 고객 데이터 (자동 생성됨)
│   ├── product_data.csv       # 상품 데이터 (자동 생성됨)
│   └── transaction_data.csv   # 거래 데이터 (자동 생성됨)
│
├── notebooks/                  # 교육 자료 ⭐
│   ├── 1_python_essentials.md
│   ├── 2_pandas_basics.md
│   ├── 3_data_cleaning.md
│   ├── 4_aggregation_merge.md
│   ├── 5_visualization.md
│   ├── 6_machine_learning.md
│   └── 7_final_project.md
│
├── code_examples/              # 완성된 코드 예제
│   ├── example_1_python.py
│   ├── example_2_pandas.py
│   └── final_project.py       # 종합 프로젝트 코드
│
└── output/                     # 결과물 저장
    ├── figures/               # 그래프 이미지
    ├── models/                # 학습된 모델
    └── reports/               # 분석 보고서
```

---

## 🎯 다음 단계

이 과정을 마친 후:

1. **실무 적용**: 자신의 업무 데이터로 분석 프로젝트 진행
2. **심화 학습**: 
   - 고급 머신러닝 (XGBoost, LightGBM)
   - 딥러닝 기초 (PyTorch, TensorFlow)
   - 시계열 분석 (Prophet, ARIMA)
3. **포트폴리오**: GitHub에 프로젝트 업로드
4. **커뮤니티**: Kaggle 대회 참가

---

## 📞 도움말

- 📖 [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- 🎓 [Scikit-learn 튜토리얼](https://scikit-learn.org/stable/tutorial/)
- 💬 [Stack Overflow](https://stackoverflow.com/questions/tagged/pandas)
- 🎯 [Kaggle Learn](https://www.kaggle.com/learn)

---

**준비 완료! 이제 학습을 시작하세요! 🚀**
