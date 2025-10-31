# 🚀 빠른 시작 가이드 (Quick Start Guide)

이 가이드는 Pandas 데이터 정제 및 가공 실무 교육을 5분 안에 시작할 수 있도록 도와줍니다.

---

## 📋 준비물 체크리스트

- [ ] Python 3.7 이상 설치됨
- [ ] pip 또는 conda 사용 가능
- [ ] 텍스트 에디터 또는 IDE (VSCode 권장)
- [ ] 인터넷 연결 (패키지 설치용)

---

## ⚡ 3단계로 시작하기

### 1단계: 필수 패키지 설치 (2분)

터미널 또는 명령 프롬프트에서 실행:

```bash
# Pip 사용 시
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

# Conda 사용 시 (Miniconda/Anaconda)
conda install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

**설치 확인:**
```bash
python -c "import pandas; print(f'Pandas {pandas.__version__} 설치 완료!')"
```

---

### 2단계: 데이터 준비 (선택 사항, 1분)

**데이터는 이미 포함되어 있습니다!** 따로 다운로드할 필요 없습니다.

만약 데이터를 재생성하고 싶다면:

```bash
cd pandas_practice/data
python download_datasets.py
```

**생성되는 파일:**
- `tips.csv` - 팁 데이터 (244행)
- `sales_data.csv` - 판매 데이터 (200행)
- `customers.csv` - 고객 데이터 (100행)
- `orders.csv` - 주문 데이터 (300행)
- `products_usd.csv` - 제품 가격 데이터 (50행)
- `public_business_data.csv` - 공공 비즈니스 데이터 (500행)

---

### 3단계: 학습 시작 (2분)

1. **notebooks 폴더로 이동**
   ```bash
   cd pandas_practice/notebooks
   ```

2. **1교시부터 순서대로 학습**
   - VSCode에서 열기: `code 1_missing_values.md`
   - 또는 GitHub에서 직접 읽기
   - 또는 메모장/텍스트 에디터로 열기

3. **실습 코드 실행**
   - 코드를 복사하여 Python 인터프리터나 Jupyter Notebook에서 실행
   - 또는 `.py` 파일로 저장 후 실행

---

## 📚 학습 순서

**권장 학습 경로:**

```
1️⃣ 1교시 (1시간) → 결측치 처리
    ↓
2️⃣ 2교시 (1시간) → groupby 집계
    ↓
3️⃣ 3교시 (1시간) → 데이터 병합
    ↓
4️⃣ 4교시 (1시간) → apply 함수
    ↓
5️⃣ 5교시 (1시간) → 데이터 탐색
    ↓
6️⃣ 6교시 (1시간) → 필터링 및 그룹핑
    ↓
7️⃣ 7교시 (1시간) → 결과 저장
    ↓
🎉 완료! 수료증 발급
```

**각 교시 학습 방법:**
1. 📖 이론 섹션 읽기 (10분)
2. 💻 실습 코드 따라하기 (40분)
3. 🧩 퀴즈 풀기 (10분)

---

## 💡 학습 팁

### ✅ 효과적인 학습 방법
- **직접 타이핑**: 복사/붙여넣기보다 직접 코드 작성
- **실험하기**: 파라미터를 바꿔가며 결과 관찰
- **에러 해결**: 에러 메시지를 읽고 스스로 해결 시도
- **메모하기**: 중요한 내용은 별도로 정리

### ❌ 피해야 할 습관
- 코드만 복사하고 이해하지 않기
- 에러가 나면 바로 포기하기
- 퀴즈를 건너뛰기
- 한 번에 모든 내용을 학습하려고 하기

---

## 🆘 문제 해결 (FAQ)

### Q1: `ModuleNotFoundError: No module named 'pandas'`
**A:** Pandas가 설치되지 않았습니다.
```bash
pip install pandas
```

### Q2: 한글이 깨져요
**A:** 한글 폰트 설정 코드를 추가하세요:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# 또는
plt.rcParams['font.family'] = 'AppleGothic'    # Mac
```

### Q3: 데이터 파일을 찾을 수 없어요
**A:** 현재 작업 디렉토리를 확인하세요:
```python
import os
print(os.getcwd())  # 현재 디렉토리 출력
```
notebooks 폴더에서 실행 중이라면 `../data/` 경로가 맞습니다.

### Q4: Jupyter Notebook에서 실행하고 싶어요
**A:** Markdown 파일 내용을 Jupyter Notebook으로 변환:
```bash
# Jupyter 설치
pip install jupyter

# Notebook 실행
jupyter notebook
```
그 후 새 노트북을 만들고 코드 블록을 복사하여 실행하세요.

### Q5: 코드가 느려요
**A:** 대부분 정상입니다. 첫 실행 시 데이터 로드와 라이브러리 import가 시간이 걸립니다.

---

## 📊 예상 학습 시간

| 단계 | 소요 시간 | 누적 시간 |
|------|----------|----------|
| 환경 설정 | 10분 | 10분 |
| 1교시 | 1시간 | 1시간 10분 |
| 2교시 | 1시간 | 2시간 10분 |
| 3교시 | 1시간 | 3시간 10분 |
| 4교시 | 1시간 | 4시간 10분 |
| 5교시 | 1시간 | 5시간 10분 |
| 6교시 | 1시간 | 6시간 10분 |
| 7교시 | 1시간 | 7시간 10분 |

**총 예상 시간:** 약 7시간 (휴식 포함 1일 집중 학습 가능)

---

## 🎯 다음 단계

학습 완료 후:

1. **복습**: 중요한 개념을 노트로 정리
2. **실습**: 실제 업무 데이터에 적용
3. **심화**: [Pandas 공식 문서](https://pandas.pydata.org/docs/) 참고
4. **프로젝트**: 캐글(Kaggle) 데이터셋으로 분석 프로젝트 수행

---

## 📞 도움이 필요하신가요?

- 📚 [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- 🔍 [Stack Overflow - Pandas 태그](https://stackoverflow.com/questions/tagged/pandas)
- 💬 [Pandas GitHub Discussions](https://github.com/pandas-dev/pandas/discussions)

---

**준비되셨나요? 그럼 시작해볼까요! 🚀**

👉 [1교시 - 결측치 처리 시작하기](./notebooks/1_missing_values.md)
