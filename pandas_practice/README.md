# 📊 Pandas 활용을 통한 데이터 정제 및 가공 실무 교육 (재직자 과정)

> **대상:** 재직자 대상 실무형 데이터 분석 교육  
> **환경:** Python 3.x + Pandas, Windows + Miniconda + VSCode (CPU 환경)  
> **총 소요 시간:** 약 7시간 (각 교시 1시간)

---

## 📁 프로젝트 구조

```
📁 pandas_practice/
│
├── data/                           # 데이터 폴더
│   └── download_datasets.py        # 데이터셋 다운로드 스크립트
│
├── notebooks/                      # 실습 노트북 폴더
│   ├── 1_missing_values.md        # 1교시: 결측치 처리
│   ├── 2_groupby_analysis.md      # 2교시: groupby 집계
│   ├── 3_merge_join.md            # 3교시: 데이터 병합
│   ├── 4_apply_function.md        # 4교시: apply 함수
│   ├── 5_data_cleaning.md         # 5교시: 종합 실습 (1)
│   ├── 6_filter_group.md          # 6교시: 종합 실습 (2)
│   └── 7_save_output.md           # 7교시: 종합 실습 (3)
│
├── output/                         # 결과 파일 저장 폴더
│   └── cleaned_result.csv          # 최종 정제된 데이터
│
└── README.md                       # 본 파일 (과정 안내)
```

---

## 📚 커리큘럼 (총 7교시)

### 🕐 1교시: Pandas 활용 실습 - 결측치(NaN) 확인 및 처리
- `fillna`, `dropna`로 결측치 평균값 대체 또는 행 삭제
- 결측치 패턴 분석 및 시각화
- 실습 데이터: Seaborn Tips 데이터셋

### 🕑 2교시: groupby()를 활용한 카테고리별 집계
- 상품 카테고리별 평균 판매가격 및 총 판매량 계산
- 다중 그룹화 및 집계 함수 활용
- 실습 데이터: 판매 데이터셋

### 🕒 3교시: 두 개 이상의 데이터프레임 병합 (merge/join)
- 고객 정보 + 구매 내역 데이터를 고객 ID 기준으로 병합
- inner, left, right, outer join 이해
- 실습 데이터: 고객 및 주문 데이터

### 🕓 4교시: apply() 함수로 사용자 정의 함수 적용
- 가격(달러) 열을 환율 변환하여 원화 가격 열 추가
- lambda 함수와 일반 함수 활용
- 실습 데이터: 제품 가격 데이터

### 🕔 5교시: 종합 실습 (1) 데이터 불러오기 및 구조 파악
- 공공데이터 불러오기, 결측치 현황 및 데이터 구조 분석
- 데이터 품질 평가
- 실습 데이터: 실제 공공데이터

### 🕕 6교시: 종합 실습 (2) 필터링 및 그룹핑
- 특정 조건으로 필터링 후 월별, 지역별, 카테고리별 집계 수행
- 복합 조건 필터링 및 다차원 분석

### 🕖 7교시: 종합 실습 (3) 결과 저장
- 가공 완료된 데이터를 CSV로 저장
- 데이터 백업 및 버전 관리

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# Pandas 설치 (Miniconda 환경)
pip install pandas seaborn matplotlib numpy scikit-learn openpyxl
```

### 2. 데이터 다운로드

```bash
# 데이터 폴더로 이동
cd pandas_practice/data

# 데이터셋 다운로드 스크립트 실행
python download_datasets.py
```

### 3. 학습 순서

1. 각 교시별 노트북 파일을 순서대로 학습
2. 이론을 읽고 실습 코드를 직접 실행
3. 각 교시 마지막의 퀴즈를 풀며 복습
4. 실습 코드를 수정하며 실험

---

## 💡 학습 팁

- **실습 중심**: 코드를 직접 작성하고 실행하며 학습
- **에러 극복**: 에러가 발생하면 에러 메시지를 읽고 해결 방법을 찾아보기
- **응용 연습**: 제공된 예제를 변형하여 다양한 케이스 테스트
- **문서 활용**: [Pandas 공식 문서](https://pandas.pydata.org/docs/)를 참고

---

## 📖 참고 자료

- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Pandas 치트시트](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

---

## ✅ 학습 완료 체크리스트

- [ ] 1교시: 결측치 처리 완료
- [ ] 2교시: groupby 집계 완료
- [ ] 3교시: 데이터 병합 완료
- [ ] 4교시: apply 함수 활용 완료
- [ ] 5교시: 데이터 불러오기 및 구조 파악 완료
- [ ] 6교시: 필터링 및 그룹핑 완료
- [ ] 7교시: 결과 저장 완료

---

**작성일:** 2025-10-31  
**대상 수준:** 초급~중급 데이터 분석 실무자  
**업데이트:** 정기적으로 최신 Pandas 기능 반영
