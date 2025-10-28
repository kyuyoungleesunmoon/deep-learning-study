# AI 교재 모듈: RNN & U-Net

이 디렉토리는 석사 수준 딥러닝 학습자를 위한 두 개의 comprehensive 교육 모듈을 포함합니다.

## 📘 모듈 A: RNN 기반 시계열 예측 (Netflix 주가)

**파일**: `notebooks/module_a_rnn_timeseries.ipynb`  
**셀 수**: 37개  
**크기**: 41 KB

### 내용
1. **이론 파트** (6개 섹션)
   - RNN, LSTM, GRU의 수학적 구조
   - 그래디언트 소실/폭주 문제
   - BPTT (Backpropagation Through Time)
   - 손실 함수 및 평가 지표

2. **합성 데이터 실험**
   - 사인파 + 노이즈 시계열 생성
   - RNN, LSTM, GRU 모델 비교
   - 하이퍼파라미터 튜닝 실험
   - 그래디언트 클리핑 효과 분석

3. **실제 데이터 학습**
   - Netflix 주가 데이터 (yfinance)
   - LSTM/GRU 학습 및 평가
   - 예측 결과 시각화
   - 오차 분석 및 한계점 논의

### 실행 방법
```bash
jupyter notebook notebooks/module_a_rnn_timeseries.ipynb
```

또는 Google Colab에서 실행

### 필요 패키지
- torch, numpy, matplotlib
- pandas, scikit-learn
- yfinance (주가 데이터)

---

## 📗 모듈 B: U-Net 기반 이미지 분할 (Oxford-IIIT Pet)

**파일**: `notebooks/module_b_unet_segmentation.ipynb`  
**셀 수**: 24개  
**크기**: 33 KB

### 내용
1. **이론 파트** (5개 섹션)
   - U-Net 인코더-디코더 구조
   - Convolution, Pooling, UpConvolution 연산
   - Skip Connection의 원리
   - Dice Loss, IoU 평가 지표

2. **합성 데이터 실험**
   - 단순 도형 (원, 사각형) 이미지 생성
   - 미니멀 U-Net 구현
   - 합성 데이터 학습 및 시각화
   - Skip connection 효과 분석

3. **실제 데이터 학습**
   - Oxford-IIIT Pet Dataset
   - 데이터 로딩 및 전처리
   - U-Net 학습 및 평가
   - 결과 분석 및 개선 방향

### 실행 방법
```bash
jupyter notebook notebooks/module_b_unet_segmentation.ipynb
```

또는 Google Colab에서 실행

### 필요 패키지
- torch, torchvision
- numpy, matplotlib
- scikit-learn

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. Jupyter 실행
```bash
jupyter notebook
```

### 3. 노트북 선택
- 시계열 예측 학습 → `module_a_rnn_timeseries.ipynb`
- 이미지 분할 학습 → `module_b_unet_segmentation.ipynb`

---

## 📝 학습 가이드

### 권장 학습 순서
1. 모듈 A 완료 (4-6시간)
2. 모듈 B 완료 (4-6시간)
3. 두 모듈 비교 및 통합 이해

### 학습 방법
1. **이론 섹션**: 수식과 설명을 천천히 읽기
2. **코드 셀**: 실행하며 결과 확인
3. **실험**: 파라미터 변경하며 직접 실험
4. **분석**: 결과를 해석하고 인사이트 도출

---

## 🔧 컨텐츠 생성

이 모듈들은 자동 생성 시스템으로 만들어졌습니다:

```bash
# Module A 재생성
python3 -c "from module_a_content import get_all_content; ..."

# Module B 재생성
python3 -c "from module_b_content import get_all_content; ..."
```

**생성기 파일:**
- `module_a_content.py` - Module A 컨텐츠 정의
- `module_b_content.py` - Module B 컨텐츠 정의
- `generate_modules.py` - 통합 생성 스크립트

---

## 📚 추가 자료

### 참고 논문
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **GRU**: Cho et al. (2014)
- **U-Net**: Ronneberger et al. (2015)

### 관련 리소스
- PyTorch 공식 문서: https://pytorch.org/docs/
- Deep Learning Book: https://www.deeplearningbook.org/

---

## 🤝 기여

개선 사항이나 오류 발견 시 issue를 열어주세요!

---

**만든이**: AI 교재 자동 생성 시스템  
**최종 업데이트**: 2025-10-28  
**버전**: 1.0
