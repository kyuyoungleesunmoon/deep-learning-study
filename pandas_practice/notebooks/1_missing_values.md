# 📚 1교시: Pandas 활용 실습 - 결측치(NaN) 확인 및 처리

> **학습 목표:** 결측치를 확인하고 `fillna`, `dropna`를 활용하여 적절히 처리하는 방법을 학습합니다.  
> **소요 시간:** 1시간  
> **사용 데이터:** Seaborn Tips 데이터셋

---

## 🧠 이론 설명

### 1.1 결측치(Missing Value)란?

**결측치(Missing Value)**는 데이터셋에서 값이 없는 경우를 의미합니다. Pandas에서는 주로 `NaN` (Not a Number) 또는 `None`으로 표현됩니다.

#### 결측치가 발생하는 원인
- 📝 **입력 누락**: 설문조사에서 응답자가 답변하지 않은 경우
- 🔌 **수집 오류**: 센서 고장이나 통신 장애로 데이터가 기록되지 않은 경우
- 🚫 **해당 없음**: 특정 항목이 해당 케이스에 적용되지 않는 경우
- 💾 **데이터 손실**: 파일 전송 중 데이터가 손실된 경우

### 1.2 결측치 확인 방법

```python
# 결측치 확인 메서드
df.isnull()       # 결측치면 True, 아니면 False 반환
df.isna()         # isnull()과 동일
df.isnull().sum() # 각 컬럼별 결측치 개수
df.info()         # 컬럼별 Non-Null Count 확인
```

### 1.3 결측치 처리 방법

#### 방법 1: 삭제 (`dropna`)
```python
df.dropna()              # 결측치가 하나라도 있는 행 삭제
df.dropna(subset=['A'])  # 특정 컬럼에 결측치가 있는 행만 삭제
df.dropna(axis=1)        # 결측치가 있는 열 삭제
df.dropna(thresh=3)      # 최소 3개 이상의 값이 있는 행만 유지
```

#### 방법 2: 채우기 (`fillna`)
```python
df.fillna(0)                    # 모든 결측치를 0으로 채우기
df.fillna(df.mean())            # 각 컬럼의 평균값으로 채우기
df['A'].fillna(df['A'].median()) # A컬럼을 중앙값으로 채우기
df.fillna(method='ffill')       # 앞 값으로 채우기 (Forward Fill)
df.fillna(method='bfill')       # 뒷 값으로 채우기 (Backward Fill)
```

### 1.4 어떤 방법을 선택해야 할까?

| 상황 | 권장 방법 | 이유 |
|------|----------|------|
| 결측치 비율이 5% 미만 | 삭제 (`dropna`) | 데이터 손실이 적고 간단함 |
| 수치형 데이터 | 평균/중앙값으로 채우기 | 분포를 크게 왜곡하지 않음 |
| 범주형 데이터 | 최빈값으로 채우기 | 가장 많이 나타나는 값 사용 |
| 시계열 데이터 | 앞/뒤 값으로 채우기 | 시간적 연속성 유지 |
| 결측치 비율이 50% 이상 | 해당 컬럼 삭제 | 정보가 너무 부족함 |

---

## 💻 실습 코드

### 환경 설정 및 데이터 로드

```python
# 필수 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
# 방법 1: Seaborn 내장 데이터
tips = sns.load_dataset('tips')

# 방법 2: 로컬 파일에서 로드 (data 폴더에서 다운로드한 경우)
# tips = pd.read_csv('../data/tips.csv')

print("=" * 80)
print("📊 Tips 데이터셋 기본 정보")
print("=" * 80)
print(f"데이터 크기: {tips.shape[0]} 행, {tips.shape[1]} 열")
print(f"\n컬럼 정보:")
print(tips.info())
print(f"\n상위 5개 데이터:")
print(tips.head())
```

### 실습 1: 인위적으로 결측치 만들기

실제 결측치 처리를 연습하기 위해 Tips 데이터에 의도적으로 결측치를 추가합니다.

```python
# 원본 데이터 복사
tips_missing = tips.copy()

# 무작위로 결측치 생성
np.random.seed(42)

# total_bill에 10개의 결측치 추가
missing_idx = np.random.choice(tips_missing.index, size=10, replace=False)
tips_missing.loc[missing_idx, 'total_bill'] = np.nan

# tip에 15개의 결측치 추가
missing_idx = np.random.choice(tips_missing.index, size=15, replace=False)
tips_missing.loc[missing_idx, 'tip'] = np.nan

# size에 8개의 결측치 추가
missing_idx = np.random.choice(tips_missing.index, size=8, replace=False)
tips_missing.loc[missing_idx, 'size'] = np.nan

print("\n" + "=" * 80)
print("🔍 결측치 생성 완료")
print("=" * 80)
```

### 실습 2: 결측치 확인

```python
print("\n[1] isnull()을 사용한 결측치 확인")
print("-" * 80)
print("각 컬럼별 결측치 개수:")
print(tips_missing.isnull().sum())

print("\n[2] 결측치 비율 계산")
print("-" * 80)
missing_ratio = (tips_missing.isnull().sum() / len(tips_missing)) * 100
print(missing_ratio.round(2))

print("\n[3] 결측치가 있는 행 확인")
print("-" * 80)
print(f"결측치가 있는 행의 개수: {tips_missing.isnull().any(axis=1).sum()}")

# 결측치가 있는 행 일부 출력
print("\n결측치가 있는 행 샘플 (상위 10개):")
print(tips_missing[tips_missing.isnull().any(axis=1)].head(10))
```

### 실습 3: 결측치 시각화

```python
# 결측치 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 그래프 1: 컬럼별 결측치 개수
missing_counts = tips_missing.isnull().sum()
missing_counts[missing_counts > 0].plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('컬럼별 결측치 개수', fontsize=14, fontweight='bold')
axes[0].set_xlabel('컬럼명')
axes[0].set_ylabel('결측치 개수')
axes[0].grid(axis='y', alpha=0.3)

# 그래프 2: 결측치 히트맵
sns.heatmap(tips_missing.isnull(), cbar=True, yticklabels=False, 
            cmap='YlOrRd', ax=axes[1])
axes[1].set_title('결측치 패턴 히트맵 (노란색=데이터, 빨간색=결측치)', 
                  fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/missing_values_visualization.png', dpi=100, bbox_inches='tight')
print("\n📊 결측치 시각화 저장 완료: ../output/missing_values_visualization.png")
plt.show()
```

### 실습 4: 결측치 삭제 (dropna)

```python
print("\n" + "=" * 80)
print("🗑️ 방법 1: 결측치 삭제 (dropna)")
print("=" * 80)

# 원본 데이터 크기
print(f"원본 데이터 크기: {tips_missing.shape}")

# 결측치가 있는 모든 행 삭제
tips_dropped = tips_missing.dropna()
print(f"결측치 삭제 후 크기: {tips_dropped.shape}")
print(f"삭제된 행의 개수: {len(tips_missing) - len(tips_dropped)}")

# 특정 컬럼의 결측치만 삭제
tips_dropped_subset = tips_missing.dropna(subset=['total_bill', 'tip'])
print(f"\ntotal_bill, tip 컬럼 결측치만 삭제 후 크기: {tips_dropped_subset.shape}")

# 검증: 결측치가 모두 제거되었는지 확인
print(f"\n결측치 삭제 후 남은 결측치:")
print(tips_dropped.isnull().sum())
```

### 실습 5: 결측치 채우기 (fillna)

```python
print("\n" + "=" * 80)
print("✏️ 방법 2: 결측치 채우기 (fillna)")
print("=" * 80)

# 평균값으로 채우기
tips_filled_mean = tips_missing.copy()
tips_filled_mean['total_bill'] = tips_filled_mean['total_bill'].fillna(
    tips_filled_mean['total_bill'].mean()
)
tips_filled_mean['tip'] = tips_filled_mean['tip'].fillna(
    tips_filled_mean['tip'].mean()
)
tips_filled_mean['size'] = tips_filled_mean['size'].fillna(
    tips_filled_mean['size'].mean()
)

print("[1] 평균값으로 채우기 완료")
print(f"total_bill 평균: {tips_missing['total_bill'].mean():.2f}")
print(f"tip 평균: {tips_missing['tip'].mean():.2f}")
print(f"size 평균: {tips_missing['size'].mean():.2f}")

# 중앙값으로 채우기
tips_filled_median = tips_missing.copy()
tips_filled_median['total_bill'] = tips_filled_median['total_bill'].fillna(
    tips_filled_median['total_bill'].median()
)
tips_filled_median['tip'] = tips_filled_median['tip'].fillna(
    tips_filled_median['tip'].median()
)
tips_filled_median['size'] = tips_filled_median['size'].fillna(
    tips_filled_median['size'].median()
)

print("\n[2] 중앙값으로 채우기 완료")
print(f"total_bill 중앙값: {tips_missing['total_bill'].median():.2f}")
print(f"tip 중앙값: {tips_missing['tip'].median():.2f}")
print(f"size 중앙값: {tips_missing['size'].median():.2f}")

# 특정 값으로 채우기
tips_filled_zero = tips_missing.fillna(0)
print("\n[3] 0으로 채우기 완료")

# 검증: 결측치가 모두 채워졌는지 확인
print(f"\n평균값으로 채운 후 남은 결측치:")
print(tips_filled_mean.isnull().sum())
```

### 실습 6: 결과 비교

```python
print("\n" + "=" * 80)
print("📊 결측치 처리 방법별 통계 비교")
print("=" * 80)

comparison = pd.DataFrame({
    '방법': ['원본(결측치O)', '삭제', '평균값', '중앙값', '0으로채움'],
    '행 개수': [
        len(tips_missing),
        len(tips_dropped),
        len(tips_filled_mean),
        len(tips_filled_median),
        len(tips_filled_zero)
    ],
    'total_bill 평균': [
        tips_missing['total_bill'].mean(),
        tips_dropped['total_bill'].mean(),
        tips_filled_mean['total_bill'].mean(),
        tips_filled_median['total_bill'].mean(),
        tips_filled_zero['total_bill'].mean()
    ],
    'tip 평균': [
        tips_missing['tip'].mean(),
        tips_dropped['tip'].mean(),
        tips_filled_mean['tip'].mean(),
        tips_filled_median['tip'].mean(),
        tips_filled_zero['tip'].mean()
    ]
})

print(comparison.round(2))
print("\n💡 해석:")
print("- 삭제 방법: 데이터 손실이 있지만 통계적 분포 유지")
print("- 평균값 채우기: 데이터 유지하나 분산 감소 가능")
print("- 중앙값 채우기: 이상치 영향 적음")
print("- 0으로 채우기: 통계적 분포 왜곡 (권장하지 않음)")
```

---

## 🧩 퀴즈

### 문제 1
다음 중 결측치를 확인하는 Pandas 메서드가 아닌 것은?

1. `isnull()`
2. `isna()`
3. `ismissing()`
4. `info()`

<details>
<summary>정답 보기</summary>

**정답: 3번 `ismissing()`**

**해설:**
- Pandas에서 결측치 확인 메서드는 `isnull()`, `isna()`, `info()` 등이 있습니다.
- `ismissing()`은 Pandas에 존재하지 않는 메서드입니다.
- `isnull()`과 `isna()`는 동일한 기능을 수행합니다.
</details>

---

### 문제 2
DataFrame `df`에서 `age` 컬럼의 결측치를 해당 컬럼의 평균값으로 채우는 올바른 코드는?

1. `df['age'].fillna(df['age'].sum())`
2. `df['age'].fillna(df['age'].mean())`
3. `df['age'].dropna(df['age'].mean())`
4. `df['age'].replace(NaN, df['age'].mean())`

<details>
<summary>정답 보기</summary>

**정답: 2번 `df['age'].fillna(df['age'].mean())`**

**해설:**
- `fillna()` 메서드는 결측치를 지정한 값으로 채웁니다.
- `df['age'].mean()`은 age 컬럼의 평균값을 계산합니다.
- `dropna()`는 삭제 메서드이므로 채우기에 사용할 수 없습니다.
- `replace()`도 사용 가능하나, 결측치 처리에는 `fillna()`가 더 적합합니다.
</details>

---

### 문제 3
다음 상황에서 가장 적절한 결측치 처리 방법은?

**상황:** 고객 만족도 설문조사 데이터에서 소득 컬럼의 결측치가 40%입니다. 이 컬럼은 분석의 핵심 변수입니다.

1. `dropna()`로 결측치가 있는 모든 행 삭제
2. 소득 컬럼을 아예 삭제
3. 평균값으로 모든 결측치 채우기
4. 다른 변수(나이, 직업 등)를 활용한 예측 모델로 결측치 추정

<details>
<summary>정답 보기</summary>

**정답: 4번 다른 변수를 활용한 예측 모델로 결측치 추정**

**해설:**
- **1번 (삭제)**: 40%의 데이터 손실은 너무 큼. 분석의 신뢰도가 크게 하락함.
- **2번 (컬럼 삭제)**: 핵심 변수를 삭제하면 분석 목적 달성 불가.
- **3번 (평균값)**: 단순 평균값 대체는 40%나 되는 데이터를 같은 값으로 만들어 분포를 크게 왜곡함.
- **4번 (예측 모델)**: 나이, 직업, 교육수준 등 관련 변수를 활용하여 결측치를 예측하면 더 정확하고 현실적인 값으로 채울 수 있음. (예: KNN Imputation, Regression Imputation)

💡 **실무 팁**: 
- 결측치 비율이 30% 이상이면 단순 평균/중앙값 대체는 피하고,
- Multiple Imputation, KNN, 또는 머신러닝 기반 방법을 고려하세요.
</details>

---

### 문제 4
다음 코드의 결과는?

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})
result = df.dropna(thresh=2)
```

결과 DataFrame의 행 개수는?

<details>
<summary>정답 보기</summary>

**정답: 3개 행**

**해설:**
```
원본 DataFrame:
     A    B   C
0  1.0  5.0   9
1  2.0  NaN  10
2  NaN  NaN  11
3  4.0  8.0  12
```

- `dropna(thresh=2)`: 최소 2개 이상의 non-null 값이 있는 행만 유지
- 0번 행: 3개 값 (1, 5, 9) → 유지 ✅
- 1번 행: 2개 값 (2, 10) → 유지 ✅
- 2번 행: 1개 값 (11) → 삭제 ❌
- 3번 행: 3개 값 (4, 8, 12) → 유지 ✅

따라서 **3개 행**이 남습니다.

💡 `thresh` 파라미터는 "최소 이만큼의 non-null 값이 있어야 한다"는 조건을 설정합니다.
</details>

---

## ✅ 1교시 학습 완료 체크리스트

- [ ] 결측치의 개념과 발생 원인 이해
- [ ] `isnull()`, `isna()`, `info()` 메서드로 결측치 확인
- [ ] `dropna()` 메서드로 결측치 삭제
- [ ] `fillna()` 메서드로 결측치 채우기 (평균, 중앙값, 특정값)
- [ ] 결측치 처리 방법별 장단점 이해
- [ ] 실습 코드 직접 실행 및 결과 확인
- [ ] 퀴즈 4문제 모두 풀이 완료

---

**다음 학습:** [2교시 - groupby()를 활용한 카테고리별 집계](./2_groupby_analysis.md)

**학습 완료일:** _____________  
**소요 시간:** _____________  
**이해도 (1~5):** ⭐⭐⭐⭐⭐
