# 📚 7교시: 종합 실습 (3) 결과 저장 및 최종 정리

> **학습 목표:** 분석 및 가공이 완료된 데이터를 다양한 형식으로 저장하고, 전체 학습 내용을 정리합니다.  
> **소요 시간:** 1시간  
> **사용 데이터:** 5-6교시에서 가공한 데이터

---

## 🧠 이론 설명

### 7.1 데이터 저장의 중요성

분석 결과를 저장하는 것은 다음과 같은 이유로 중요합니다:

- **재사용성**: 동일한 전처리를 반복하지 않음
- **공유**: 팀원이나 고객과 결과 공유
- **백업**: 데이터 손실 방지
- **버전 관리**: 처리 단계별로 저장하여 추적 가능

### 7.2 주요 파일 형식

| 형식 | 확장자 | 장점 | 단점 | 용도 |
|------|--------|------|------|------|
| **CSV** | .csv | 범용성, 가벼움 | 타입 정보 손실 | 일반적인 데이터 교환 |
| **Excel** | .xlsx | 시트, 서식 지원 | 용량 큼 | 보고서, 비즈니스 |
| **JSON** | .json | 계층 구조 | 용량 큼 | 웹 API, 설정 파일 |
| **Pickle** | .pkl | 모든 정보 보존 | Python 전용 | 임시 저장, 캐싱 |
| **Parquet** | .parquet | 압축률 좋음 | 범용성 낮음 | 빅데이터, 데이터 웨어하우스 |

### 7.3 파일 저장 시 고려사항

1. **인코딩**: 한글 포함 시 `encoding='utf-8-sig'` 또는 `'cp949'`
2. **인덱스**: `index=False`로 불필요한 인덱스 컬럼 제외
3. **압축**: 대용량 데이터는 `compression='gzip'`
4. **파일명**: 날짜 포함하여 버전 관리 (예: `result_20241031.csv`)

---

## 💻 실습 코드

### 환경 설정 및 데이터 준비

```python
# 필수 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🎯 7교시: 결과 저장 및 최종 정리")
print("=" * 80)

# 데이터 로드 및 전처리 (5-6교시 내용 종합)
df = pd.read_csv('../data/public_business_data.csv', encoding='utf-8-sig')

# 날짜 변환
df['접수일자'] = pd.to_datetime(df['접수일자'])
df['월'] = df['접수일자'].dt.month
df['년도'] = df['접수일자'].dt.year

# 기업 규모 분류
def classify_company_size(row):
    revenue = row['매출액']
    employees = row['종업원수']
    if pd.isna(revenue) or pd.isna(employees):
        return '정보부족'
    if revenue >= 100_000_000 and employees >= 100:
        return '대기업'
    elif revenue >= 50_000_000 or employees >= 50:
        return '중기업'
    elif revenue >= 10_000_000:
        return '소기업'
    else:
        return '영세기업'

df['기업규모'] = df.apply(classify_company_size, axis=1)

print(f"\n데이터 준비 완료: {df.shape[0]:,} 행 × {df.shape[1]:,} 열")
```

### 실습 1: CSV 파일로 저장

```python
print("\n" + "=" * 80)
print("📌 실습 1: CSV 파일로 저장")
print("=" * 80)

# 출력 디렉토리 생성
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# 1. 전체 데이터 저장 (결측치 처리 후)
print("\n[1] 전체 데이터 저장")
print("-" * 80)

# 결측치 처리
df_cleaned = df.copy()

# 수치형 결측치는 중앙값으로 채우기
df_cleaned['매출액'] = df_cleaned['매출액'].fillna(df_cleaned['매출액'].median())
df_cleaned['종업원수'] = df_cleaned['종업원수'].fillna(df_cleaned['종업원수'].median())

# 문자형 결측치는 '알수없음'으로 채우기
df_cleaned['지역'] = df_cleaned['지역'].fillna('알수없음')
df_cleaned['업종'] = df_cleaned['업종'].fillna('알수없음')
df_cleaned['수출여부'] = df_cleaned['수출여부'].fillna('N')
df_cleaned['인증보유'] = df_cleaned['인증보유'].fillna('없음')

# CSV 저장
output_file1 = os.path.join(output_dir, 'cleaned_business_data.csv')
df_cleaned.to_csv(output_file1, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file1}")
print(f"   파일 크기: {os.path.getsize(output_file1)/1024:.2f} KB")

# 2. 지역별 통계 저장
print("\n[2] 지역별 통계 저장")
print("-" * 80)

region_summary = df_cleaned.groupby('지역').agg({
    '매출액': ['count', 'sum', 'mean', 'median', 'std'],
    '종업원수': ['mean', 'median'],
    '수출여부': lambda x: (x == 'Y').sum()
}).round(0)

region_summary.columns = ['기업수', '총매출', '평균매출', '중앙매출', '매출표준편차', 
                          '평균종업원', '중앙종업원', '수출기업수']
region_summary['수출비율(%)'] = (region_summary['수출기업수'] / region_summary['기업수'] * 100).round(1)

output_file2 = os.path.join(output_dir, 'region_summary.csv')
region_summary.to_csv(output_file2, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file2}")

# 3. 업종별 통계 저장
print("\n[3] 업종별 통계 저장")
print("-" * 80)

industry_summary = df_cleaned.groupby('업종').agg({
    '매출액': ['count', 'mean', 'median'],
    '종업원수': ['mean', 'median'],
    '수출여부': lambda x: (x == 'Y').sum()
}).round(0)

industry_summary.columns = ['기업수', '평균매출', '중앙매출', '평균종업원', '중앙종업원', '수출기업수']

output_file3 = os.path.join(output_dir, 'industry_summary.csv')
industry_summary.to_csv(output_file3, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file3}")
```

### 실습 2: Excel 파일로 저장 (여러 시트)

```python
print("\n" + "=" * 80)
print("📌 실습 2: Excel 파일로 저장 (여러 시트)")
print("=" * 80)

# Excel Writer 사용하여 여러 시트 저장
output_excel = os.path.join(output_dir, 'business_analysis_report.xlsx')

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # 시트 1: 전체 데이터
    df_cleaned.to_excel(writer, sheet_name='전체데이터', index=False)
    
    # 시트 2: 지역별 통계
    region_summary.to_excel(writer, sheet_name='지역별통계')
    
    # 시트 3: 업종별 통계
    industry_summary.to_excel(writer, sheet_name='업종별통계')
    
    # 시트 4: 기업규모별 통계
    size_summary = df_cleaned.groupby('기업규모').agg({
        '매출액': ['count', 'mean'],
        '종업원수': 'mean',
        '수출여부': lambda x: (x == 'Y').sum()
    }).round(0)
    size_summary.columns = ['기업수', '평균매출', '평균종업원', '수출기업수']
    size_summary.to_excel(writer, sheet_name='기업규모별통계')
    
    # 시트 5: 월별 추이
    monthly_summary = df_cleaned.groupby('월').agg({
        '매출액': ['count', 'mean', 'sum'],
        '종업원수': 'mean'
    }).round(0)
    monthly_summary.columns = ['접수건수', '평균매출', '총매출', '평균종업원']
    monthly_summary.to_excel(writer, sheet_name='월별추이')

print(f"✅ 저장 완료: {output_excel}")
print(f"   파일 크기: {os.path.getsize(output_excel)/1024:.2f} KB")
print(f"   포함된 시트: 전체데이터, 지역별통계, 업종별통계, 기업규모별통계, 월별추이")
```

### 실습 3: JSON 형식으로 저장

```python
print("\n" + "=" * 80)
print("📌 실습 3: JSON 형식으로 저장")
print("=" * 80)

# 지역별 통계를 JSON으로 저장
output_json = os.path.join(output_dir, 'region_summary.json')

# DataFrame을 딕셔너리로 변환 후 JSON 저장
region_dict = region_summary.to_dict('index')

import json
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(region_dict, f, ensure_ascii=False, indent=2)

print(f"✅ 저장 완료: {output_json}")
print(f"   파일 크기: {os.path.getsize(output_json)/1024:.2f} KB")

# JSON 내용 일부 출력
print("\nJSON 파일 내용 (일부):")
print(json.dumps(list(region_dict.items())[:2], ensure_ascii=False, indent=2))
```

### 실습 4: 필터링된 데이터 저장

```python
print("\n" + "=" * 80)
print("📌 실습 4: 필터링된 데이터 저장")
print("=" * 80)

# 1. 고매출 기업 (1억 이상)
print("\n[1] 고매출 기업 (1억 이상)")
print("-" * 80)
high_revenue = df_cleaned[df_cleaned['매출액'] >= 100_000_000]
output_file = os.path.join(output_dir, 'high_revenue_companies.csv')
high_revenue.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file}")
print(f"   기업 수: {len(high_revenue):,}개")

# 2. 수출 기업
print("\n[2] 수출 기업")
print("-" * 80)
export_companies = df_cleaned[df_cleaned['수출여부'] == 'Y']
output_file = os.path.join(output_dir, 'export_companies.csv')
export_companies.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file}")
print(f"   기업 수: {len(export_companies):,}개")

# 3. 서울/경기 제조업
print("\n[3] 서울/경기 제조업")
print("-" * 80)
seoul_gyeonggi_manufacturing = df_cleaned[
    (df_cleaned['지역'].isin(['서울', '경기'])) & 
    (df_cleaned['업종'] == '제조업')
]
output_file = os.path.join(output_dir, 'seoul_gyeonggi_manufacturing.csv')
seoul_gyeonggi_manufacturing.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file}")
print(f"   기업 수: {len(seoul_gyeonggi_manufacturing):,}개")
```

### 실습 5: 날짜가 포함된 파일명으로 저장

```python
print("\n" + "=" * 80)
print("📌 실습 5: 날짜가 포함된 파일명으로 저장 (버전 관리)")
print("=" * 80)

# 현재 날짜/시간을 파일명에 포함
today = datetime.now().strftime('%Y%m%d')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 날짜 포함 파일명
output_file_dated = os.path.join(output_dir, f'cleaned_data_{today}.csv')
df_cleaned.to_csv(output_file_dated, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file_dated}")

# 타임스탬프 포함 파일명 (분석 시점 기록)
output_file_timestamped = os.path.join(output_dir, f'analysis_result_{timestamp}.csv')
region_summary.to_csv(output_file_timestamped, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_file_timestamped}")

print("\n💡 버전 관리 팁:")
print("- 파일명에 날짜 포함으로 히스토리 추적")
print("- 중요한 결과는 여러 버전 보관")
print("- 자동화 스크립트에서 유용")
```

### 실습 6: 압축 파일로 저장

```python
print("\n" + "=" * 80)
print("📌 실습 6: 압축 파일로 저장")
print("=" * 80)

# Gzip 압축으로 저장
output_file_compressed = os.path.join(output_dir, 'cleaned_data_compressed.csv.gz')
df_cleaned.to_csv(output_file_compressed, index=False, encoding='utf-8', compression='gzip')

# 파일 크기 비교
original_size = os.path.getsize(output_file1)
compressed_size = os.path.getsize(output_file_compressed)
compression_ratio = (1 - compressed_size / original_size) * 100

print(f"✅ 압축 저장 완료: {output_file_compressed}")
print(f"\n파일 크기 비교:")
print(f"  원본: {original_size/1024:.2f} KB")
print(f"  압축: {compressed_size/1024:.2f} KB")
print(f"  압축률: {compression_ratio:.1f}% 절약")

print("\n💡 압축 활용:")
print("- 대용량 데이터 저장 시 용량 절약")
print("- 네트워크 전송 시간 단축")
print("- Pandas에서 자동으로 압축 해제하여 읽기")
```

### 실습 7: 최종 분석 리포트 생성

```python
print("\n" + "=" * 80)
print("📌 실습 7: 최종 분석 리포트 생성")
print("=" * 80)

# 종합 리포트 작성
report = []
report.append("=" * 80)
report.append("📊 비즈니스 데이터 분석 최종 리포트")
report.append("=" * 80)
report.append(f"\n작성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
report.append(f"분석 기간: {df_cleaned['접수일자'].min().strftime('%Y-%m-%d')} ~ {df_cleaned['접수일자'].max().strftime('%Y-%m-%d')}")

report.append("\n" + "=" * 80)
report.append("1. 데이터 개요")
report.append("=" * 80)
report.append(f"총 기업 수: {len(df_cleaned):,}개")
report.append(f"분석 변수: {len(df_cleaned.columns)}개")
report.append(f"데이터 품질: 결측치 처리 완료")

report.append("\n" + "=" * 80)
report.append("2. 지역별 분석")
report.append("=" * 80)
top_region = region_summary['기업수'].idxmax()
report.append(f"최다 기업 지역: {top_region} ({region_summary.loc[top_region, '기업수']:,}개)")
top_revenue_region = region_summary['평균매출'].idxmax()
report.append(f"평균 매출 1위: {top_revenue_region} ({region_summary.loc[top_revenue_region, '평균매출']:,.0f}원)")
top_export_region = region_summary['수출비율(%)'].idxmax()
report.append(f"수출 비율 1위: {top_export_region} ({region_summary.loc[top_export_region, '수출비율(%)']:.1f}%)")

report.append("\n" + "=" * 80)
report.append("3. 업종별 분석")
report.append("=" * 80)
top_industry = industry_summary['기업수'].idxmax()
report.append(f"최다 기업 업종: {top_industry} ({industry_summary.loc[top_industry, '기업수']:,}개)")
top_revenue_industry = industry_summary['평균매출'].idxmax()
report.append(f"평균 매출 1위: {top_revenue_industry} ({industry_summary.loc[top_revenue_industry, '평균매출']:,.0f}원)")

report.append("\n" + "=" * 80)
report.append("4. 핵심 통계")
report.append("=" * 80)
report.append(f"전체 평균 매출: {df_cleaned['매출액'].mean():,.0f}원")
report.append(f"전체 중앙 매출: {df_cleaned['매출액'].median():,.0f}원")
report.append(f"전체 평균 종업원: {df_cleaned['종업원수'].mean():.1f}명")
report.append(f"수출 기업 비율: {(df_cleaned['수출여부']=='Y').sum()/len(df_cleaned)*100:.1f}%")

report.append("\n" + "=" * 80)
report.append("5. 주요 발견사항")
report.append("=" * 80)
report.append(f"- 매출액 1억 이상 기업: {len(high_revenue):,}개 ({len(high_revenue)/len(df_cleaned)*100:.1f}%)")
report.append(f"- 대기업: {(df_cleaned['기업규모']=='대기업').sum():,}개")
report.append(f"- 중기업: {(df_cleaned['기업규모']=='중기업').sum():,}개")
report.append(f"- 소기업: {(df_cleaned['기업규모']=='소기업').sum():,}개")

report.append("\n" + "=" * 80)
report.append("6. 저장된 파일 목록")
report.append("=" * 80)
report.append("- cleaned_business_data.csv: 전체 정제 데이터")
report.append("- region_summary.csv: 지역별 통계")
report.append("- industry_summary.csv: 업종별 통계")
report.append("- business_analysis_report.xlsx: 종합 리포트 (5개 시트)")
report.append("- high_revenue_companies.csv: 고매출 기업")
report.append("- export_companies.csv: 수출 기업")

report.append("\n" + "=" * 80)
report.append("분석 종료")
report.append("=" * 80)

# 리포트 출력
report_text = '\n'.join(report)
print(report_text)

# 리포트를 텍스트 파일로 저장
report_file = os.path.join(output_dir, 'analysis_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✅ 리포트 저장 완료: {report_file}")
```

### 실습 8: 저장된 파일 목록 확인

```python
print("\n" + "=" * 80)
print("📌 실습 8: 저장된 파일 목록 확인")
print("=" * 80)

# output 폴더의 모든 파일 목록
output_files = sorted(os.listdir(output_dir))

print(f"\n총 {len(output_files)}개 파일이 생성되었습니다:\n")

file_info = []
for filename in output_files:
    filepath = os.path.join(output_dir, filename)
    if os.path.isfile(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        file_info.append({
            '파일명': filename,
            '크기(KB)': f"{size_kb:.2f}",
            '형식': os.path.splitext(filename)[1]
        })

file_df = pd.DataFrame(file_info)
print(file_df.to_string(index=False))

print(f"\n총 용량: {sum([float(f['크기(KB)']) for f in file_info]):.2f} KB")
```

### 실습 9: 데이터 로드 테스트

```python
print("\n" + "=" * 80)
print("📌 실습 9: 저장된 데이터 로드 테스트")
print("=" * 80)

# CSV 파일 읽기 테스트
print("\n[1] CSV 파일 읽기")
print("-" * 80)
test_df = pd.read_csv(output_file1, encoding='utf-8-sig')
print(f"✅ 로드 성공: {test_df.shape}")
print(test_df.head(3))

# Excel 파일 읽기 테스트
print("\n[2] Excel 파일 읽기 (특정 시트)")
print("-" * 80)
test_region = pd.read_excel(output_excel, sheet_name='지역별통계')
print(f"✅ 로드 성공: {test_region.shape}")
print(test_region.head(3))

# 압축 파일 읽기 테스트
print("\n[3] 압축 파일 읽기")
print("-" * 80)
test_compressed = pd.read_csv(output_file_compressed, compression='gzip')
print(f"✅ 로드 성공: {test_compressed.shape}")

print("\n💡 모든 파일이 정상적으로 저장되고 읽힙니다!")
```

### 실습 10: 7교시 학습 마무리

```python
print("\n" + "=" * 80)
print("📌 실습 10: Pandas 실무 교육 과정 완료!")
print("=" * 80)

completion_summary = """
🎉 축하합니다! Pandas 데이터 정제 및 가공 실무 교육 7교시를 모두 완료하셨습니다!

📚 학습한 내용 요약:

1교시: 결측치 처리
  ✅ fillna(), dropna() 메서드
  ✅ 결측치 패턴 분석 및 시각화

2교시: groupby() 집계
  ✅ 카테고리별 그룹화 및 통계
  ✅ agg()로 다중 집계 함수 적용

3교시: 데이터 병합
  ✅ merge()로 inner/left/right/outer join
  ✅ 다중 키 병합

4교시: apply() 함수
  ✅ 사용자 정의 함수 적용
  ✅ lambda 함수 활용
  ✅ 복잡한 데이터 변환

5교시: 데이터 구조 파악
  ✅ 기본 정보 및 통계량 확인
  ✅ 결측치 및 이상치 탐지
  ✅ 데이터 품질 평가

6교시: 필터링 및 그룹핑
  ✅ 복합 조건 필터링 (AND/OR/NOT)
  ✅ 다중 그룹화 및 피벗 테이블
  ✅ 월별/지역별/업종별 분석

7교시: 결과 저장
  ✅ CSV, Excel, JSON 형식 저장
  ✅ 압축 및 버전 관리
  ✅ 최종 리포트 작성

💡 다음 단계 추천:
  - 실제 업무 데이터에 적용해보기
  - 더 복잡한 데이터 분석 프로젝트 수행
  - Pandas 공식 문서로 심화 학습
  - Matplotlib/Seaborn으로 시각화 강화
  - SQL과 Pandas 연계 학습

🌟 수고하셨습니다! 🌟
"""

print(completion_summary)

# 완료 증명서 (텍스트 파일)
certificate = f"""
{'=' * 80}
                    📜 수료증 📜
{'=' * 80}

                Pandas 활용을 통한
        데이터 정제 및 가공 실무 교육 (재직자 과정)

본 증명서는 아래 교육생이 Pandas 데이터 분석 실무 교육 과정
(총 7교시)을 성실히 이수하였음을 증명합니다.

교육 과정:
  - 1교시: 결측치 처리
  - 2교시: groupby() 집계
  - 3교시: 데이터 병합
  - 4교시: apply() 함수
  - 5교시: 데이터 구조 파악
  - 6교시: 필터링 및 그룹핑
  - 7교시: 결과 저장

완료일: {datetime.now().strftime('%Y년 %m월 %d일')}

{'=' * 80}
"""

certificate_file = os.path.join(output_dir, 'completion_certificate.txt')
with open(certificate_file, 'w', encoding='utf-8') as f:
    f.write(certificate)

print(f"\n✅ 수료증 저장 완료: {certificate_file}")
```

---

## 🧩 퀴즈

### 문제 1
CSV 파일 저장 시 한글이 깨지지 않으려면?

<details>
<summary>정답 보기</summary>

**정답: `encoding='utf-8-sig'` 또는 `encoding='cp949'` 사용**

```python
# 권장 방법 (엑셀에서도 잘 열림)
df.to_csv('파일.csv', encoding='utf-8-sig', index=False)

# Windows 환경 (구형 엑셀)
df.to_csv('파일.csv', encoding='cp949', index=False)
```

**BOM(Byte Order Mark) 차이:**
- `utf-8`: BOM 없음 → 일부 프로그램에서 한글 깨짐
- `utf-8-sig`: BOM 있음 → 대부분 프로그램에서 정상 표시
- `cp949`: Windows 한글 인코딩 → 구형 엑셀에서 안전

💡 **실무 팁**: 엑셀로 열 파일은 `utf-8-sig` 사용!
</details>

---

### 문제 2
DataFrame을 Excel 파일의 여러 시트에 저장하려면?

<details>
<summary>정답 보기</summary>

**정답: `ExcelWriter` 사용**

```python
with pd.ExcelWriter('report.xlsx', engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='시트1', index=False)
    df2.to_excel(writer, sheet_name='시트2', index=False)
    df3.to_excel(writer, sheet_name='시트3', index=False)
```

**주요 파라미터:**
- `engine='openpyxl'`: xlsx 파일 생성 (권장)
- `engine='xlsxwriter'`: 복잡한 서식 지원
- `sheet_name`: 시트 이름 지정
- `index=False`: 인덱스 컬럼 제외

💡 **실무 활용**: 하나의 Excel 파일에 여러 분석 결과 정리!
</details>

---

### 문제 3
대용량 데이터를 효율적으로 저장하는 방법은?

<details>
<summary>정답 보기</summary>

**답변: 압축 사용 또는 Parquet 형식**

**방법 1: Gzip 압축**
```python
# 저장 (압축)
df.to_csv('data.csv.gz', compression='gzip', index=False)

# 읽기 (자동 압축 해제)
df = pd.read_csv('data.csv.gz')
```

**방법 2: Parquet 형식 (빅데이터 권장)**
```python
# 저장
df.to_parquet('data.parquet', compression='snappy')

# 읽기
df = pd.read_parquet('data.parquet')
```

**형식별 비교:**

| 형식 | 크기 | 속도 | 범용성 |
|------|------|------|--------|
| CSV | 100% | 느림 | 최고 |
| CSV.GZ | 20% | 보통 | 높음 |
| Parquet | 10% | 빠름 | 보통 |

💡 **실무 선택:**
- 교환/공유: CSV
- 저장 용량 중요: CSV.GZ
- 빅데이터/속도: Parquet
</details>

---

## ✅ 7교시 학습 완료 체크리스트

- [ ] CSV 파일 저장 (인코딩 처리)
- [ ] Excel 파일 저장 (여러 시트)
- [ ] JSON 형식 저장
- [ ] 필터링된 데이터 저장
- [ ] 날짜 포함 파일명으로 저장
- [ ] 압축 파일 저장
- [ ] 최종 분석 리포트 작성
- [ ] 저장된 파일 로드 테스트
- [ ] 퀴즈 3문제 모두 풀이 완료
- [ ] 🎉 전체 7교시 과정 완료!

---

## 🎓 최종 평가

### 전체 교육 과정 이해도 평가

- [ ] 1교시 내용을 자신 있게 설명할 수 있다
- [ ] 2교시 내용을 자신 있게 설명할 수 있다
- [ ] 3교시 내용을 자신 있게 설명할 수 있다
- [ ] 4교시 내용을 자신 있게 설명할 수 있다
- [ ] 5교시 내용을 자신 있게 설명할 수 있다
- [ ] 6교시 내용을 자신 있게 설명할 수 있다
- [ ] 7교시 내용을 자신 있게 설명할 수 있다
- [ ] 실무 데이터에 즉시 적용할 수 있다

---

**이전 학습:** [6교시 - 필터링 및 그룹핑](./6_filter_group.md)  
**첫 학습으로:** [1교시 - 결측치 처리](./1_missing_values.md)

**🎉 전체 과정 완료일:** _____________  
**총 소요 시간:** _____________  
**전체 만족도 (1~5):** ⭐⭐⭐⭐⭐

---

## 📚 추가 학습 자료

### 공식 문서
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/index.html)

### 추천 도서
- "Python for Data Analysis" by Wes McKinney
- "Pandas Cookbook" by Theodore Petrou

### 온라인 강좌
- [Kaggle Learn - Pandas](https://www.kaggle.com/learn/pandas)
- [DataCamp - Pandas Courses](https://www.datacamp.com/courses/topic:pandas)

**감사합니다! 🙏**
