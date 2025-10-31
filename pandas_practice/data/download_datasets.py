"""
데이터셋 다운로드 스크립트
Pandas 실무 교육을 위한 공개 데이터셋을 준비합니다.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
import os

print("=" * 80)
print("📦 Pandas 실무 교육용 데이터셋 다운로드 시작")
print("=" * 80)

# 현재 스크립트 위치 확인
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\n📁 데이터 저장 위치: {current_dir}")

# ============================================================
# 1교시: Tips 데이터셋 (Seaborn 내장)
# ============================================================
print("\n[1/5] Tips 데이터셋 다운로드 중...")
tips = sns.load_dataset('tips')
tips_file = os.path.join(current_dir, 'tips.csv')
tips.to_csv(tips_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {tips_file}")
print(f"   - 행 수: {len(tips)}, 열 수: {len(tips.columns)}")

# ============================================================
# 2교시: 판매 데이터셋 생성
# ============================================================
print("\n[2/5] 판매 데이터셋 생성 중...")
np.random.seed(42)
sales_data = pd.DataFrame({
    '제품ID': [f'P{str(i).zfill(4)}' for i in range(1, 201)],
    '제품명': np.random.choice(['노트북', '마우스', '키보드', '모니터', '헤드셋'], 200),
    '카테고리': np.random.choice(['전자제품', '컴퓨터부품', '액세서리'], 200),
    '판매가격': np.random.randint(10000, 2000000, 200),
    '판매량': np.random.randint(1, 100, 200),
    '재고': np.random.randint(0, 500, 200),
    '지역': np.random.choice(['서울', '부산', '대구', '인천', '광주'], 200),
    '판매일': pd.date_range('2024-01-01', periods=200, freq='D')
})
sales_file = os.path.join(current_dir, 'sales_data.csv')
sales_data.to_csv(sales_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {sales_file}")
print(f"   - 행 수: {len(sales_data)}, 열 수: {len(sales_data.columns)}")

# ============================================================
# 3교시: 고객 및 주문 데이터셋 생성
# ============================================================
print("\n[3/5] 고객 및 주문 데이터셋 생성 중...")

# 고객 데이터
customers = pd.DataFrame({
    '고객ID': [f'C{str(i).zfill(4)}' for i in range(1, 101)],
    '고객명': [f'고객{i}' for i in range(1, 101)],
    '연령대': np.random.choice(['20대', '30대', '40대', '50대', '60대'], 100),
    '등급': np.random.choice(['일반', '실버', '골드', 'VIP'], 100),
    '가입일': pd.date_range('2020-01-01', periods=100, freq='10D')
})
customers_file = os.path.join(current_dir, 'customers.csv')
customers.to_csv(customers_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {customers_file}")

# 주문 데이터
orders = pd.DataFrame({
    '주문ID': [f'O{str(i).zfill(5)}' for i in range(1, 301)],
    '고객ID': np.random.choice([f'C{str(i).zfill(4)}' for i in range(1, 101)], 300),
    '주문금액': np.random.randint(10000, 500000, 300),
    '주문일': pd.date_range('2024-01-01', periods=300, freq='D'),
    '배송상태': np.random.choice(['배송완료', '배송중', '주문완료'], 300)
})
orders_file = os.path.join(current_dir, 'orders.csv')
orders.to_csv(orders_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {orders_file}")

# ============================================================
# 4교시: 제품 가격 데이터셋 생성 (달러 기준)
# ============================================================
print("\n[4/5] 제품 가격 데이터셋 생성 중...")
products = pd.DataFrame({
    '제품코드': [f'PROD{str(i).zfill(3)}' for i in range(1, 51)],
    '제품명': [f'상품{i}' for i in range(1, 51)],
    '가격_달러': np.random.uniform(10, 500, 50).round(2),
    '카테고리': np.random.choice(['의류', '가전', '식품', '도서', '가구'], 50),
    '재고수량': np.random.randint(0, 100, 50)
})
products_file = os.path.join(current_dir, 'products_usd.csv')
products.to_csv(products_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {products_file}")
print(f"   - 행 수: {len(products)}, 열 수: {len(products.columns)}")

# ============================================================
# 5-7교시: 공공데이터 스타일 데이터셋 생성 (결측치 포함)
# ============================================================
print("\n[5/5] 공공데이터 스타일 데이터셋 생성 중...")
np.random.seed(123)

# 결측치가 포함된 실전 데이터
n_rows = 500
public_data = pd.DataFrame({
    '일련번호': range(1, n_rows + 1),
    '접수일자': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
    '지역': np.random.choice(['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산', None], n_rows, p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
    '업종': np.random.choice(['제조업', '서비스업', '유통업', '건설업', None], n_rows, p=[0.3, 0.3, 0.25, 0.1, 0.05]),
    '매출액': np.random.randint(1000000, 100000000, n_rows),
    '종업원수': np.random.randint(5, 500, n_rows),
    '설립연도': np.random.randint(1990, 2024, n_rows),
    '수출여부': np.random.choice(['Y', 'N', None], n_rows, p=[0.3, 0.6, 0.1]),
    '인증보유': np.random.choice(['ISO9001', 'ISO14001', '없음', None], n_rows, p=[0.2, 0.15, 0.55, 0.1])
})

# 일부 매출액과 종업원수에 결측치 추가
missing_indices = np.random.choice(n_rows, size=50, replace=False)
public_data.loc[missing_indices[:25], '매출액'] = np.nan
public_data.loc[missing_indices[25:], '종업원수'] = np.nan

public_file = os.path.join(current_dir, 'public_business_data.csv')
public_data.to_csv(public_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {public_file}")
print(f"   - 행 수: {len(public_data)}, 열 수: {len(public_data.columns)}")
print(f"   - 결측치 포함: 지역, 업종, 매출액, 종업원수, 수출여부, 인증보유")

# ============================================================
# 완료 메시지
# ============================================================
print("\n" + "=" * 80)
print("✅ 모든 데이터셋 다운로드 및 생성 완료!")
print("=" * 80)
print("\n📂 생성된 파일 목록:")
print("  1. tips.csv - 팁 데이터 (1교시)")
print("  2. sales_data.csv - 판매 데이터 (2교시)")
print("  3. customers.csv - 고객 데이터 (3교시)")
print("  4. orders.csv - 주문 데이터 (3교시)")
print("  5. products_usd.csv - 제품 가격 데이터 (4교시)")
print("  6. public_business_data.csv - 공공데이터 (5-7교시)")
print("\n🎓 이제 notebooks 폴더의 각 교시 파일을 순서대로 학습하세요!")
print("=" * 80)
