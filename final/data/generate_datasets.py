"""
실습용 데이터셋 자동 생성 스크립트
Python 데이터 분석 실무 종합 과정용
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# 랜덤 시드 고정 (재현 가능성)
np.random.seed(42)
random.seed(42)

def generate_sales_data(n_records=1000):
    """전자상거래 판매 데이터 생성"""
    print("📊 판매 데이터 생성 중...")
    
    # 날짜 생성 (최근 2년)
    start_date = datetime.now() - timedelta(days=730)
    dates = [start_date + timedelta(days=x) for x in range(730)]
    
    # 기본 데이터
    categories = ['전자제품', '의류', '식품', '가구', '도서', '화장품', '스포츠', '완구']
    regions = ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산']
    payment_methods = ['신용카드', '체크카드', '계좌이체', '간편결제', '현금']
    
    data = {
        'order_id': [f'ORD{str(i).zfill(6)}' for i in range(1, n_records + 1)],
        'order_date': [random.choice(dates).strftime('%Y-%m-%d') for _ in range(n_records)],
        'customer_id': [f'CUST{str(random.randint(1, 500)).zfill(4)}' for _ in range(n_records)],
        'product_category': [random.choice(categories) for _ in range(n_records)],
        'product_name': [f'상품_{random.randint(1, 100)}' for _ in range(n_records)],
        'quantity': np.random.randint(1, 10, n_records),
        'unit_price': np.random.randint(10000, 500000, n_records),
        'region': [random.choice(regions) for _ in range(n_records)],
        'payment_method': [random.choice(payment_methods) for _ in range(n_records)],
        'customer_age': np.random.randint(20, 70, n_records),
        'customer_gender': [random.choice(['남성', '여성']) for _ in range(n_records)],
    }
    
    df = pd.DataFrame(data)
    
    # 총 판매액 계산
    df['total_amount'] = df['quantity'] * df['unit_price']
    
    # 할인 적용 (20% 확률로 10-30% 할인)
    df['discount_rate'] = 0
    discount_mask = np.random.random(n_records) < 0.2
    df.loc[discount_mask, 'discount_rate'] = np.random.randint(10, 31, discount_mask.sum())
    df['discount_amount'] = (df['total_amount'] * df['discount_rate'] / 100).astype(int)
    df['final_amount'] = df['total_amount'] - df['discount_amount']
    
    # 결측치 의도적으로 추가 (약 5%)
    missing_indices = np.random.choice(df.index, size=int(n_records * 0.05), replace=False)
    df.loc[missing_indices, 'customer_age'] = np.nan
    
    missing_indices2 = np.random.choice(df.index, size=int(n_records * 0.03), replace=False)
    df.loc[missing_indices2, 'region'] = np.nan
    
    return df


def generate_customer_data(n_customers=500):
    """고객 정보 데이터 생성"""
    print("👥 고객 데이터 생성 중...")
    
    member_types = ['일반', '실버', '골드', 'VIP']
    occupations = ['회사원', '자영업', '학생', '프리랜서', '주부', '기타']
    
    data = {
        'customer_id': [f'CUST{str(i).zfill(4)}' for i in range(1, n_customers + 1)],
        'customer_name': [f'고객_{i}' for i in range(1, n_customers + 1)],
        'email': [f'customer{i}@example.com' for i in range(1, n_customers + 1)],
        'phone': [f'010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}' for _ in range(n_customers)],
        'registration_date': [(datetime.now() - timedelta(days=random.randint(1, 1095))).strftime('%Y-%m-%d') 
                              for _ in range(n_customers)],
        'member_type': [random.choice(member_types) for _ in range(n_customers)],
        'occupation': [random.choice(occupations) for _ in range(n_customers)],
        'total_purchases': np.random.randint(0, 100, n_customers),
        'total_spent': np.random.randint(0, 10000000, n_customers),
    }
    
    df = pd.DataFrame(data)
    
    # VIP 고객은 구매액이 많도록 조정
    vip_mask = df['member_type'] == 'VIP'
    df.loc[vip_mask, 'total_spent'] = np.random.randint(5000000, 20000000, vip_mask.sum())
    df.loc[vip_mask, 'total_purchases'] = np.random.randint(50, 200, vip_mask.sum())
    
    return df


def generate_product_data(n_products=200):
    """상품 정보 데이터 생성"""
    print("📦 상품 데이터 생성 중...")
    
    categories = ['전자제품', '의류', '식품', '가구', '도서', '화장품', '스포츠', '완구']
    brands = ['브랜드A', '브랜드B', '브랜드C', '브랜드D', '브랜드E', '자체브랜드']
    
    data = {
        'product_id': [f'PROD{str(i).zfill(4)}' for i in range(1, n_products + 1)],
        'product_name': [f'상품_{i}' for i in range(1, n_products + 1)],
        'category': [random.choice(categories) for _ in range(n_products)],
        'brand': [random.choice(brands) for _ in range(n_products)],
        'price': np.random.randint(10000, 500000, n_products),
        'cost': np.random.randint(5000, 300000, n_products),
        'stock_quantity': np.random.randint(0, 1000, n_products),
        'weight_kg': np.round(np.random.uniform(0.1, 50.0, n_products), 2),
        'rating': np.round(np.random.uniform(3.0, 5.0, n_products), 1),
        'review_count': np.random.randint(0, 500, n_products),
    }
    
    df = pd.DataFrame(data)
    
    # 원가가 판매가보다 높으면 수정
    df.loc[df['cost'] >= df['price'], 'cost'] = (df.loc[df['cost'] >= df['price'], 'price'] * 0.6).astype(int)
    
    return df


def generate_transaction_data(n_transactions=2000):
    """거래 내역 데이터 생성 (시계열)"""
    print("💳 거래 내역 데이터 생성 중...")
    
    # 날짜 생성 (최근 1년)
    start_date = datetime.now() - timedelta(days=365)
    
    data = {
        'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(1, n_transactions + 1)],
        'transaction_datetime': [(start_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )).strftime('%Y-%m-%d %H:%M:%S') for _ in range(n_transactions)],
        'customer_id': [f'CUST{str(random.randint(1, 500)).zfill(4)}' for _ in range(n_transactions)],
        'amount': np.random.randint(10000, 1000000, n_transactions),
        'status': [random.choice(['완료', '완료', '완료', '완료', '취소', '환불']) for _ in range(n_transactions)],
    }
    
    df = pd.DataFrame(data)
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
    df = df.sort_values('transaction_datetime').reset_index(drop=True)
    
    return df


def main():
    """메인 함수: 모든 데이터셋 생성 및 저장"""
    print("="*60)
    print("🚀 실습용 데이터셋 생성 시작")
    print("="*60)
    
    # 현재 스크립트의 디렉토리 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 판매 데이터
    sales_df = generate_sales_data(n_records=1000)
    sales_path = os.path.join(script_dir, 'sales_data.csv')
    sales_df.to_csv(sales_path, index=False, encoding='utf-8-sig')
    print(f"✅ 판매 데이터 저장 완료: {sales_path}")
    print(f"   - 레코드 수: {len(sales_df)}")
    print(f"   - 컬럼 수: {len(sales_df.columns)}")
    print()
    
    # 2. 고객 데이터
    customer_df = generate_customer_data(n_customers=500)
    customer_path = os.path.join(script_dir, 'customer_data.csv')
    customer_df.to_csv(customer_path, index=False, encoding='utf-8-sig')
    print(f"✅ 고객 데이터 저장 완료: {customer_path}")
    print(f"   - 레코드 수: {len(customer_df)}")
    print(f"   - 컬럼 수: {len(customer_df.columns)}")
    print()
    
    # 3. 상품 데이터
    product_df = generate_product_data(n_products=200)
    product_path = os.path.join(script_dir, 'product_data.csv')
    product_df.to_csv(product_path, index=False, encoding='utf-8-sig')
    print(f"✅ 상품 데이터 저장 완료: {product_path}")
    print(f"   - 레코드 수: {len(product_df)}")
    print(f"   - 컬럼 수: {len(product_df.columns)}")
    print()
    
    # 4. 거래 내역 데이터
    transaction_df = generate_transaction_data(n_transactions=2000)
    transaction_path = os.path.join(script_dir, 'transaction_data.csv')
    transaction_df.to_csv(transaction_path, index=False, encoding='utf-8-sig')
    print(f"✅ 거래 내역 데이터 저장 완료: {transaction_path}")
    print(f"   - 레코드 수: {len(transaction_df)}")
    print(f"   - 컬럼 수: {len(transaction_df.columns)}")
    print()
    
    print("="*60)
    print("🎉 모든 데이터셋 생성 완료!")
    print("="*60)
    print("\n📁 생성된 파일:")
    print(f"  1. {os.path.basename(sales_path)} - 판매 데이터")
    print(f"  2. {os.path.basename(customer_path)} - 고객 데이터")
    print(f"  3. {os.path.basename(product_path)} - 상품 데이터")
    print(f"  4. {os.path.basename(transaction_path)} - 거래 내역 데이터")
    print("\n💡 데이터 미리보기:")
    print("\n[판매 데이터 샘플]")
    print(sales_df.head(3))
    print("\n[고객 데이터 샘플]")
    print(customer_df.head(3))
    

if __name__ == "__main__":
    main()
