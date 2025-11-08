"""
2교시 예제: Pandas 기본 데이터 분석
"""

import pandas as pd

def main():
    # 데이터 불러오기
    df = pd.read_csv('../data/sales_data.csv')
    
    # 기본 정보
    print("=== 데이터 기본 정보 ===")
    print(f"데이터 크기: {df.shape}")
    print(f"\n상위 5개 데이터:")
    print(df.head())
    
    # 카테고리별 매출
    print("\n=== 카테고리별 총 매출 ===")
    category_sales = df.groupby('product_category')['final_amount'].sum()
    print(category_sales.sort_values(ascending=False))
    
    # 고액 거래 필터링
    high_value = df[df['final_amount'] >= 1000000]
    print(f"\n100만원 이상 거래: {len(high_value)}건")
    
if __name__ == "__main__":
    main()
