"""
1교시 예제: Python 핵심 기능
"""

# 매출 데이터 분석 함수
def analyze_sales(sales_data):
    """매출 데이터를 분석하는 함수"""
    total = sum(sales_data)
    average = total / len(sales_data)
    maximum = max(sales_data)
    minimum = min(sales_data)
    
    return {
        'total': total,
        'average': average,
        'max': maximum,
        'min': minimum
    }

# 실행 예제
if __name__ == "__main__":
    monthly_sales = [3200000, 4100000, 3800000, 5200000, 4500000]
    result = analyze_sales(monthly_sales)
    
    print("=== 매출 분석 결과 ===")
    print(f"총 매출: {result['total']:,}원")
    print(f"평균 매출: {result['average']:,.0f}원")
    print(f"최고 매출: {result['max']:,}원")
    print(f"최저 매출: {result['min']:,}원")
