# 1교시: Python 핵심 복습 & 데이터 구조

> **학습 시간**: 1시간  
> **난이도**: ⭐⭐  
> **목표**: 데이터 분석에 필요한 Python 핵심 문법을 복습하고 실무 활용법을 익힙니다.

---

## 📚 학습 내용

1. Python 기본 자료형
2. 조건문과 반복문
3. 함수 정의와 활용
4. List Comprehension
5. Lambda 함수
6. 실무 예제

---

## 1. Python 기본 자료형

### 1.1 리스트 (List)

**리스트는 순서가 있는 데이터 모음**입니다. 대괄호 `[]`를 사용합니다.

```python
# 리스트 생성
sales = [150000, 200000, 180000, 220000, 190000]
products = ['노트북', '마우스', '키보드', '모니터', '헤드셋']

# 인덱싱 (0부터 시작)
print(f"첫 번째 매출: {sales[0]}원")  # 150000원
print(f"마지막 매출: {sales[-1]}원")  # 190000원

# 슬라이싱
print(f"처음 3개 매출: {sales[:3]}")  # [150000, 200000, 180000]
print(f"2번째~4번째 매출: {sales[1:4]}")  # [200000, 180000, 220000]

# 리스트 메서드
sales.append(250000)  # 끝에 추가
print(f"추가 후: {sales}")

total = sum(sales)
average = total / len(sales)
print(f"총 매출: {total:,}원, 평균 매출: {average:,.0f}원")
```

### 1.2 딕셔너리 (Dictionary)

**딕셔너리는 키-값 쌍으로 저장하는 자료구조**입니다. 중괄호 `{}`를 사용합니다.

```python
# 딕셔너리 생성
employee = {
    'name': '김철수',
    'age': 32,
    'department': '영업팀',
    'position': '대리',
    'salary': 3500000
}

# 값 접근
print(f"이름: {employee['name']}")
print(f"부서: {employee['department']}")
print(f"급여: {employee['salary']:,}원")

# 값 수정
employee['salary'] = 3800000
print(f"인상된 급여: {employee['salary']:,}원")

# 새 키-값 추가
employee['email'] = 'kim@company.com'

# 키 존재 여부 확인
if 'email' in employee:
    print(f"이메일: {employee['email']}")

# 모든 키와 값 순회
for key, value in employee.items():
    print(f"{key}: {value}")
```

### 1.3 튜플 (Tuple)

**튜플은 수정 불가능한 리스트**입니다. 소괄호 `()`를 사용합니다.

```python
# 튜플 생성 (좌표, 날짜 등 변경되지 않아야 하는 데이터)
coordinates = (37.5665, 126.9780)  # 서울 좌표
date = (2024, 11, 7)

# 값 접근
latitude, longitude = coordinates
print(f"위도: {latitude}, 경도: {longitude}")

year, month, day = date
print(f"날짜: {year}년 {month}월 {day}일")
```

### 1.4 집합 (Set)

**집합은 중복을 허용하지 않는 자료구조**입니다.

```python
# 집합 생성
customers_monday = {'고객A', '고객B', '고객C', '고객D'}
customers_tuesday = {'고객B', '고객D', '고객E', '고객F'}

# 교집합 (양일 모두 방문한 고객)
both_days = customers_monday & customers_tuesday
print(f"양일 모두 방문: {both_days}")

# 합집합 (한 번이라도 방문한 고객)
any_day = customers_monday | customers_tuesday
print(f"한 번이라도 방문: {any_day}")

# 차집합 (월요일만 방문)
monday_only = customers_monday - customers_tuesday
print(f"월요일만 방문: {monday_only}")
```

---

## 2. 조건문과 반복문

### 2.1 조건문 (if-elif-else)

```python
# 매출 등급 분류
sales_amount = 5000000

if sales_amount >= 10000000:
    grade = 'S'
    bonus_rate = 0.15
elif sales_amount >= 5000000:
    grade = 'A'
    bonus_rate = 0.10
elif sales_amount >= 3000000:
    grade = 'B'
    bonus_rate = 0.05
else:
    grade = 'C'
    bonus_rate = 0.02

bonus = sales_amount * bonus_rate
print(f"매출: {sales_amount:,}원")
print(f"등급: {grade}, 보너스율: {bonus_rate*100}%")
print(f"보너스: {bonus:,.0f}원")
```

### 2.2 for 반복문

```python
# 리스트 순회
products = ['노트북', '마우스', '키보드', '모니터']
prices = [1200000, 25000, 85000, 350000]

print("=== 상품 목록 ===")
for i in range(len(products)):
    print(f"{i+1}. {products[i]}: {prices[i]:,}원")

# 딕셔너리 순회
sales_by_region = {
    '서울': 5000000,
    '부산': 3200000,
    '대구': 2800000,
    '인천': 3500000
}

print("\n=== 지역별 매출 ===")
for region, amount in sales_by_region.items():
    print(f"{region}: {amount:,}원")
    
# enumerate로 인덱스와 값 동시 접근
print("\n=== 상품 번호와 함께 출력 ===")
for idx, product in enumerate(products, start=1):
    print(f"{idx}. {product}")
```

### 2.3 while 반복문

```python
# 목표 매출 달성까지 반복
current_sales = 0
target_sales = 10000000
day = 0

while current_sales < target_sales:
    day += 1
    daily_sales = 2000000  # 일 매출
    current_sales += daily_sales
    print(f"{day}일차: {current_sales:,}원 (목표까지 {target_sales - current_sales:,}원 남음)")

print(f"\n목표 달성! 총 {day}일 소요")
```

---

## 3. 함수 정의와 활용

### 3.1 기본 함수

```python
def calculate_tax(amount, tax_rate=0.1):
    """
    세금 계산 함수
    
    Args:
        amount: 금액
        tax_rate: 세율 (기본값 10%)
    
    Returns:
        세후 금액
    """
    tax = amount * tax_rate
    after_tax = amount - tax
    return after_tax, tax

# 함수 호출
price = 1000000
final_price, tax_amount = calculate_tax(price)
print(f"상품가: {price:,}원")
print(f"세금: {tax_amount:,.0f}원")
print(f"최종가: {final_price:,.0f}원")

# 세율 변경
final_price2, tax_amount2 = calculate_tax(price, tax_rate=0.13)
print(f"\n세율 13% 적용시 최종가: {final_price2:,.0f}원")
```

### 3.2 여러 값 반환

```python
def analyze_sales(sales_list):
    """매출 데이터 분석"""
    total = sum(sales_list)
    average = total / len(sales_list)
    maximum = max(sales_list)
    minimum = min(sales_list)
    
    return {
        'total': total,
        'average': average,
        'max': maximum,
        'min': minimum,
        'count': len(sales_list)
    }

# 함수 사용
monthly_sales = [3200000, 4100000, 3800000, 5200000, 4500000]
result = analyze_sales(monthly_sales)

print("=== 매출 분석 결과 ===")
print(f"총 매출: {result['total']:,}원")
print(f"평균 매출: {result['average']:,.0f}원")
print(f"최고 매출: {result['max']:,}원")
print(f"최저 매출: {result['min']:,}원")
print(f"데이터 수: {result['count']}개")
```

---

## 4. List Comprehension (리스트 컴프리헨션)

**리스트를 간결하게 생성하는 Python의 강력한 기능**입니다.

### 4.1 기본 사용법

```python
# 기존 방식
squares = []
for i in range(1, 11):
    squares.append(i ** 2)
print(f"제곱 수: {squares}")

# List Comprehension 방식 (훨씬 간결!)
squares_comp = [i ** 2 for i in range(1, 11)]
print(f"제곱 수 (컴프리헨션): {squares_comp}")
```

### 4.2 조건 포함

```python
# 짝수만 필터링
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [n for n in numbers if n % 2 == 0]
print(f"짝수: {even_numbers}")

# 매출 데이터에서 목표 달성한 값만 추출
sales = [2500000, 4200000, 3100000, 5800000, 2900000]
target = 3000000
achieved = [s for s in sales if s >= target]
print(f"목표 달성 매출: {achieved}")
print(f"달성률: {len(achieved)}/{len(sales)} ({len(achieved)/len(sales)*100:.1f}%)")
```

### 4.3 데이터 변환

```python
# 상품명을 대문자로 변환
products = ['laptop', 'mouse', 'keyboard', 'monitor']
products_upper = [p.upper() for p in products]
print(f"대문자 변환: {products_upper}")

# 가격에 할인 적용
prices = [100000, 50000, 80000, 120000]
discount_rate = 0.2
discounted_prices = [int(p * (1 - discount_rate)) for p in prices]
print(f"원가: {prices}")
print(f"할인가 (20% 할인): {discounted_prices}")

# 문자열에서 숫자만 추출
order_ids = ['ORD001', 'ORD002', 'ORD003', 'ORD004']
order_numbers = [int(oid.replace('ORD', '')) for oid in order_ids]
print(f"주문번호: {order_numbers}")
```

### 4.4 중첩 리스트 평탄화

```python
# 2차원 리스트를 1차원으로
sales_matrix = [
    [100, 200, 150],  # 1주차
    [180, 220, 210],  # 2주차
    [190, 230, 200]   # 3주차
]

all_sales = [sale for week in sales_matrix for sale in week]
print(f"전체 매출 데이터: {all_sales}")
print(f"총 매출: {sum(all_sales):,}원")
```

---

## 5. Lambda 함수

**Lambda는 이름 없는 익명 함수**로, 간단한 함수를 한 줄로 작성할 때 사용합니다.

### 5.1 기본 사용법

```python
# 일반 함수
def add(x, y):
    return x + y

# Lambda 함수
add_lambda = lambda x, y: x + y

print(f"일반 함수: {add(10, 20)}")
print(f"Lambda 함수: {add_lambda(10, 20)}")
```

### 5.2 실무 활용 - 정렬

```python
# 직원 리스트 (딕셔너리)
employees = [
    {'name': '김철수', 'age': 32, 'salary': 3500000},
    {'name': '이영희', 'age': 28, 'salary': 4200000},
    {'name': '박민수', 'age': 35, 'salary': 3800000},
    {'name': '정수진', 'age': 30, 'salary': 4500000}
]

# 급여 순으로 정렬
sorted_by_salary = sorted(employees, key=lambda x: x['salary'], reverse=True)

print("=== 급여 순 정렬 ===")
for emp in sorted_by_salary:
    print(f"{emp['name']}: {emp['salary']:,}원")

# 나이 순으로 정렬
sorted_by_age = sorted(employees, key=lambda x: x['age'])

print("\n=== 나이 순 정렬 ===")
for emp in sorted_by_age:
    print(f"{emp['name']}: {emp['age']}세")
```

### 5.3 map, filter와 함께 사용

```python
# map: 모든 요소에 함수 적용
prices = [10000, 20000, 30000, 40000]
prices_with_vat = list(map(lambda x: x * 1.1, prices))
print(f"원가: {prices}")
print(f"VAT 포함: {prices_with_vat}")

# filter: 조건에 맞는 요소만 선택
sales = [2500000, 4200000, 3100000, 5800000, 2900000]
high_sales = list(filter(lambda x: x >= 4000000, sales))
print(f"\n전체 매출: {sales}")
print(f"400만원 이상: {high_sales}")
```

---

## 6. 실무 예제

### 예제 1: 매출 데이터 분석 함수

```python
def analyze_monthly_sales(sales_data):
    """
    월별 매출 데이터를 분석하는 함수
    
    Args:
        sales_data: {'월': 매출액} 형태의 딕셔너리
    
    Returns:
        분석 결과 딕셔너리
    """
    amounts = list(sales_data.values())
    
    # 기본 통계
    total = sum(amounts)
    average = total / len(amounts)
    max_month = max(sales_data, key=sales_data.get)
    min_month = min(sales_data, key=sales_data.get)
    
    # 목표 달성 여부 (월 평균 500만원)
    target = 5000000
    achieved_months = [month for month, amount in sales_data.items() if amount >= target]
    
    return {
        '총 매출': total,
        '평균 매출': average,
        '최고 매출 월': max_month,
        '최고 매출액': sales_data[max_month],
        '최저 매출 월': min_month,
        '최저 매출액': sales_data[min_month],
        '목표 달성 월': achieved_months,
        '달성률': len(achieved_months) / len(sales_data) * 100
    }

# 실행
sales_2024 = {
    '1월': 4200000,
    '2월': 3800000,
    '3월': 5600000,
    '4월': 5100000,
    '5월': 4800000,
    '6월': 6200000
}

result = analyze_monthly_sales(sales_2024)

print("=== 2024년 상반기 매출 분석 ===")
print(f"총 매출: {result['총 매출']:,}원")
print(f"평균 매출: {result['평균 매출']:,.0f}원")
print(f"최고 매출: {result['최고 매출 월']} ({result['최고 매출액']:,}원)")
print(f"최저 매출: {result['최저 매출 월']} ({result['최저 매출액']:,}원)")
print(f"목표 달성 월: {', '.join(result['목표 달성 월'])}")
print(f"달성률: {result['달성률']:.1f}%")
```

### 예제 2: 고객 등급 분류 시스템

```python
def classify_customer_grade(purchase_amount, purchase_count):
    """
    구매 금액과 횟수로 고객 등급 분류
    """
    if purchase_amount >= 10000000 and purchase_count >= 50:
        return 'VIP', 0.20
    elif purchase_amount >= 5000000 and purchase_count >= 30:
        return 'GOLD', 0.15
    elif purchase_amount >= 3000000 and purchase_count >= 15:
        return 'SILVER', 0.10
    else:
        return 'BRONZE', 0.05

# 고객 리스트
customers = [
    {'id': 'C001', 'name': '김고객', 'amount': 12000000, 'count': 65},
    {'id': 'C002', 'name': '이고객', 'amount': 6500000, 'count': 35},
    {'id': 'C003', 'name': '박고객', 'amount': 4200000, 'count': 22},
    {'id': 'C004', 'name': '정고객', 'amount': 2100000, 'count': 12}
]

# 등급 부여 및 혜택 계산
print("=== 고객 등급 및 혜택 ===")
for customer in customers:
    grade, discount_rate = classify_customer_grade(
        customer['amount'], 
        customer['count']
    )
    customer['grade'] = grade
    customer['discount_rate'] = discount_rate
    
    print(f"{customer['name']} ({customer['id']})")
    print(f"  - 총 구매액: {customer['amount']:,}원")
    print(f"  - 구매 횟수: {customer['count']}회")
    print(f"  - 등급: {grade}")
    print(f"  - 할인율: {discount_rate*100}%")
    print()

# VIP 고객만 필터링
vip_customers = [c for c in customers if c['grade'] == 'VIP']
print(f"VIP 고객 수: {len(vip_customers)}명")
```

### 예제 3: 상품 재고 관리

```python
def check_inventory(inventory, safety_stock=10):
    """
    재고 확인 및 발주 필요 상품 추출
    """
    # 안전 재고 미달 상품
    low_stock = {
        product: stock 
        for product, stock in inventory.items() 
        if stock < safety_stock
    }
    
    # 재고 과다 상품 (안전 재고의 5배 이상)
    excess_stock = {
        product: stock 
        for product, stock in inventory.items() 
        if stock > safety_stock * 5
    }
    
    return low_stock, excess_stock

# 현재 재고
current_inventory = {
    '노트북': 5,
    '마우스': 120,
    '키보드': 8,
    '모니터': 15,
    '헤드셋': 3,
    '웹캠': 80
}

low, excess = check_inventory(current_inventory, safety_stock=10)

print("=== 재고 관리 시스템 ===")
print(f"\n⚠️ 발주 필요 (안전 재고 미달):")
for product, stock in low.items():
    order_qty = 20 - stock  # 목표 재고 20개
    print(f"  - {product}: 현재 {stock}개, 발주 필요량 {order_qty}개")

print(f"\n📦 재고 과다 (세일 검토):")
for product, stock in excess.items():
    print(f"  - {product}: {stock}개")
```

---

## 💪 실습 문제

### 문제 1: 직원 급여 인상 프로그램

다음 조건으로 직원 급여 인상액을 계산하는 프로그램을 작성하세요:
- 5년 이상 근속: 10% 인상
- 3~5년 미만: 7% 인상
- 1~3년 미만: 5% 인상
- 1년 미만: 3% 인상

```python
def calculate_raise(current_salary, years_of_service):
    """급여 인상액 계산"""
    # TODO: 코드 작성
    pass

# 테스트
employees = [
    {'name': '김직원', 'salary': 3000000, 'years': 6},
    {'name': '이직원', 'salary': 3500000, 'years': 4},
    {'name': '박직원', 'salary': 2800000, 'years': 2}
]

# 결과 출력
```

### 문제 2: 상위 N개 상품 추출

매출 데이터에서 상위 3개 상품을 추출하는 함수를 작성하세요.

```python
sales_data = {
    '노트북': 15000000,
    '마우스': 2500000,
    '키보드': 4200000,
    '모니터': 8500000,
    '헤드셋': 3200000
}

# TODO: 상위 3개 추출 함수 작성
```

### 문제 3: 월별 매출 증감률 계산

전월 대비 증감률을 계산하는 프로그램을 작성하세요.

```python
monthly_sales = [3200000, 3500000, 3100000, 4200000, 4500000]

# TODO: 증감률 계산 (예: [0%, 9.4%, -11.4%, 35.5%, 7.1%])
```

---

## 📝 정리

이번 시간에 배운 내용:

✅ **Python 기본 자료형**: List, Dict, Tuple, Set  
✅ **제어문**: if-elif-else, for, while  
✅ **함수**: def, return, 매개변수  
✅ **List Comprehension**: 간결한 리스트 생성  
✅ **Lambda 함수**: 익명 함수, map/filter  
✅ **실무 예제**: 매출 분석, 고객 등급, 재고 관리

---

## 🔗 다음 시간 예고

**2교시: Pandas 기초 - 데이터 불러오기 & 탐색**

- DataFrame 생성 및 구조
- CSV 파일 읽기
- 데이터 탐색 (head, info, describe)
- 행/열 선택 및 필터링

---

**수고하셨습니다! 🎉**
