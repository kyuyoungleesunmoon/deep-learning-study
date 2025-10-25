# 직장인을 위한 파이썬 완성 강의 (7교시)

---

# 📘 1교시: 함수(Function)의 정의 및 호출 / 원화→달러 환율 함수 만들기

---

## 1) 이론 설명

### 함수란 무엇인가?

**함수(Function)**는 특정 작업을 수행하는 코드 묶음입니다. 마치 회사에서 사용하는 "업무 매뉴얼"과 같습니다. 

**직장인 예시로 이해하기:**
- 회사에서 "출장비 정산" 업무를 할 때마다 같은 절차를 반복합니다
- 영수증 모으기 → 엑셀에 입력 → 합계 계산 → 결재 요청
- 이 절차를 매번 처음부터 다시 생각하지 않고, "출장비 정산 매뉴얼"을 따르면 됩니다
- **함수도 마찬가지입니다!** 자주 사용하는 코드를 함수로 만들어두면, 필요할 때마다 간단히 호출만 하면 됩니다

### 함수를 사용하는 이유

1. **코드 재사용**: 같은 코드를 여러 번 작성하지 않아도 됩니다
2. **유지보수 용이**: 수정이 필요하면 함수 한 곳만 고치면 됩니다
3. **가독성 향상**: 복잡한 코드를 의미 있는 이름의 함수로 나눌 수 있습니다
4. **협업 효율**: 동료가 내가 만든 함수를 쉽게 사용할 수 있습니다

### 함수의 기본 구조

```python
def 함수이름(매개변수1, 매개변수2):
    """함수 설명 (docstring)"""
    # 실행할 코드
    결과 = 매개변수1 + 매개변수2
    return 결과  # 결과값 반환
```

**구성 요소:**
- `def`: 함수를 정의한다는 키워드 (define의 약자)
- `함수이름`: 함수의 이름 (스네이크 케이스 사용 권장)
- `매개변수`: 함수에 전달하는 입력값
- `return`: 함수의 결과값을 반환
- `docstring`: 함수 설명 (선택사항이지만 권장)

### 함수 호출하기

```python
결과값 = 함수이름(인자1, 인자2)
```

- **인자(argument)**: 함수를 호출할 때 실제로 전달하는 값

### 매개변수와 반환값

**매개변수(Parameter)가 없는 함수:**
```python
def say_hello():
    print("안녕하세요!")
    
say_hello()  # 출력: 안녕하세요!
```

**반환값(Return)이 없는 함수:**
```python
def greet(name):
    print(f"{name}님, 환영합니다!")
    # return이 없으면 None을 반환
    
greet("홍길동")  # 출력: 홍길동님, 환영합니다!
```

**매개변수와 반환값이 모두 있는 함수:**
```python
def add(a, b):
    return a + b
    
result = add(10, 20)  # result = 30
```

---

## 2) 실습 예제 코드

### 예제 1: 기본 함수 만들기

```python
# 인사 함수 - 매개변수 없음, 반환값 없음
def greet():
    """간단한 인사말을 출력하는 함수"""
    print("안녕하세요! 좋은 하루 되세요!")

# 함수 호출
greet()  # 출력: 안녕하세요! 좋은 하루 되세요!
```

### 예제 2: 매개변수가 있는 함수

```python
# 직원 정보 출력 함수
def print_employee_info(name, department, position):
    """직원 정보를 출력하는 함수
    
    Args:
        name: 직원 이름
        department: 부서명
        position: 직급
    """
    print(f"이름: {name}")
    print(f"부서: {department}")
    print(f"직급: {position}")

# 함수 호출
print_employee_info("김철수", "영업팀", "대리")
# 출력:
# 이름: 김철수
# 부서: 영업팀
# 직급: 대리
```

### 예제 3: 반환값이 있는 함수

```python
# 월급 계산 함수
def calculate_monthly_salary(annual_salary):
    """연봉을 월급으로 변환하는 함수
    
    Args:
        annual_salary: 연봉 (원)
    
    Returns:
        월급 (원)
    """
    monthly_salary = annual_salary / 12
    return monthly_salary

# 함수 호출 및 결과 저장
연봉 = 36000000  # 3천6백만원
월급 = calculate_monthly_salary(연봉)
print(f"연봉 {연봉:,}원의 월급: {월급:,.0f}원")
# 출력: 연봉 36,000,000원의 월급: 3,000,000원
```

### 예제 4: 원화를 달러로 환전하는 함수 (핵심 예제)

```python
# 원화를 달러로 환전하는 함수
def krw_to_usd(krw_amount, exchange_rate=1300):
    """원화를 달러로 환전하는 함수
    
    Args:
        krw_amount: 원화 금액
        exchange_rate: 환율 (기본값 1300원, 생략 가능)
    
    Returns:
        달러 금액
    """
    usd_amount = krw_amount / exchange_rate
    return usd_amount

# 사용 예시 1: 환율 지정하지 않음 (기본값 1300 사용)
원화 = 130000
달러 = krw_to_usd(원화)
print(f"{원화:,}원 = ${달러:.2f}")
# 출력: 130,000원 = $100.00

# 사용 예시 2: 환율 직접 지정
원화 = 260000
환율 = 1350
달러 = krw_to_usd(원화, 환율)
print(f"{원화:,}원 = ${달러:.2f} (환율: {환율}원)")
# 출력: 260,000원 = $192.59 (환율: 1350원)
```

### 예제 5: 달러를 원화로 환전하는 함수 (역방향)

```python
# 달러를 원화로 환전하는 함수
def usd_to_krw(usd_amount, exchange_rate=1300):
    """달러를 원화로 환전하는 함수
    
    Args:
        usd_amount: 달러 금액
        exchange_rate: 환율 (기본값 1300원)
    
    Returns:
        원화 금액
    """
    krw_amount = usd_amount * exchange_rate
    return krw_amount

# 사용 예시
달러 = 500
원화 = usd_to_krw(달러)
print(f"${달러} = {원화:,.0f}원")
# 출력: $500 = 650,000원
```

### 예제 6: 여러 값을 반환하는 함수

```python
# 환전 정보를 한 번에 계산하는 함수
def currency_exchange(krw_amount, exchange_rate=1300):
    """원화→달러 환전 시 상세 정보를 반환
    
    Args:
        krw_amount: 원화 금액
        exchange_rate: 환율
    
    Returns:
        tuple: (달러 금액, 수수료, 실제 받는 달러)
    """
    usd_amount = krw_amount / exchange_rate
    fee = usd_amount * 0.02  # 2% 수수료
    actual_usd = usd_amount - fee
    
    return usd_amount, fee, actual_usd

# 사용 예시
원화 = 1000000
달러, 수수료, 실제달러 = currency_exchange(원화)

print(f"환전 금액: {원화:,}원")
print(f"환전 달러: ${달러:.2f}")
print(f"수수료(2%): ${수수료:.2f}")
print(f"실제 수령: ${실제달러:.2f}")
# 출력:
# 환전 금액: 1,000,000원
# 환전 달러: $769.23
# 수수료(2%): $15.38
# 실제 수령: $753.85
```

---

## 3) 코드 상세 설명

### 함수 정의 단계별 설명

```python
def krw_to_usd(krw_amount, exchange_rate=1300):
    """원화를 달러로 환전하는 함수"""
    usd_amount = krw_amount / exchange_rate
    return usd_amount
```

**1단계: 함수 선언**
- `def krw_to_usd(...)`: "krw_to_usd"라는 이름의 함수를 정의합니다
- 함수명은 소문자와 언더스코어(_)로 작성 (스네이크 케이스)

**2단계: 매개변수 설정**
- `krw_amount`: 원화 금액 (필수 매개변수)
- `exchange_rate=1300`: 환율 (기본값이 있는 선택 매개변수)
- 기본값이 있으면 함수 호출 시 생략 가능

**3단계: 함수 본문 (실행 코드)**
- `usd_amount = krw_amount / exchange_rate`
- 원화를 환율로 나누어 달러로 계산

**4단계: 반환값**
- `return usd_amount`
- 계산된 달러 금액을 함수 밖으로 반환
- `return`을 만나면 함수 실행이 종료됩니다

### 함수 호출 과정

```python
result = krw_to_usd(130000, 1300)
```

1. `krw_to_usd` 함수를 찾습니다
2. `krw_amount`에 130000을 대입
3. `exchange_rate`에 1300을 대입
4. 함수 내부 코드 실행: `usd_amount = 130000 / 1300 = 100.0`
5. `return 100.0` 으로 결과 반환
6. `result` 변수에 100.0이 저장됨

### 기본값(Default Value) 활용

```python
def krw_to_usd(krw_amount, exchange_rate=1300):
    ...
```

- `exchange_rate=1300`은 기본값 설정
- 호출 시 환율을 지정하지 않으면 자동으로 1300 사용

```python
# 환율 생략 → 기본값 1300 사용
krw_to_usd(130000)

# 환율 명시 → 1350 사용
krw_to_usd(130000, 1350)
```

### 여러 값 반환 (Tuple Unpacking)

```python
def currency_exchange(krw_amount, exchange_rate=1300):
    usd_amount = krw_amount / exchange_rate
    fee = usd_amount * 0.02
    actual_usd = usd_amount - fee
    return usd_amount, fee, actual_usd  # 3개 값을 튜플로 반환
```

**반환 과정:**
- 여러 값을 쉼표로 구분하면 자동으로 튜플(tuple)로 묶임
- `return (usd_amount, fee, actual_usd)`와 동일

**받기 과정:**
```python
달러, 수수료, 실제달러 = currency_exchange(1000000)
```
- 튜플의 각 요소가 순서대로 변수에 할당됨 (언패킹)

---

## 4) 실습 미션

### 미션 1: 기본 함수 만들기 ⭐
**목표:** 직원 출퇴근 시간을 출력하는 함수를 작성하세요.

**요구사항:**
- 함수명: `print_work_hours`
- 매개변수: `start_time` (출근 시간), `end_time` (퇴근 시간)
- 출력 형식: "출근: 09:00, 퇴근: 18:00"

**예상 출력:**
```
출근: 09:00, 퇴근: 18:00
```

---

### 미션 2: 근무 시간 계산 함수 ⭐⭐
**목표:** 출퇴근 시간을 입력받아 총 근무 시간을 계산하는 함수를 작성하세요.

**요구사항:**
- 함수명: `calculate_work_hours`
- 매개변수: `start_hour` (출근 시각, 정수), `end_hour` (퇴근 시각, 정수)
- 점심시간 1시간 자동 제외
- 반환값: 실제 근무 시간

**예시:**
```python
hours = calculate_work_hours(9, 18)  # 9시 출근, 18시 퇴근
print(f"근무 시간: {hours}시간")  # 출력: 근무 시간: 8시간
```

---

### 미션 3: 환율 계산기 업그레이드 ⭐⭐⭐
**목표:** 원화를 여러 통화로 환전하는 함수를 작성하세요.

**요구사항:**
- 함수명: `convert_currency`
- 매개변수: 
  - `krw_amount`: 원화 금액
  - `currency`: 통화 종류 ("USD", "JPY", "EUR" 중 하나)
- 환율 정보:
  - USD: 1300원
  - JPY: 900원 (100엔 기준)
  - EUR: 1400원
- 반환값: 환전된 금액

**예시:**
```python
usd = convert_currency(130000, "USD")
print(f"${usd:.2f}")  # $100.00

jpy = convert_currency(90000, "JPY")
print(f"¥{jpy:.0f}")  # ¥10000

eur = convert_currency(140000, "EUR")
print(f"€{eur:.2f}")  # €100.00
```

---

### 미션 4: 왕복 환전 손실 계산 ⭐⭐⭐
**목표:** 원화→달러→원화로 환전 시 수수료로 인한 손실을 계산하세요.

**요구사항:**
- 함수명: `calculate_exchange_loss`
- 매개변수: `krw_amount` (초기 원화 금액)
- 환율: 1달러 = 1300원 (고정)
- 수수료: 왕복 각 2% (총 2번 수수료 부과)
- 반환값: (최종 원화, 손실액)

**계산 과정:**
1. 원화 → 달러 환전 (2% 수수료)
2. 달러 → 원화 환전 (2% 수수료)
3. 최종 원화와 손실액 반환

**예시:**
```python
final, loss = calculate_exchange_loss(1000000)
print(f"초기: 1,000,000원")
print(f"최종: {final:,.0f}원")
print(f"손실: {loss:,.0f}원")
```

---

## 5) 퀴즈

### 퀴즈 1 (기본 개념) 📝
다음 중 함수를 정의할 때 사용하는 키워드는?

**A)** `func`  
**B)** `define`  
**C)** `def`  
**D)** `function`

---

### 퀴즈 2 (매개변수와 반환값) 📝
다음 함수의 실행 결과는?

```python
def multiply(a, b=10):
    return a * b

result = multiply(5)
print(result)
```

**A)** 5  
**B)** 10  
**C)** 50  
**D)** 에러 발생

---

### 퀴즈 3 (응용 문제) 📝
다음 코드의 출력 결과는?

```python
def discount_price(price, rate=0.1):
    discount = price * rate
    final_price = price - discount
    return final_price

item_price = 50000
result = discount_price(item_price, 0.2)
print(f"{result:,.0f}원")
```

**A)** 5,000원  
**B)** 10,000원  
**C)** 40,000원  
**D)** 45,000원

---

## 6) 정답 및 해설

### 퀴즈 1 정답: **C) def**

**해설:**
- Python에서 함수를 정의할 때는 `def` 키워드를 사용합니다
- `def`는 "define"의 약자입니다
- 문법: `def 함수이름(매개변수):`

**예시:**
```python
def my_function():
    print("Hello!")
```

---

### 퀴즈 2 정답: **C) 50**

**해설:**
- 함수 정의: `def multiply(a, b=10)`
- `b=10`은 기본값(default value)입니다
- `multiply(5)` 호출 시:
  - `a = 5`
  - `b = 10` (기본값 사용)
  - `return 5 * 10 = 50`

**기본값의 특징:**
- 매개변수를 생략하면 기본값이 자동으로 사용됩니다
- `multiply(5, 20)`처럼 값을 지정하면 기본값 대신 지정한 값 사용

---

### 퀴즈 3 정답: **C) 40,000원**

**해설:**

**단계별 계산:**
1. 함수 호출: `discount_price(50000, 0.2)`
   - `price = 50000`
   - `rate = 0.2` (20% 할인)

2. 함수 내부 실행:
   - `discount = 50000 * 0.2 = 10000`
   - `final_price = 50000 - 10000 = 40000`
   - `return 40000`

3. 출력: `40,000원`

**포인트:**
- 원가 50,000원에서 20% 할인 → 10,000원 할인
- 최종 가격 = 50,000 - 10,000 = 40,000원

---

## 💡 1교시 핵심 요약

### 함수의 4가지 핵심 요소
1. **정의**: `def` 키워드로 시작
2. **매개변수**: 함수에 전달할 입력값
3. **실행 코드**: 함수가 수행할 작업
4. **반환값**: `return`으로 결과를 돌려줌

### 실무 활용 팁
✅ 함수명은 동사로 시작하면 이해하기 쉽습니다
   - `calculate_salary()` ⭐ (좋음)
   - `salary()` ❌ (애매함)

✅ 하나의 함수는 하나의 기능만 수행
   - 함수가 너무 길면 여러 개로 나누기

✅ 기본값 활용으로 편의성 향상
   - 자주 사용하는 값은 기본값으로 설정

✅ docstring으로 함수 설명 작성
   - 나중에 다시 봐도 이해하기 쉬움

---

## 🎯 다음 교시 예고

**2교시: 텍스트(.txt) 파일 입출력**
- 파일 읽기/쓰기 기본
- `open()`, `read()`, `write()` 함수
- `with` 문을 활용한 안전한 파일 처리
- **실습 프로젝트:** 오늘의 목표 자동 기록 프로그램

---

**1교시를 완료하셨습니다! 수고하셨습니다! 🎉**

"다음 교시"를 입력하시면 2교시 내용을 시작하겠습니다.
