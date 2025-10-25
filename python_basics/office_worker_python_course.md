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

---
---

# 📘 2교시: 텍스트(.txt) 파일 입출력 / 오늘의 목표 자동 기록 프로그램

---

## 1) 이론 설명

### 파일 입출력이란?

**파일 입출력(File I/O)**은 프로그램에서 파일을 읽고 쓰는 작업입니다. 직장에서 문서를 작성하고 저장하는 것과 같습니다.

**직장인 예시로 이해하기:**
- 회의록을 작성하고 파일로 저장 → **파일 쓰기 (Write)**
- 저장된 회의록을 다시 열어서 확인 → **파일 읽기 (Read)**
- 회의록에 추가 내용 작성 → **파일 추가 (Append)**

**왜 파일 입출력이 중요한가?**
- 데이터를 영구적으로 보관할 수 있습니다
- 프로그램을 종료해도 데이터가 사라지지 않습니다
- 다른 프로그램과 데이터를 공유할 수 있습니다

### 파일 열기: open() 함수

```python
파일객체 = open(파일경로, 모드, encoding='utf-8')
```

**주요 모드:**
- `'r'` (read): 읽기 모드 - 파일을 읽기만 함
- `'w'` (write): 쓰기 모드 - 새로 쓰기 (기존 내용 삭제)
- `'a'` (append): 추가 모드 - 파일 끝에 내용 추가
- `'r+'`: 읽기+쓰기 모드

**encoding 파라미터:**
- `encoding='utf-8'`: 한글이 깨지지 않도록 UTF-8 인코딩 지정
- Windows에서 한글 파일 다룰 때 필수!

### 파일 닫기: close() 메서드

```python
파일객체.close()
```

**왜 파일을 닫아야 하나?**
- 파일을 열어두면 메모리 낭비
- 다른 프로그램이 파일을 사용할 수 없음
- 데이터 손실 방지

### with 문: 자동으로 파일 닫기

```python
with open('파일.txt', 'r', encoding='utf-8') as f:
    내용 = f.read()
# with 블록을 벗어나면 자동으로 파일이 닫힘
```

**장점:**
- 파일을 자동으로 닫아주므로 안전
- close()를 잊어버릴 걱정 없음
- 에러가 발생해도 파일이 제대로 닫힘

### 파일 읽기 메서드

**1. read() - 전체 읽기**
```python
내용 = f.read()  # 파일 전체를 문자열로 읽음
```

**2. readline() - 한 줄씩 읽기**
```python
줄 = f.readline()  # 한 줄만 읽음
```

**3. readlines() - 모든 줄을 리스트로**
```python
줄들 = f.readlines()  # 각 줄이 리스트의 요소가 됨
```

### 파일 쓰기 메서드

**1. write() - 문자열 쓰기**
```python
f.write("안녕하세요\n")  # \n을 직접 추가해야 줄바꿈
```

**2. writelines() - 여러 줄 쓰기**
```python
줄들 = ["첫 번째 줄\n", "두 번째 줄\n"]
f.writelines(줄들)
```

---

## 2) 실습 예제 코드

### 예제 1: 파일 쓰기 기본 (write 모드)

```python
# 파일에 텍스트 쓰기
# 'w' 모드는 파일이 없으면 생성하고, 있으면 내용을 지우고 새로 씀

# 방법 1: 수동으로 close()
file = open('greetings.txt', 'w', encoding='utf-8')
file.write("안녕하세요!\n")
file.write("파이썬으로 파일을 작성합니다.\n")
file.close()

print("✅ greetings.txt 파일이 생성되었습니다.")
```

### 예제 2: with 문을 사용한 안전한 파일 쓰기

```python
# with 문 사용 (권장 방법)
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write("=== 업무 보고서 ===\n")
    f.write("날짜: 2024년 10월 25일\n")
    f.write("작성자: 홍길동\n")
    f.write("\n")
    f.write("오늘의 업무:\n")
    f.write("1. 고객 미팅\n")
    f.write("2. 프로젝트 진행 상황 점검\n")
    f.write("3. 주간 보고서 작성\n")

# with 블록이 끝나면 자동으로 파일이 닫힘
print("✅ report.txt 파일이 생성되었습니다.")
```

### 예제 3: 파일 읽기 - read()로 전체 읽기

```python
# 파일 전체 내용을 한 번에 읽기
with open('report.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print("📄 파일 내용:")
    print(content)

# 출력:
# 📄 파일 내용:
# === 업무 보고서 ===
# 날짜: 2024년 10월 25일
# 작성자: 홍길동
# 
# 오늘의 업무:
# 1. 고객 미팅
# 2. 프로젝트 진행 상황 점검
# 3. 주간 보고서 작성
```

### 예제 4: 파일 읽기 - readline()으로 한 줄씩 읽기

```python
# 한 줄씩 읽기
with open('report.txt', 'r', encoding='utf-8') as f:
    print("📄 첫 세 줄만 읽기:")
    line1 = f.readline()  # 첫 번째 줄
    line2 = f.readline()  # 두 번째 줄
    line3 = f.readline()  # 세 번째 줄
    
    print(line1.strip())  # strip()으로 양쪽 공백 제거
    print(line2.strip())
    print(line3.strip())

# 출력:
# 📄 첫 세 줄만 읽기:
# === 업무 보고서 ===
# 날짜: 2024년 10월 25일
# 작성자: 홍길동
```

### 예제 5: 파일 읽기 - readlines()로 모든 줄을 리스트로

```python
# 모든 줄을 리스트로 읽기
with open('report.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    print(f"📄 총 {len(lines)}줄입니다.")
    print("\n각 줄 출력:")
    for i, line in enumerate(lines, 1):
        print(f"{i}. {line.strip()}")

# 출력:
# 📄 총 8줄입니다.
# 
# 각 줄 출력:
# 1. === 업무 보고서 ===
# 2. 날짜: 2024년 10월 25일
# 3. 작성자: 홍길동
# 4. 
# 5. 오늘의 업무:
# 6. 1. 고객 미팅
# 7. 2. 프로젝트 진행 상황 점검
# 8. 3. 주간 보고서 작성
```

### 예제 6: 파일 추가 (append 모드)

```python
# 기존 파일에 내용 추가하기
with open('report.txt', 'a', encoding='utf-8') as f:
    f.write("\n")
    f.write("=== 추가 사항 ===\n")
    f.write("- 내일 고객사 방문 예정\n")
    f.write("- 제안서 준비 필요\n")

print("✅ 파일에 내용이 추가되었습니다.")

# 추가된 내용 확인
with open('report.txt', 'r', encoding='utf-8') as f:
    print("\n📄 업데이트된 파일 내용:")
    print(f.read())
```

### 예제 7: 오늘의 목표 기록 프로그램 (핵심 예제)

```python
import datetime

def save_daily_goal():
    """오늘의 목표를 파일에 저장하는 프로그램"""
    
    # 오늘 날짜 가져오기
    today = datetime.date.today()
    date_str = today.strftime("%Y년 %m월 %d일")
    
    # 사용자로부터 목표 입력받기
    print(f"📅 {date_str}")
    print("오늘의 목표를 입력하세요 (완료하려면 빈 줄 입력):")
    
    goals = []
    goal_num = 1
    
    while True:
        goal = input(f"{goal_num}. ")
        if goal == "":  # 빈 줄이면 종료
            break
        goals.append(f"{goal_num}. {goal}\n")
        goal_num += 1
    
    # 파일명: goals_2024-10-25.txt 형식
    filename = f"goals_{today}.txt"
    
    # 파일에 저장
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== {date_str} 목표 ===\n")
        f.write("\n")
        f.writelines(goals)
        f.write("\n")
        f.write("화이팅! 💪\n")
    
    print(f"\n✅ '{filename}' 파일에 목표가 저장되었습니다!")
    
    # 저장된 내용 확인
    with open(filename, 'r', encoding='utf-8') as f:
        print("\n📄 저장된 내용:")
        print(f.read())

# 프로그램 실행 예시:
# save_daily_goal()

# 실행 예시:
# 📅 2024년 10월 25일
# 오늘의 목표를 입력하세요 (완료하려면 빈 줄 입력):
# 1. 보고서 작성 완료하기
# 2. 고객 미팅 준비
# 3. 이메일 답장 보내기
# 4. [엔터]
# 
# ✅ 'goals_2024-10-25.txt' 파일에 목표가 저장되었습니다!
```

### 예제 8: 목표 달성 체크 프로그램

```python
def check_goals(filename):
    """저장된 목표를 불러와서 달성 여부를 체크하는 프로그램"""
    
    try:
        # 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print("📄 저장된 목표:")
        for line in lines:
            print(line.strip())
        
        print("\n" + "="*40)
        print("달성한 목표 앞에 체크 표시를 입력하세요 (완료: O, 미완료: X)")
        print("="*40 + "\n")
        
        # 목표만 추출 (숫자로 시작하는 줄)
        goals = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
        
        checked_goals = []
        for goal in goals:
            check = input(f"{goal.strip()} -> ")
            if check.upper() == 'O':
                checked_goals.append(f"✅ {goal}")
            else:
                checked_goals.append(f"❌ {goal}")
        
        # 결과 파일에 저장
        result_filename = filename.replace('.txt', '_결과.txt')
        with open(result_filename, 'w', encoding='utf-8') as f:
            f.write("=== 목표 달성 결과 ===\n\n")
            f.writelines(checked_goals)
            
            completed = sum(1 for g in checked_goals if '✅' in g)
            total = len(checked_goals)
            f.write(f"\n달성률: {completed}/{total} ({completed/total*100:.0f}%)\n")
        
        print(f"\n✅ 결과가 '{result_filename}'에 저장되었습니다!")
        
    except FileNotFoundError:
        print(f"❌ '{filename}' 파일을 찾을 수 없습니다.")

# 사용 예시:
# check_goals('goals_2024-10-25.txt')
```

---

## 3) 코드 상세 설명

### open() 함수의 동작 원리

```python
file = open('example.txt', 'w', encoding='utf-8')
```

**단계별 설명:**
1. `'example.txt'`: 파일 경로 (현재 디렉토리에 생성)
2. `'w'`: 쓰기 모드 
   - 파일이 없으면 새로 생성
   - 파일이 있으면 기존 내용 삭제하고 새로 씀
3. `encoding='utf-8'`: 한글 등 유니코드 문자를 올바르게 처리
4. 반환값: 파일 객체 (파일을 제어할 수 있는 객체)

### with 문의 동작 원리

```python
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()
# 여기서 파일이 자동으로 닫힘
```

**장점:**
- **자동 정리**: with 블록을 벗어날 때 자동으로 `f.close()` 호출
- **예외 안전**: 에러가 발생해도 파일이 확실히 닫힘
- **메모리 효율**: 파일 핸들을 빨리 해제하여 메모리 절약

**일반 방식과 비교:**
```python
# 일반 방식 (권장하지 않음)
f = open('file.txt', 'r', encoding='utf-8')
try:
    content = f.read()
finally:
    f.close()  # 반드시 닫아야 함

# with 문 (권장)
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()
# 자동으로 닫힘!
```

### 파일 모드별 동작 차이

| 모드 | 설명 | 파일이 없을 때 | 파일이 있을 때 |
|------|------|---------------|---------------|
| `'r'` | 읽기 | 에러 발생 | 읽기만 가능 |
| `'w'` | 쓰기 | 새로 생성 | 기존 내용 삭제 |
| `'a'` | 추가 | 새로 생성 | 끝에 추가 |
| `'r+'` | 읽기+쓰기 | 에러 발생 | 읽기/쓰기 가능 |

**실무 팁:**
- 로그 파일: `'a'` 모드 사용 (계속 추가)
- 설정 파일 읽기: `'r'` 모드
- 보고서 생성: `'w'` 모드 (매번 새로 작성)

### read() vs readline() vs readlines()

```python
# 파일 내용:
# Line 1
# Line 2
# Line 3

# read() - 전체를 하나의 문자열로
content = f.read()
# 결과: "Line 1\nLine 2\nLine 3"

# readline() - 한 줄씩
line = f.readline()  # "Line 1\n"
line = f.readline()  # "Line 2\n"

# readlines() - 모든 줄을 리스트로
lines = f.readlines()
# 결과: ["Line 1\n", "Line 2\n", "Line 3"]
```

**선택 기준:**
- 파일이 작고 전체를 처리: `read()`
- 파일이 크고 한 줄씩 처리: `readline()` 또는 `for line in f:`
- 모든 줄을 리스트로 필요: `readlines()`

### strip() 메서드의 중요성

```python
# readlines()로 읽으면 각 줄에 \n이 포함됨
lines = f.readlines()  # ["안녕하세요\n", "반갑습니다\n"]

# strip()으로 양쪽 공백과 \n 제거
for line in lines:
    print(line.strip())  # "안녕하세요", "반갑습니다"
```

**strip() 변형:**
- `strip()`: 양쪽 공백/줄바꿈 제거
- `lstrip()`: 왼쪽만 제거
- `rstrip()`: 오른쪽만 제거

---

## 4) 실습 미션

### 미션 1: 직원 명단 저장 ⭐
**목표:** 직원 정보를 파일에 저장하는 프로그램을 작성하세요.

**요구사항:**
- 파일명: `employees.txt`
- 저장 내용:
  ```
  === 직원 명단 ===
  1. 홍길동 - 영업팀 - 대리
  2. 김철수 - 개발팀 - 과장
  3. 이영희 - 인사팀 - 사원
  ```
- `with` 문 사용
- `encoding='utf-8'` 지정

---

### 미션 2: 파일 읽고 특정 줄 출력 ⭐⭐
**목표:** employees.txt 파일을 읽어서 "개발팀" 직원만 출력하세요.

**요구사항:**
- 파일을 한 줄씩 읽기
- "개발팀"이 포함된 줄만 출력
- 결과 예시: `2. 김철수 - 개발팀 - 과장`

**힌트:**
```python
if "개발팀" in line:
    print(line.strip())
```

---

### 미션 3: 출퇴근 기록 프로그램 ⭐⭐⭐
**목표:** 매일 출퇴근 시간을 기록하고 조회하는 프로그램을 작성하세요.

**요구사항:**
- 함수 1: `record_attendance(name, time_in, time_out)`
  - 파일명: `attendance.txt`
  - 추가 모드('a')로 기록
  - 형식: `2024-10-25, 홍길동, 09:00, 18:00`
  
- 함수 2: `view_attendance()`
  - attendance.txt 파일 전체 내용 출력
  
**예시:**
```python
record_attendance("홍길동", "09:00", "18:00")
record_attendance("김철수", "08:50", "17:30")
view_attendance()
```

**출력:**
```
=== 출퇴근 기록 ===
2024-10-25, 홍길동, 09:00, 18:00
2024-10-25, 김철수, 08:50, 17:30
```

---

### 미션 4: 회의록 자동 작성기 ⭐⭐⭐
**목표:** 회의 내용을 입력받아 정형화된 회의록 파일을 생성하세요.

**요구사항:**
- 입력 정보: 회의 제목, 날짜, 참석자, 안건 (여러 개)
- 파일명: `meeting_YYYYMMDD.txt` 형식
- 출력 형식:
  ```
  ==================
  회의록
  ==================
  제목: [회의 제목]
  날짜: [날짜]
  참석자: [참석자 명단]
  
  안건:
  1. [안건1]
  2. [안건2]
  ...
  ==================
  ```

**힌트:**
- `input()`으로 사용자 입력받기
- 안건은 while 문으로 여러 개 입력받기
- f-string으로 파일명 생성

---

## 5) 퀴즈

### 퀴즈 1 (파일 모드) 📝
다음 중 기존 파일에 내용을 추가하고 싶을 때 사용하는 모드는?

**A)** `'r'`  
**B)** `'w'`  
**C)** `'a'`  
**D)** `'x'`

---

### 퀴즈 2 (파일 읽기) 📝
다음 코드의 실행 결과는?

```python
with open('test.txt', 'w', encoding='utf-8') as f:
    f.write("Hello\n")
    f.write("World\n")

with open('test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))
```

**A)** 1  
**B)** 2  
**C)** 10  
**D)** 에러 발생

---

### 퀴즈 3 (응용 문제) 📝
다음 코드의 실행 후 `data.txt` 파일의 최종 내용은?

```python
# 첫 번째 쓰기
with open('data.txt', 'w', encoding='utf-8') as f:
    f.write("First\n")

# 두 번째 쓰기
with open('data.txt', 'w', encoding='utf-8') as f:
    f.write("Second\n")

# 세 번째 추가
with open('data.txt', 'a', encoding='utf-8') as f:
    f.write("Third\n")
```

**A)** First  
**B)** Second  
**C)** Second<br>Third  
**D)** First<br>Second<br>Third

---

## 6) 정답 및 해설

### 퀴즈 1 정답: **C) 'a'**

**해설:**
- `'r'` (read): 읽기 전용 - 파일 수정 불가
- `'w'` (write): 쓰기 모드 - **기존 내용을 모두 삭제**하고 새로 씀
- `'a'` (append): **추가 모드 - 기존 내용 유지하고 끝에 추가** ✅
- `'x'`: 배타적 생성 - 파일이 이미 존재하면 에러

**실무 활용:**
```python
# 로그 파일에 계속 기록할 때
with open('log.txt', 'a', encoding='utf-8') as f:
    f.write("2024-10-25 10:30 - 사용자 로그인\n")
```

---

### 퀴즈 2 정답: **B) 2**

**해설:**

**단계별 실행:**
1. 파일 쓰기:
   ```python
   f.write("Hello\n")  # 첫 번째 줄
   f.write("World\n")  # 두 번째 줄
   ```

2. 파일 읽기:
   ```python
   lines = f.readlines()
   # 결과: ["Hello\n", "World\n"]
   ```

3. 줄 개수:
   ```python
   print(len(lines))  # 2
   ```

**포인트:**
- `readlines()`는 각 줄을 리스트의 요소로 반환
- `\n`(줄바꿈)으로 구분하여 2개의 요소가 생성됨

---

### 퀴즈 3 정답: **C) Second<br>Third**

**해설:**

**단계별 실행:**

1. **첫 번째 쓰기** (`'w'` 모드):
   ```python
   with open('data.txt', 'w', encoding='utf-8') as f:
       f.write("First\n")
   ```
   파일 내용: `First`

2. **두 번째 쓰기** (`'w'` 모드):
   ```python
   with open('data.txt', 'w', encoding='utf-8') as f:
       f.write("Second\n")
   ```
   파일 내용: `Second` (First가 삭제됨!)

3. **세 번째 추가** (`'a'` 모드):
   ```python
   with open('data.txt', 'a', encoding='utf-8') as f:
       f.write("Third\n")
   ```
   파일 내용: `Second\nThird` (Second 뒤에 추가)

**최종 파일 내용:**
```
Second
Third
```

**주의사항:**
- `'w'` 모드는 파일을 열 때 기존 내용을 **모두 삭제**합니다
- 기존 내용을 보존하려면 `'a'` 모드를 사용해야 합니다

---

## 💡 2교시 핵심 요약

### 파일 입출력 3단계
1. **열기**: `open(파일명, 모드, encoding='utf-8')`
2. **작업**: `read()`, `write()`, `readlines()` 등
3. **닫기**: `close()` 또는 `with` 문으로 자동

### 파일 모드 기억하기
- **'r'**: Read (읽기) - 파일이 없으면 에러
- **'w'**: Write (쓰기) - 기존 내용 삭제
- **'a'**: Append (추가) - 끝에 추가

### 실무 활용 팁
✅ 항상 `with` 문 사용 (자동으로 파일 닫힘)
✅ `encoding='utf-8'` 지정 (한글 깨짐 방지)
✅ `strip()` 으로 불필요한 공백/줄바꿈 제거
✅ 로그 파일은 `'a'` 모드로 계속 추가
✅ 파일명에 날짜 포함하면 관리 편함 (`log_2024-10-25.txt`)

### 에러 처리
```python
try:
    with open('file.txt', 'r', encoding='utf-8') as f:
        content = f.read()
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"에러 발생: {e}")
```

---

## 🎯 다음 교시 예고

**3교시: CSV 라이브러리를 활용한 데이터 처리**
- CSV 파일이란?
- `csv` 모듈로 CSV 읽기/쓰기
- `DictReader`와 `DictWriter` 활용
- **실습 프로젝트:** 고객 명단을 CSV로 읽어 리스트로 변환

---

**2교시를 완료하셨습니다! 수고하셨습니다! 🎉**

"다음 교시"를 입력하시면 3교시 내용을 시작하겠습니다.
