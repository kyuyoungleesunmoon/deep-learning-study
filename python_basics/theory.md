# Python 기초 - 변수 선언 규칙과 기본 자료형

> **난이도**: 초급 (프로그래밍 경험이 없어도 학습 가능)
> 
> **예상 학습 시간**: 3시간 30분 (이론 2시간 + 실습 1시간 30분)

---

## 🎯 학습 목표

이 과정을 마치면 다음을 할 수 있게 됩니다:

1. **변수 관리**: Python에서 올바른 변수명을 지어 데이터를 저장하고 관리할 수 있습니다
2. **숫자 연산**: 정수, 실수, 복소수를 이해하고 다양한 수학 연산을 수행할 수 있습니다
3. **문자열 처리**: 문자열을 자유롭게 조작하고 원하는 형식으로 출력할 수 있습니다
4. **자료형 변환**: 상황에 맞게 데이터 타입을 변환할 수 있습니다
5. **함수 활용**: Python 내장 함수를 활용하여 효율적인 코드를 작성할 수 있습니다
6. **코드 문서화**: 적절한 주석으로 코드를 설명하고 유지보수하기 쉽게 만들 수 있습니다

---

## 📊 학습 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│                    Python 기초 학습 과정                     │
└─────────────────────────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │  1. 변수 기초 (30분)                    │
         │  • 변수란?                              │
         │  • 메모리 개념                          │
         │  • 네이밍 규칙 (snake_case)            │
         └────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │  2. 숫자형 자료형 (40분)                │
         │  • int, float, complex                 │
         │  • 산술 연산                            │
         │  • 내장 함수 & math 모듈               │
         └────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │  3. 문자열 자료형 (50분)                │
         │  • 선언 방법                            │
         │  • 인덱싱 & 슬라이싱                   │
         │  • 문자열 메서드 & 포맷팅              │
         └────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │  4. 자료형 변환 (30분)                  │
         │  • 문자열 ↔ 숫자                       │
         │  • 정수 ↔ 실수                         │
         │  • 변환 시 주의사항                    │
         └────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │  5. 내장 함수 (25분)                    │
         │  • type(), isinstance()                │
         │  • len(), range(), id()                │
         │  • 실전 활용                            │
         └────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │  6. 주석 활용법 (20분)                  │
         │  • 한 줄 / 여러 줄 주석                │
         │  • 주석 작성 원칙                      │
         │  • docstring                           │
         └────────────────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────┐
         │  종합 실습 & 평가 (30분)                │
         │  • 통합 프로젝트                        │
         │  • 종합 퀴즈                            │
         └────────────────────────────────────────┘
                              ▼
                    🎓 학습 완료!
           다음 단계: 제어문, 자료구조, 함수
```

**학습 팁**:
- 각 섹션을 순서대로 학습하세요
- 이론을 읽은 후 바로 실습 코드를 실행해보세요
- 연습 문제를 직접 풀어보고 해설과 비교하세요
- 퀴즈로 개념을 확인하세요

---

## 📚 목차

1. [파이썬 기초 및 변수 선언 개념](#1-파이썬-기초-및-변수-선언-개념)
2. [숫자형 자료형](#2-숫자형-자료형)
3. [문자열(str) 자료형](#3-문자열str-자료형)
4. [자료형 변환](#4-자료형-변환)
5. [내장 함수와 type() 활용](#5-내장-함수와-type-활용)
6. [주석 활용법](#6-주석-활용법)

---

## 1. 파이썬 기초 및 변수 선언 개념

### 1.1 변수란 무엇인가?

변수(Variable)는 **데이터를 저장하는 공간**입니다. 마치 상자에 물건을 담듯이, 변수에 값을 저장하여 나중에 사용할 수 있습니다.

```python
# 변수에 값 저장하기
name = "홍길동"
age = 25
```

위 예제에서:
- `name`이라는 변수에 "홍길동"이라는 문자열을 저장
- `age`라는 변수에 25라는 숫자를 저장

### 1.2 메모리와 변수의 관계

컴퓨터의 메모리(RAM)는 여러 개의 방으로 나뉘어져 있습니다. 변수를 선언하면 메모리의 특정 위치에 데이터가 저장됩니다.

```
메모리 구조 (개념도):
┌─────────────┬─────────────┬─────────────┐
│ 주소: 1000  │ 주소: 1001  │ 주소: 1002  │
│ name="홍길동"│ age=25      │ ...         │
└─────────────┴─────────────┴─────────────┘
```

Python에서는 `id()` 함수로 변수가 저장된 메모리 주소를 확인할 수 있습니다:

```python
x = 100
print(id(x))  # 예: 140712345678912 (실제 값은 실행할 때마다 다름)
```

### 1.3 변수 이름 규칙 (Naming Convention)

Python에서 변수 이름을 지을 때는 다음 규칙을 따라야 합니다:

#### ✅ 허용되는 것:
- 영문 대소문자 (a-z, A-Z)
- 숫자 (0-9, 단 첫 글자는 불가)
- 언더스코어 (_)
- 한글도 가능하지만 권장하지 않음

#### ❌ 허용되지 않는 것:
- 숫자로 시작 (예: `1name` ❌)
- 특수문자 (예: `@name`, `name!` ❌)
- 공백 (예: `my name` ❌)
- Python 예약어 (예: `if`, `for`, `while` 등)

#### 올바른 변수 이름 예시:
```python
# 좋은 예
user_name = "김철수"
total_price = 15000
student_age = 20
_private_var = 100  # 언더스코어로 시작 가능

# 나쁜 예 (에러 발생)
# 2nd_name = "이순신"    # 숫자로 시작 ❌
# user-name = "박영희"   # 하이픈 사용 ❌
# my name = "최민수"     # 공백 사용 ❌
```

### 1.4 스네이크 케이스 (Snake Case)

Python에서는 **스네이크 케이스(snake_case)**를 변수 이름의 표준으로 사용합니다.

```python
# 스네이크 케이스: 단어 사이를 언더스코어(_)로 연결
user_name = "홍길동"
student_age = 25
total_price = 50000
is_active = True

# 다른 언어에서 사용하는 카멜 케이스 (Python에서는 비권장)
userName = "홍길동"
studentAge = 25
```

**왜 스네이크 케이스를 사용할까?**
- Python 공식 스타일 가이드(PEP 8)에서 권장
- 가독성이 좋음
- Python 커뮤니티의 표준

### 1.5 의미 있는 변수 이름 짓기

변수 이름은 **무엇을 저장하는지 명확하게** 표현해야 합니다.

```python
# 나쁜 예 - 의미를 알 수 없음
a = 25
x = "홍길동"
data = 15000

# 좋은 예 - 의미가 명확함
age = 25
student_name = "홍길동"
product_price = 15000
```

### 1.6 변수 값 변경하기

변수는 언제든지 새로운 값으로 변경할 수 있습니다.

```python
# 변수 선언
count = 10
print(count)  # 출력: 10

# 변수 값 변경
count = 20
print(count)  # 출력: 20

# 변수 값을 이용한 계산 후 저장
count = count + 5
print(count)  # 출력: 25
```

### 1.7 여러 변수 동시에 선언하기

Python에서는 여러 변수를 한 줄에 선언할 수 있습니다.

```python
# 방법 1: 같은 값 할당
x = y = z = 0
print(x, y, z)  # 출력: 0 0 0

# 방법 2: 다른 값 할당
name, age, city = "김철수", 30, "서울"
print(name)  # 출력: 김철수
print(age)   # 출력: 30
print(city)  # 출력: 서울
```

---

### 📝 섹션 1 퀴즈

**1. [OX] Python에서 변수명은 숫자로 시작할 수 있다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** 변수명은 숫자로 시작할 수 없습니다. 영문자나 언더스코어(_)로 시작해야 합니다.
</details>

**2. [객관식] Python에서 권장하는 변수 네이밍 컨벤션은?**
   - ① camelCase
   - ② PascalCase
   - ③ snake_case
   - ④ kebab-case

<details>
<summary>정답 보기</summary>
**③ snake_case**

Python에서는 PEP 8 스타일 가이드에 따라 변수명에 snake_case를 사용합니다.
</details>

**3. [객관식] 다음 중 올바른 변수명은?**
   - ① my-name
   - ② 2nd_place
   - ③ _private_var
   - ④ for

<details>
<summary>정답 보기</summary>
**③ _private_var**

- ①은 하이픈(-)을 사용할 수 없음
- ②는 숫자로 시작할 수 없음
- ④는 Python 예약어라서 사용 불가
- ③은 언더스코어로 시작 가능
</details>

**4. [OX] 변수에 저장된 값은 변경할 수 없다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** Python의 변수는 언제든지 새로운 값으로 변경할 수 있습니다.
```python
count = 10
count = 20  # 변경 가능
```
</details>

**5. [단답형] `id()` 함수는 무엇을 반환하나요?**
<details>
<summary>정답 보기</summary>
**변수가 저장된 메모리 주소**

`id()` 함수는 객체의 고유 식별자(메모리 주소)를 반환합니다.
</details>

---

## 2. 숫자형 자료형

Python에서 숫자를 다루는 세 가지 자료형이 있습니다.

### 2.1 정수형 (int)

정수(Integer)는 소수점이 없는 숫자입니다.

```python
# 양수
positive_num = 100

# 음수
negative_num = -50

# 0
zero = 0

print(type(positive_num))  # 출력: <class 'int'>
```

**큰 숫자 표현하기:**
```python
# 언더스코어로 자릿수 구분 (가독성 향상)
population = 51_000_000  # 5천1백만
print(population)  # 출력: 51000000

# 실행에는 영향 없음 - 읽기 편하게 하기 위함
price = 1_500_000
print(price)  # 출력: 1500000
```

### 2.2 실수형 (float)

실수(Float)는 소수점이 있는 숫자입니다.

```python
# 일반 실수
height = 175.5
temperature = -12.8
pi = 3.141592

print(type(height))  # 출력: <class 'float'>

# 지수 표현법 (Scientific Notation)
scientific = 1.5e3  # 1.5 × 10³ = 1500.0
print(scientific)   # 출력: 1500.0

small_num = 1.5e-3  # 1.5 × 10⁻³ = 0.0015
print(small_num)    # 출력: 0.0015
```

### 2.3 복소수 (complex)

복소수는 실수부와 허수부로 구성됩니다. 허수 단위는 `j`를 사용합니다.

```python
# 복소수 생성
complex_num = 3 + 4j
print(complex_num)  # 출력: (3+4j)
print(type(complex_num))  # 출력: <class 'complex'>

# 실수부와 허수부 확인
print(complex_num.real)  # 출력: 3.0 (실수부)
print(complex_num.imag)  # 출력: 4.0 (허수부)
```

> **참고**: 복소수는 일반적인 프로그래밍에서는 잘 사용되지 않지만, 과학 계산이나 신호 처리에서 중요합니다.

### 2.4 산술 연산자

Python에서 사용 가능한 산술 연산자입니다.

| 연산자 | 의미 | 예시 | 결과 |
|--------|------|------|------|
| `+` | 덧셈 | `5 + 3` | `8` |
| `-` | 뺄셈 | `5 - 3` | `2` |
| `*` | 곱셈 | `5 * 3` | `15` |
| `/` | 나눗셈 | `5 / 2` | `2.5` |
| `//` | 몫 | `5 // 2` | `2` |
| `%` | 나머지 | `5 % 2` | `1` |
| `**` | 거듭제곱 | `5 ** 2` | `25` |

**예제:**
```python
# 기본 연산
a = 10
b = 3

print(a + b)   # 출력: 13
print(a - b)   # 출력: 7
print(a * b)   # 출력: 30
print(a / b)   # 출력: 3.3333333333333335
print(a // b)  # 출력: 3 (몫)
print(a % b)   # 출력: 1 (나머지)
print(a ** b)  # 출력: 1000 (10의 3승)
```

### 2.5 복합 대입 연산자

연산과 대입을 동시에 수행하는 연산자입니다.

```python
number = 10

# 일반 방식
number = number + 5
print(number)  # 출력: 15

# 복합 대입 연산자 사용
number = 10
number += 5  # number = number + 5 와 동일
print(number)  # 출력: 15

# 다양한 복합 대입 연산자
number -= 3  # number = number - 3
print(number)  # 출력: 12

number *= 2  # number = number * 2
print(number)  # 출력: 24

number /= 4  # number = number / 4
print(number)  # 출력: 6.0
```

### 2.6 숫자형 내장 함수

Python에서 제공하는 유용한 숫자 관련 함수들입니다.

```python
# abs() - 절대값
print(abs(-10))     # 출력: 10
print(abs(3.14))    # 출력: 3.14

# round() - 반올림
print(round(3.7))      # 출력: 4
print(round(3.14159, 2))  # 출력: 3.14 (소수점 둘째 자리까지)

# pow() - 거듭제곱
print(pow(2, 3))    # 출력: 8 (2의 3승)
print(pow(5, 2))    # 출력: 25

# max(), min() - 최대값, 최소값
print(max(10, 20, 30))  # 출력: 30
print(min(10, 20, 30))  # 출력: 10

# sum() - 합계 (리스트나 튜플 사용)
numbers = [1, 2, 3, 4, 5]
print(sum(numbers))     # 출력: 15
```

### 2.7 math 모듈 활용

더 많은 수학 함수를 사용하려면 `math` 모듈을 import 해야 합니다.

```python
import math

# 제곱근
print(math.sqrt(16))    # 출력: 4.0

# 올림, 내림
print(math.ceil(3.2))   # 출력: 4 (올림)
print(math.floor(3.8))  # 출력: 3 (내림)

# 원주율과 자연상수
print(math.pi)          # 출력: 3.141592653589793
print(math.e)           # 출력: 2.718281828459045

# 삼각함수 (라디안 단위)
print(math.sin(math.pi / 2))  # 출력: 1.0
print(math.cos(0))            # 출력: 1.0
```

---

### 📝 섹션 2 퀴즈

**1. [객관식] 다음 중 정수(int)가 아닌 것은?**
   - ① 100
   - ② -50
   - ③ 3.14
   - ④ 0

<details>
<summary>정답 보기</summary>
**③ 3.14**

3.14는 소수점이 있으므로 실수(float)입니다.
</details>

**2. [OX] `10 / 3`의 결과는 정수이다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** `/` 연산자는 항상 실수(float)를 반환합니다. 결과는 3.333...이 됩니다. 정수 몫을 구하려면 `//` 연산자를 사용해야 합니다.
</details>

**3. [객관식] `5 ** 2`의 결과는?**
   - ① 7
   - ② 10
   - ③ 25
   - ④ 오류 발생

<details>
<summary>정답 보기</summary>
**③ 25**

`**`는 거듭제곱 연산자입니다. 5²= 25
</details>

**4. [단답형] 복소수에서 허수 단위를 나타내는 문자는?**
<details>
<summary>정답 보기</summary>
**j**

Python에서는 허수 단위를 `j`로 표현합니다. (예: `3 + 4j`)
</details>

**5. [객관식] `round(3.7)`의 결과는?**
   - ① 3
   - ② 3.0
   - ③ 4
   - ④ 4.0

<details>
<summary>정답 보기</summary>
**③ 4**

`round()` 함수는 반올림한 정수를 반환합니다. 3.7은 4로 반올림됩니다.
</details>

---

## 3. 문자열(str) 자료형

문자열은 문자들의 순서 있는 집합입니다.

### 3.1 문자열 선언 방법

Python에서는 작은따옴표(`'`) 또는 큰따옴표(`"`)로 문자열을 만들 수 있습니다.

```python
# 작은따옴표
str1 = 'Hello'

# 큰따옴표
str2 = "World"

# 둘 다 동일하게 동작
print(str1)  # 출력: Hello
print(str2)  # 출력: World
```

**따옴표 안에 따옴표 사용하기:**
```python
# 큰따옴표 안에 작은따옴표
sentence1 = "It's a beautiful day"
print(sentence1)  # 출력: It's a beautiful day

# 작은따옴표 안에 큰따옴표
sentence2 = 'He said "Hello"'
print(sentence2)  # 출력: He said "Hello"

# 이스케이프 문자 사용
sentence3 = 'It\'s a beautiful day'
print(sentence3)  # 출력: It's a beautiful day
```

**여러 줄 문자열:**
```python
# 삼중 따옴표 사용
multiline = """첫 번째 줄
두 번째 줄
세 번째 줄"""
print(multiline)

# 작은 따옴표도 가능
multiline2 = '''Hello
World
Python'''
print(multiline2)
```

### 3.2 이스케이프 문자

특수한 의미를 가진 문자들입니다.

| 이스케이프 문자 | 의미 |
|----------------|------|
| `\n` | 줄바꿈 (New Line) |
| `\t` | 탭 (Tab) |
| `\\` | 백슬래시 자체 |
| `\'` | 작은따옴표 자체 |
| `\"` | 큰따옴표 자체 |

```python
# 줄바꿈
print("첫 번째 줄\n두 번째 줄")
# 출력:
# 첫 번째 줄
# 두 번째 줄

# 탭
print("이름:\t홍길동")
# 출력: 이름:	홍길동

# 백슬래시
print("파일 경로: C:\\Users\\Python")
# 출력: 파일 경로: C:\Users\Python
```

### 3.3 문자열 연산

```python
# 문자열 연결 (concatenation)
str1 = "Hello"
str2 = "World"
result = str1 + " " + str2
print(result)  # 출력: Hello World

# 문자열 반복
repeat = "Python! " * 3
print(repeat)  # 출력: Python! Python! Python! 

# 문자열 길이
text = "안녕하세요"
print(len(text))  # 출력: 5
```

### 3.4 인덱싱 (Indexing)

문자열의 각 문자는 **인덱스(위치 번호)**로 접근할 수 있습니다.

```python
text = "Python"

# 인덱스는 0부터 시작
print(text[0])  # 출력: P
print(text[1])  # 출력: y
print(text[2])  # 출력: t

# 음수 인덱스 (뒤에서부터)
print(text[-1])  # 출력: n (마지막 문자)
print(text[-2])  # 출력: o (뒤에서 두 번째)
```

**인덱스 개념도:**
```
문자열: P  y  t  h  o  n
인덱스: 0  1  2  3  4  5
음수:  -6 -5 -4 -3 -2 -1
```

### 3.5 슬라이싱 (Slicing)

문자열의 일부분을 추출하는 기능입니다.

**기본 문법:** `문자열[시작:끝:간격]`

```python
text = "Python Programming"

# 기본 슬라이싱
print(text[0:6])    # 출력: Python (인덱스 0~5)
print(text[7:18])   # 출력: Programming

# 시작 생략 (처음부터)
print(text[:6])     # 출력: Python

# 끝 생략 (끝까지)
print(text[7:])     # 출력: Programming

# 전체 복사
print(text[:])      # 출력: Python Programming

# 간격 지정
print(text[::2])    # 출력: Pto rgamn (2칸씩 건너뛰기)

# 역순으로 출력
print(text[::-1])   # 출력: gnimmargorP nohtyP
```

**슬라이싱 연습:**
```python
sentence = "Hello World"

print(sentence[0:5])    # 출력: Hello
print(sentence[6:11])   # 출력: World
print(sentence[6:])     # 출력: World
print(sentence[:5])     # 출력: Hello
print(sentence[-5:])    # 출력: World (뒤에서 5글자)
```

### 3.6 문자열 메서드

문자열 객체가 제공하는 유용한 함수들입니다.

```python
text = "python programming"

# upper() - 대문자로 변환
print(text.upper())  # 출력: PYTHON PROGRAMMING

# lower() - 소문자로 변환
text2 = "HELLO WORLD"
print(text2.lower())  # 출력: hello world

# capitalize() - 첫 글자만 대문자
print(text.capitalize())  # 출력: Python programming

# title() - 각 단어의 첫 글자를 대문자로
print(text.title())  # 출력: Python Programming

# strip() - 양쪽 공백 제거
text3 = "  hello  "
print(text3.strip())  # 출력: hello

# replace() - 문자열 치환
text4 = "Hello World"
print(text4.replace("World", "Python"))  # 출력: Hello Python

# split() - 문자열 분리 (리스트로 반환)
text5 = "apple,banana,orange"
print(text5.split(","))  # 출력: ['apple', 'banana', 'orange']

# count() - 특정 문자 개수 세기
text6 = "hello world"
print(text6.count("l"))  # 출력: 3

# find() - 특정 문자의 위치 찾기
print(text6.find("world"))  # 출력: 6 (시작 위치)
print(text6.find("python"))  # 출력: -1 (없으면 -1 반환)

# startswith(), endswith() - 시작/끝 문자열 확인
print(text6.startswith("hello"))  # 출력: True
print(text6.endswith("world"))    # 출력: True
```

### 3.7 문자열 포맷팅

문자열에 변수 값을 삽입하는 방법입니다.

#### 방법 1: % 포맷팅 (옛날 방식)
```python
name = "홍길동"
age = 25

print("제 이름은 %s이고, 나이는 %d살입니다." % (name, age))
# 출력: 제 이름은 홍길동이고, 나이는 25살입니다.
```

#### 방법 2: format() 메서드
```python
name = "김철수"
age = 30

print("제 이름은 {}이고, 나이는 {}살입니다.".format(name, age))
# 출력: 제 이름은 김철수이고, 나이는 30살입니다.

# 인덱스 지정
print("나이: {1}, 이름: {0}".format(name, age))
# 출력: 나이: 30, 이름: 김철수
```

#### 방법 3: f-string (Python 3.6+, 권장)
```python
name = "이영희"
age = 28
height = 165.5

print(f"제 이름은 {name}이고, 나이는 {age}살입니다.")
# 출력: 제 이름은 이영희이고, 나이는 28살입니다.

# 수식 계산 가능
print(f"10년 후 나이: {age + 10}살")
# 출력: 10년 후 나이: 38살

# 소수점 자리 지정
print(f"키: {height:.1f}cm")
# 출력: 키: 165.5cm
```

**f-string 고급 기능:**
```python
# 정렬
name = "Python"
print(f"{name:>10}")  # 출력:     Python (오른쪽 정렬, 총 10자리)
print(f"{name:<10}")  # 출력: Python     (왼쪽 정렬)
print(f"{name:^10}")  # 출력:   Python   (가운데 정렬)

# 천 단위 구분자
price = 1500000
print(f"가격: {price:,}원")  # 출력: 가격: 1,500,000원

# 퍼센트 표시
ratio = 0.856
print(f"비율: {ratio:.1%}")  # 출력: 비율: 85.6%
```

---

### 📝 섹션 3 퀴즈

**1. [객관식] 문자열 "Python"에서 첫 번째 문자를 가져오는 방법은?**
   - ① `"Python"[1]`
   - ② `"Python"[0]`
   - ③ `"Python"[-1]`
   - ④ `"Python".first()`

<details>
<summary>정답 보기</summary>
**② `"Python"[0]`**

Python의 인덱스는 0부터 시작합니다. 첫 번째 문자는 인덱스 0입니다.
</details>

**2. [OX] 문자열 슬라이싱 `text[2:5]`는 인덱스 2, 3, 4, 5의 문자를 가져온다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** 슬라이싱은 시작 인덱스는 포함하지만 끝 인덱스는 포함하지 않습니다. `text[2:5]`는 인덱스 2, 3, 4의 문자만 가져옵니다.
</details>

**3. [객관식] `"hello".upper()`의 결과는?**
   - ① "hello"
   - ② "HELLO"
   - ③ "Hello"
   - ④ 오류 발생

<details>
<summary>정답 보기</summary>
**② "HELLO"**

`upper()` 메서드는 모든 문자를 대문자로 변환합니다.
</details>

**4. [단답형] 문자열 "Python"을 역순으로 만드는 슬라이싱 표현은?**
<details>
<summary>정답 보기</summary>
**`"Python"[::-1]`**

`[::-1]`은 간격을 -1로 지정하여 문자열을 역순으로 만듭니다. 결과: "nohtyP"
</details>

**5. [객관식] f-string에서 변수 name="홍길동"을 출력하는 올바른 방법은?**
   - ① `f"이름: name"`
   - ② `f"이름: {name}"`
   - ③ `f"이름: $name"`
   - ④ `f"이름: [name]"`

<details>
<summary>정답 보기</summary>
**② `f"이름: {name}"`**

f-string에서는 변수를 `{}`로 감싸서 사용합니다.
</details>

---

## 4. 자료형 변환

서로 다른 자료형 간의 변환 방법입니다.

### 4.1 문자열 → 숫자

```python
# 문자열 → 정수
str_num = "123"
int_num = int(str_num)
print(int_num)        # 출력: 123
print(type(int_num))  # 출력: <class 'int'>

# 문자열 → 실수
str_float = "3.14"
float_num = float(str_float)
print(float_num)        # 출력: 3.14
print(type(float_num))  # 출력: <class 'float'>
```

### 4.2 숫자 → 문자열

```python
# 정수 → 문자열
age = 25
str_age = str(age)
print(str_age)        # 출력: 25
print(type(str_age))  # 출력: <class 'str'>

# 실수 → 문자열
pi = 3.14159
str_pi = str(pi)
print(str_pi)        # 출력: 3.14159
print(type(str_pi))  # 출력: <class 'str'>
```

### 4.3 정수 ↔ 실수

```python
# 정수 → 실수
num = 10
float_num = float(num)
print(float_num)  # 출력: 10.0

# 실수 → 정수 (소수점 버림)
pi = 3.14159
int_pi = int(pi)
print(int_pi)  # 출력: 3
```

### 4.4 변환 시 주의사항

```python
# 올바른 변환
print(int("100"))      # 출력: 100
print(float("3.14"))   # 출력: 3.14

# 에러 발생 예시
# print(int("3.14"))   # ValueError: invalid literal
# print(int("Hello"))  # ValueError: invalid literal

# 해결 방법: 먼저 float으로 변환 후 int로 변환
print(int(float("3.14")))  # 출력: 3
```

### 4.5 실전 예제

```python
# 사용자 입력 받기 (input은 항상 문자열 반환)
# age_str = input("나이를 입력하세요: ")
# age = int(age_str)  # 문자열을 정수로 변환
# print(f"10년 후 나이: {age + 10}살")

# 문자열 숫자 계산
num1 = "100"
num2 = "200"
result = int(num1) + int(num2)
print(result)  # 출력: 300

# 잘못된 예 (문자열 연결됨)
wrong = num1 + num2
print(wrong)  # 출력: 100200
```

---

### 📝 섹션 4 퀴즈

**1. [객관식] `int("3.14")`의 결과는?**
   - ① 3
   - ② 3.14
   - ③ 오류 발생
   - ④ "3.14"

<details>
<summary>정답 보기</summary>
**③ 오류 발생 (ValueError)**

문자열 "3.14"를 정수로 직접 변환할 수 없습니다. 먼저 `float()`으로 변환한 후 `int()`를 사용해야 합니다.
```python
int(float("3.14"))  # 3
```
</details>

**2. [OX] `str(100)`의 결과는 문자열 "100"이다.**
<details>
<summary>정답 보기</summary>
⭕ **맞습니다.** `str()` 함수는 숫자를 문자열로 변환합니다.
</details>

**3. [객관식] 실수 3.9를 정수로 변환하면?**
   - ① 3
   - ② 4
   - ③ 3.0
   - ④ 오류 발생

<details>
<summary>정답 보기</summary>
**① 3**

`int()` 함수는 소수점을 버립니다(반올림하지 않음). `int(3.9)`는 3이 됩니다.
</details>

**4. [단답형] 불린 값 `False`를 정수로 변환하면 어떤 값이 되나요?**
<details>
<summary>정답 보기</summary>
**0**

`int(False)`는 0을 반환하고, `int(True)`는 1을 반환합니다.
</details>

**5. [OX] 빈 문자열 ""를 bool()로 변환하면 True가 된다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** 빈 문자열은 `False`로 평가됩니다. `bool("")`는 False입니다.
</details>

---

## 5. 내장 함수와 type() 활용

### 5.1 type() 함수

변수의 자료형을 확인하는 함수입니다.

```python
# 다양한 자료형 확인
num = 100
print(type(num))  # 출력: <class 'int'>

pi = 3.14
print(type(pi))   # 출력: <class 'float'>

name = "Python"
print(type(name))  # 출력: <class 'str'>

flag = True
print(type(flag))  # 출력: <class 'bool'>
```

### 5.2 isinstance() 함수

특정 자료형인지 확인하는 함수입니다.

```python
num = 100

# 정수인지 확인
print(isinstance(num, int))     # 출력: True
print(isinstance(num, str))     # 출력: False

# 여러 자료형 중 하나인지 확인
print(isinstance(num, (int, float)))  # 출력: True
```

### 5.3 자주 사용하는 내장 함수

```python
# print() - 출력
print("Hello, World!")

# input() - 입력 받기 (문자열로 반환)
# name = input("이름을 입력하세요: ")

# len() - 길이
text = "Python"
print(len(text))  # 출력: 6

# range() - 숫자 범위 생성
for i in range(5):
    print(i)  # 출력: 0 1 2 3 4

# dir() - 객체의 속성과 메서드 목록
text = "hello"
print(dir(text))  # 문자열이 사용할 수 있는 모든 메서드 출력

# help() - 도움말
# help(str)  # 문자열 자료형에 대한 도움말
```

### 5.4 변수 확인 함수

```python
# id() - 메모리 주소 확인
x = 100
print(id(x))

# 같은 값은 같은 메모리 주소를 가짐 (작은 정수의 경우)
y = 100
print(id(x) == id(y))  # 출력: True

# isinstance()로 자료형 검증
def process_number(value):
    if isinstance(value, (int, float)):
        return value * 2
    else:
        return "숫자가 아닙니다"

print(process_number(10))      # 출력: 20
print(process_number("hello")) # 출력: 숫자가 아닙니다
```

---

### 📝 섹션 5 퀴즈

**1. [객관식] `type(3.14)`의 결과는?**
   - ① `<class 'int'>`
   - ② `<class 'float'>`
   - ③ `<class 'str'>`
   - ④ `3.14`

<details>
<summary>정답 보기</summary>
**② `<class 'float'>`**

`type()` 함수는 변수의 자료형을 반환합니다. 3.14는 실수형입니다.
</details>

**2. [OX] `len()` 함수는 문자열의 길이만 측정할 수 있다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** `len()` 함수는 문자열뿐만 아니라 리스트, 튜플, 딕셔너리 등 다양한 자료형의 길이를 측정할 수 있습니다.
</details>

**3. [객관식] `range(1, 10, 2)`가 생성하는 숫자는?**
   - ① 1, 2, 3, ..., 10
   - ② 1, 3, 5, 7, 9
   - ③ 2, 4, 6, 8, 10
   - ④ 1, 2, 10

<details>
<summary>정답 보기</summary>
**② 1, 3, 5, 7, 9**

`range(시작, 끝, 간격)`에서 1부터 10 미만까지 2씩 증가하는 숫자를 생성합니다.
</details>

**4. [단답형] 특정 변수가 정수형인지 확인하는 함수는?**
<details>
<summary>정답 보기</summary>
**`isinstance(변수, int)`**

예: `isinstance(100, int)`는 True를 반환합니다.
</details>

**5. [객관식] `max(10, 5, 20, 15)`의 결과는?**
   - ① 10
   - ② 5
   - ③ 20
   - ④ 15

<details>
<summary>정답 보기</summary>
**③ 20**

`max()` 함수는 인자 중 가장 큰 값을 반환합니다.
</details>

---

## 6. 주석 활용법

주석(Comment)은 코드에 설명을 추가하는 방법으로, 프로그램 실행에는 영향을 주지 않습니다.

### 6.1 한 줄 주석 (#)

```python
# 이것은 주석입니다
print("Hello")  # 코드 옆에도 주석 가능

# 변수 선언
name = "홍길동"  # 학생 이름
age = 25         # 학생 나이
```

### 6.2 여러 줄 주석 (""" """ 또는 ''' ''')

```python
"""
이것은 여러 줄 주석입니다.
프로그램의 설명이나
긴 내용을 작성할 때 사용합니다.
"""

def calculate_sum(a, b):
    """
    두 수의 합을 계산하는 함수
    
    Parameters:
        a (int/float): 첫 번째 숫자
        b (int/float): 두 번째 숫자
    
    Returns:
        int/float: 두 수의 합
    """
    return a + b
```

### 6.3 주석 작성 모범 사례

```python
# 좋은 주석 - 코드의 "왜"를 설명
# 사용자 나이가 19세 이상인지 확인 (성인 인증용)
is_adult = age >= 19

# 나쁜 주석 - 코드가 "무엇을 하는지" 그대로 반복
# age가 19 이상이면 True
is_adult = age >= 19

# 좋은 주석 - 복잡한 로직 설명
# 윤년 계산: 4의 배수이면서 100의 배수가 아니거나, 400의 배수
is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# 좋은 주석 - 임시 해결책이나 개선 필요한 부분 표시
# TODO: 나중에 데이터베이스 연동으로 변경 필요
users = ["홍길동", "김철수", "이영희"]

# FIXME: 음수 입력 시 오류 발생, 예외 처리 추가 필요
result = calculate_sum(a, b)
```

### 6.4 주석을 피해야 하는 경우

```python
# 나쁜 예 - 너무 많은 주석
# x에 10을 할당
x = 10
# x를 출력
print(x)
# y에 20을 할당
y = 20

# 좋은 예 - 의미 있는 변수명으로 주석 대체
# 나쁜 예
s = 3600  # 초 단위

# 좋은 예 (주석 없이도 이해 가능)
seconds_in_hour = 3600
```

### 6.5 docstring (문서화 문자열)

함수, 클래스, 모듈의 설명을 작성하는 특별한 주석입니다.

```python
def greet(name, age):
    """
    사용자에게 인사하는 함수
    
    Args:
        name (str): 사용자 이름
        age (int): 사용자 나이
    
    Returns:
        str: 인사 메시지
    
    Examples:
        >>> greet("홍길동", 25)
        '안녕하세요, 홍길동님! (25세)'
    """
    return f"안녕하세요, {name}님! ({age}세)"

# docstring 확인
print(greet.__doc__)
```

---

### 📝 섹션 6 퀴즈

**1. [OX] 주석은 프로그램 실행에 영향을 준다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** 주석은 프로그램 실행에 전혀 영향을 주지 않습니다. 코드 설명용으로만 사용됩니다.
</details>

**2. [객관식] 한 줄 주석을 작성하는 기호는?**
   - ① `//`
   - ② `#`
   - ③ `/* */`
   - ④ `--`

<details>
<summary>정답 보기</summary>
**② `#`**

Python에서는 `#`으로 한 줄 주석을 작성합니다.
</details>

**3. [객관식] 여러 줄 주석을 작성하는 방법은?**
   - ① `# ... #`
   - ② `/* ... */`
   - ③ `""" ... """`
   - ④ `-- ... --`

<details>
<summary>정답 보기</summary>
**③ `""" ... """`**

삼중 따옴표(`"""` 또는 `'''`)로 여러 줄 주석을 작성할 수 있습니다.
</details>

**4. [OX] 좋은 주석은 코드가 "무엇을 하는지"를 설명한다.**
<details>
<summary>정답 보기</summary>
❌ **틀렸습니다.** 좋은 주석은 코드의 "왜(Why)"를 설명해야 합니다. "무엇을(What)" 하는지는 코드 자체로 명확해야 합니다.
</details>

**5. [단답형] 함수나 클래스의 설명을 작성하는 특별한 주석을 무엇이라고 하나요?**
<details>
<summary>정답 보기</summary>
**docstring (문서화 문자열)**

함수 정의 바로 아래에 삼중 따옴표로 작성하며, `__doc__` 속성이나 `help()` 함수로 확인할 수 있습니다.
</details>

---

## 📝 학습 마무리

### 핵심 요약

1. **변수**: 데이터를 저장하는 공간, 스네이크 케이스 사용
2. **숫자형**: int(정수), float(실수), complex(복소수)
3. **문자열**: 따옴표로 선언, 인덱싱/슬라이싱으로 접근
4. **자료형 변환**: int(), float(), str() 함수 사용
5. **내장 함수**: type(), len(), isinstance() 등
6. **주석**: # 또는 """ """로 코드 설명 추가

### 다음 단계

이제 `practice.py` 또는 `practice.ipynb` 파일을 열어 실습 문제를 풀어보세요!

- 각 섹션별 예제 코드 실행
- 연습 문제 풀이
- 직접 코드를 수정하고 실험하기

---

**학습 시간 기록:**
- [ ] 섹션 1: 변수 기초 (30분)
- [ ] 섹션 2: 숫자형 자료형 (30분)
- [ ] 섹션 3: 문자열 자료형 (40분)
- [ ] 섹션 4: 자료형 변환 (20분)
- [ ] 섹션 5: 내장 함수 (15분)
- [ ] 섹션 6: 주석 (15분)
- [ ] 실습 문제 풀이 (1시간 30분)

**총 학습 시간: 약 3시간 30분**
