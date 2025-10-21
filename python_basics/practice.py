"""
Python 기초 - 변수와 자료형 실습 파일

이 파일은 theory.md와 함께 사용하는 실습 코드입니다.
각 섹션별로 예제 코드와 연습 문제가 포함되어 있습니다.

실행 방법:
    python practice.py

또는 섹션별로 실행하고 싶다면 해당 섹션의 함수만 호출하세요.
"""


def section1_variables():
    """
    섹션 1: 변수 선언 기초
    예상 시간: 30분 (이론 15분 + 실습 15분)
    """
    print("=" * 60)
    print("섹션 1: 파이썬 기초 및 변수 선언 개념")
    print("=" * 60)
    
    # 예제 1: 변수 선언과 출력
    print("\n[예제 1] 변수 선언과 출력")
    name = "홍길동"
    age = 25
    height = 175.5
    print(f"이름: {name}")
    print(f"나이: {age}세")
    print(f"키: {height}cm")
    
    # 예제 2: 변수 값 변경
    print("\n[예제 2] 변수 값 변경")
    count = 10
    print(f"초기값: {count}")
    count = 20
    print(f"변경 후: {count}")
    count = count + 5
    print(f"계산 후: {count}")
    
    # 예제 3: 여러 변수 동시 선언
    print("\n[예제 3] 여러 변수 동시 선언")
    x = y = z = 0
    print(f"x={x}, y={y}, z={z}")
    
    name, age, city = "김철수", 30, "서울"
    print(f"이름: {name}, 나이: {age}, 도시: {city}")
    
    # 예제 4: 변수 이름 규칙
    print("\n[예제 4] 올바른 변수 이름")
    user_name = "이영희"
    student_age = 22
    total_price = 15000
    _private_var = 100
    print(f"사용자명: {user_name}")
    print(f"학생 나이: {student_age}")
    print(f"총 가격: {total_price}원")
    
    # 예제 5: 메모리 주소 확인
    print("\n[예제 5] 메모리 주소 확인")
    a = 100
    b = 100
    print(f"a의 메모리 주소: {id(a)}")
    print(f"b의 메모리 주소: {id(b)}")
    print(f"같은 주소인가? {id(a) == id(b)}")
    
    print("\n" + "-" * 60)
    print("연습 문제 1 - 변수 선언")
    print("-" * 60)
    
    # 연습 문제는 직접 작성해보세요!
    print("""
연습 문제:
1. student_name이라는 변수에 자신의 이름을 저장하고 출력하세요.
2. student_id라는 변수에 학번(예: 20230001)을 저장하고 출력하세요.
3. gpa라는 변수에 학점(예: 4.5)을 저장하고 출력하세요.
4. 위 세 변수를 f-string으로 한 줄에 출력하세요.
5. gpa 변수의 값을 0.3 증가시키고 출력하세요.

아래에 코드를 작성하세요:
""")
    
    # 여기서부터 연습 문제 풀이 공간
    # TODO: 학생들이 직접 작성
    
    print("\n")


def section1_practice_solution():
    """섹션 1 연습 문제 해설"""
    print("=" * 60)
    print("섹션 1 연습 문제 해설")
    print("=" * 60)
    
    # 1. 이름 저장
    student_name = "김철수"
    print(f"1. 학생 이름: {student_name}")
    
    # 2. 학번 저장
    student_id = 20230001
    print(f"2. 학번: {student_id}")
    
    # 3. 학점 저장
    gpa = 4.5
    print(f"3. 학점: {gpa}")
    
    # 4. 한 줄에 출력
    print(f"4. {student_name}님의 학번은 {student_id}이고, 학점은 {gpa}입니다.")
    
    # 5. 학점 증가
    gpa = gpa + 0.3
    # 또는 gpa += 0.3
    print(f"5. 증가된 학점: {gpa}")
    
    print("\n")


def section2_numbers():
    """
    섹션 2: 숫자형 자료형
    예상 시간: 40분 (이론 20분 + 실습 20분)
    """
    print("=" * 60)
    print("섹션 2: 숫자형 자료형")
    print("=" * 60)
    
    # 예제 1: 정수형 (int)
    print("\n[예제 1] 정수형 (int)")
    positive = 100
    negative = -50
    zero = 0
    large_num = 51_000_000  # 5천1백만
    print(f"양수: {positive}")
    print(f"음수: {negative}")
    print(f"0: {zero}")
    print(f"큰 숫자: {large_num:,}")
    print(f"자료형: {type(positive)}")
    
    # 예제 2: 실수형 (float)
    print("\n[예제 2] 실수형 (float)")
    pi = 3.141592
    temperature = -12.8
    scientific = 1.5e3  # 1500.0
    print(f"원주율: {pi}")
    print(f"온도: {temperature}°C")
    print(f"과학적 표기법: {scientific}")
    print(f"자료형: {type(pi)}")
    
    # 예제 3: 복소수 (complex)
    print("\n[예제 3] 복소수 (complex)")
    complex_num = 3 + 4j
    print(f"복소수: {complex_num}")
    print(f"실수부: {complex_num.real}")
    print(f"허수부: {complex_num.imag}")
    
    # 예제 4: 산술 연산
    print("\n[예제 4] 산술 연산")
    a, b = 10, 3
    print(f"a = {a}, b = {b}")
    print(f"덧셈: {a} + {b} = {a + b}")
    print(f"뺄셈: {a} - {b} = {a - b}")
    print(f"곱셈: {a} * {b} = {a * b}")
    print(f"나눗셈: {a} / {b} = {a / b}")
    print(f"몫: {a} // {b} = {a // b}")
    print(f"나머지: {a} % {b} = {a % b}")
    print(f"거듭제곱: {a} ** {b} = {a ** b}")
    
    # 예제 5: 복합 대입 연산자
    print("\n[예제 5] 복합 대입 연산자")
    number = 10
    print(f"초기값: {number}")
    number += 5
    print(f"+= 5: {number}")
    number -= 3
    print(f"-= 3: {number}")
    number *= 2
    print(f"*= 2: {number}")
    number /= 4
    print(f"/= 4: {number}")
    
    # 예제 6: 내장 함수
    print("\n[예제 6] 숫자 내장 함수")
    print(f"abs(-10) = {abs(-10)}")
    print(f"round(3.7) = {round(3.7)}")
    print(f"round(3.14159, 2) = {round(3.14159, 2)}")
    print(f"pow(2, 3) = {pow(2, 3)}")
    print(f"max(10, 20, 30) = {max(10, 20, 30)}")
    print(f"min(10, 20, 30) = {min(10, 20, 30)}")
    
    # 예제 7: math 모듈
    print("\n[예제 7] math 모듈 활용")
    import math
    print(f"sqrt(16) = {math.sqrt(16)}")
    print(f"ceil(3.2) = {math.ceil(3.2)}")
    print(f"floor(3.8) = {math.floor(3.8)}")
    print(f"pi = {math.pi}")
    print(f"e = {math.e}")
    
    print("\n" + "-" * 60)
    print("연습 문제 2 - 숫자형 자료형")
    print("-" * 60)
    
    print("""
연습 문제:
1. 두 개의 정수 변수 num1=15, num2=4를 선언하고 사칙연산 결과를 출력하세요.
2. 원의 반지름을 저장하는 변수 radius=5.0을 만들고, 원의 넓이를 계산하세요.
   (넓이 = π × r²)
3. 온도를 섭씨(celsius)로 입력받아 화씨(fahrenheit)로 변환하세요.
   (화씨 = 섭씨 × 9/5 + 32)
4. 1부터 10까지의 숫자 중 짝수의 합을 계산하세요. (2+4+6+8+10)
5. 금액 10000원을 3명이 나눠 가질 때, 1인당 금액과 나머지를 계산하세요.

아래에 코드를 작성하세요:
""")
    
    print("\n")


def section2_practice_solution():
    """섹션 2 연습 문제 해설"""
    print("=" * 60)
    print("섹션 2 연습 문제 해설")
    print("=" * 60)
    
    import math
    
    # 1. 사칙연산
    num1, num2 = 15, 4
    print(f"1. {num1} + {num2} = {num1 + num2}")
    print(f"   {num1} - {num2} = {num1 - num2}")
    print(f"   {num1} * {num2} = {num1 * num2}")
    print(f"   {num1} / {num2} = {num1 / num2}")
    
    # 2. 원의 넓이
    radius = 5.0
    area = math.pi * radius ** 2
    print(f"\n2. 반지름 {radius}인 원의 넓이: {area:.2f}")
    
    # 3. 섭씨를 화씨로 변환
    celsius = 25
    fahrenheit = celsius * 9/5 + 32
    print(f"\n3. 섭씨 {celsius}°C = 화씨 {fahrenheit}°F")
    
    # 4. 짝수의 합
    even_sum = 2 + 4 + 6 + 8 + 10
    print(f"\n4. 1부터 10까지 짝수의 합: {even_sum}")
    # 또는 리스트 사용
    even_sum2 = sum([2, 4, 6, 8, 10])
    print(f"   (다른 방법) 짝수의 합: {even_sum2}")
    
    # 5. 금액 나누기
    total_money = 10000
    people = 3
    per_person = total_money // people
    remainder = total_money % people
    print(f"\n5. 1인당 금액: {per_person}원, 나머지: {remainder}원")
    
    print("\n")


def section3_strings():
    """
    섹션 3: 문자열 자료형
    예상 시간: 50분 (이론 25분 + 실습 25분)
    """
    print("=" * 60)
    print("섹션 3: 문자열(str) 자료형")
    print("=" * 60)
    
    # 예제 1: 문자열 선언
    print("\n[예제 1] 문자열 선언")
    str1 = 'Hello'
    str2 = "World"
    str3 = """여러 줄
문자열
예제"""
    print(str1)
    print(str2)
    print(str3)
    
    # 예제 2: 이스케이프 문자
    print("\n[예제 2] 이스케이프 문자")
    print("첫 번째 줄\n두 번째 줄")
    print("이름:\t홍길동")
    print("경로: C:\\Users\\Python")
    print("He said \"Hello\"")
    
    # 예제 3: 문자열 연산
    print("\n[예제 3] 문자열 연산")
    str1 = "Hello"
    str2 = "World"
    print(f"연결: {str1 + ' ' + str2}")
    print(f"반복: {'Python! ' * 3}")
    print(f"길이: len('{str1}') = {len(str1)}")
    
    # 예제 4: 인덱싱
    print("\n[예제 4] 인덱싱")
    text = "Python"
    print(f"text = '{text}'")
    print(f"text[0] = {text[0]}")
    print(f"text[2] = {text[2]}")
    print(f"text[-1] = {text[-1]}")
    print(f"text[-3] = {text[-3]}")
    
    # 예제 5: 슬라이싱
    print("\n[예제 5] 슬라이싱")
    text = "Python Programming"
    print(f"text = '{text}'")
    print(f"text[0:6] = '{text[0:6]}'")
    print(f"text[7:] = '{text[7:]}'")
    print(f"text[:6] = '{text[:6]}'")
    print(f"text[::2] = '{text[::2]}'")
    print(f"text[::-1] = '{text[::-1]}'")
    
    # 예제 6: 문자열 메서드
    print("\n[예제 6] 문자열 메서드")
    text = "python programming"
    print(f"원본: '{text}'")
    print(f"upper(): '{text.upper()}'")
    print(f"capitalize(): '{text.capitalize()}'")
    print(f"title(): '{text.title()}'")
    print(f"count('p'): {text.count('p')}")
    print(f"replace('python', 'java'): '{text.replace('python', 'java')}'")
    
    # 예제 7: 문자열 분리와 결합
    print("\n[예제 7] 문자열 분리와 결합")
    fruits = "apple,banana,orange"
    fruit_list = fruits.split(",")
    print(f"분리: {fruit_list}")
    
    joined = " | ".join(fruit_list)
    print(f"결합: {joined}")
    
    # 예제 8: 문자열 포맷팅
    print("\n[예제 8] 문자열 포맷팅")
    name = "홍길동"
    age = 25
    height = 175.5
    
    # % 포맷팅
    print("제 이름은 %s이고, 나이는 %d살입니다." % (name, age))
    
    # format() 메서드
    print("제 이름은 {}이고, 나이는 {}살입니다.".format(name, age))
    
    # f-string (권장)
    print(f"제 이름은 {name}이고, 나이는 {age}살, 키는 {height}cm입니다.")
    print(f"10년 후 나이: {age + 10}살")
    
    # 예제 9: f-string 고급 기능
    print("\n[예제 9] f-string 고급 기능")
    price = 1500000
    ratio = 0.856
    name = "Python"
    
    print(f"가격: {price:,}원")
    print(f"비율: {ratio:.1%}")
    print(f"정렬: '{name:>10}' (오른쪽)")
    print(f"정렬: '{name:<10}' (왼쪽)")
    print(f"정렬: '{name:^10}' (가운데)")
    
    print("\n" + "-" * 60)
    print("연습 문제 3 - 문자열")
    print("-" * 60)
    
    print("""
연습 문제:
1. "Hello, Python!"이라는 문자열을 변수에 저장하고, 
   - 전체 길이
   - 대문자로 변환
   - 소문자로 변환
   을 출력하세요.

2. email = "user@example.com"에서
   - @를 기준으로 사용자명과 도메인을 분리하세요.
   - 사용자명과 도메인을 각각 출력하세요.

3. sentence = "Python is powerful and Python is easy"에서
   - "Python"이 몇 번 나오는지 세세요.
   - "Python"을 "Java"로 모두 바꾸세요.

4. 이름, 나이, 전화번호를 저장하고 다음 형식으로 출력하세요:
   "이름: 홍길동 | 나이: 25세 | 전화번호: 010-1234-5678"

5. text = "  Python Programming  "의 앞뒤 공백을 제거하고,
   역순으로 출력하세요.

아래에 코드를 작성하세요:
""")
    
    print("\n")


def section3_practice_solution():
    """섹션 3 연습 문제 해설"""
    print("=" * 60)
    print("섹션 3 연습 문제 해설")
    print("=" * 60)
    
    # 1. 문자열 기본 조작
    text = "Hello, Python!"
    print(f"1. 원본: '{text}'")
    print(f"   길이: {len(text)}")
    print(f"   대문자: '{text.upper()}'")
    print(f"   소문자: '{text.lower()}'")
    
    # 2. 이메일 분리
    email = "user@example.com"
    username, domain = email.split("@")
    print(f"\n2. 이메일: {email}")
    print(f"   사용자명: {username}")
    print(f"   도메인: {domain}")
    
    # 3. 문자열 찾기와 치환
    sentence = "Python is powerful and Python is easy"
    print(f"\n3. 원본: '{sentence}'")
    print(f"   'Python' 개수: {sentence.count('Python')}")
    print(f"   치환: '{sentence.replace('Python', 'Java')}'")
    
    # 4. 정보 출력
    name = "홍길동"
    age = 25
    phone = "010-1234-5678"
    print(f"\n4. 이름: {name} | 나이: {age}세 | 전화번호: {phone}")
    
    # 5. 공백 제거와 역순
    text = "  Python Programming  "
    cleaned = text.strip()
    reversed_text = cleaned[::-1]
    print(f"\n5. 원본: '{text}'")
    print(f"   공백 제거: '{cleaned}'")
    print(f"   역순: '{reversed_text}'")
    
    print("\n")


def section4_type_conversion():
    """
    섹션 4: 자료형 변환
    예상 시간: 30분 (이론 15분 + 실습 15분)
    """
    print("=" * 60)
    print("섹션 4: 자료형 변환")
    print("=" * 60)
    
    # 예제 1: 문자열 → 숫자
    print("\n[예제 1] 문자열 → 숫자")
    str_num = "123"
    int_num = int(str_num)
    print(f"문자열: '{str_num}' (type: {type(str_num)})")
    print(f"정수: {int_num} (type: {type(int_num)})")
    
    str_float = "3.14"
    float_num = float(str_float)
    print(f"문자열: '{str_float}' (type: {type(str_float)})")
    print(f"실수: {float_num} (type: {type(float_num)})")
    
    # 예제 2: 숫자 → 문자열
    print("\n[예제 2] 숫자 → 문자열")
    age = 25
    str_age = str(age)
    print(f"정수: {age} (type: {type(age)})")
    print(f"문자열: '{str_age}' (type: {type(str_age)})")
    
    # 예제 3: 정수 ↔ 실수
    print("\n[예제 3] 정수 ↔ 실수")
    num = 10
    float_num = float(num)
    print(f"정수 → 실수: {num} → {float_num}")
    
    pi = 3.14159
    int_pi = int(pi)
    print(f"실수 → 정수: {pi} → {int_pi} (소수점 버림)")
    
    # 예제 4: 변환 시 주의사항
    print("\n[예제 4] 변환 시 주의사항")
    print("올바른 변환:")
    print(f"  int('100') = {int('100')}")
    print(f"  float('3.14') = {float('3.14')}")
    
    print("\n잘못된 변환 (주석 처리됨):")
    print("  # int('3.14')  -> ValueError 발생")
    print("  # int('Hello') -> ValueError 발생")
    
    print("\n해결 방법:")
    print(f"  int(float('3.14')) = {int(float('3.14'))}")
    
    # 예제 5: 실전 예제
    print("\n[예제 5] 실전 예제")
    # 문자열 숫자 더하기
    num1_str = "100"
    num2_str = "200"
    
    # 잘못된 방법
    wrong = num1_str + num2_str
    print(f"잘못된 더하기: '{num1_str}' + '{num2_str}' = '{wrong}'")
    
    # 올바른 방법
    correct = int(num1_str) + int(num2_str)
    print(f"올바른 더하기: {num1_str} + {num2_str} = {correct}")
    
    # 예제 6: 다양한 자료형 변환
    print("\n[예제 6] 다양한 자료형 변환")
    
    # 불린 → 정수
    print(f"int(True) = {int(True)}")
    print(f"int(False) = {int(False)}")
    
    # 정수 → 불린
    print(f"bool(0) = {bool(0)}")
    print(f"bool(1) = {bool(1)}")
    print(f"bool(100) = {bool(100)}")
    
    # 문자열 → 불린
    print(f"bool('') = {bool('')}")
    print(f"bool('Hello') = {bool('Hello')}")
    
    print("\n" + "-" * 60)
    print("연습 문제 4 - 자료형 변환")
    print("-" * 60)
    
    print("""
연습 문제:
1. 문자열 "456"을 정수로 변환하고, 100을 더한 결과를 출력하세요.

2. 문자열 "3.14159"를 실수로 변환하고, 소수점 둘째 자리까지 
   반올림한 결과를 출력하세요.

3. birth_year = "2000"이고 current_year = 2024일 때,
   나이를 계산하고 "당신은 24세입니다." 형식으로 출력하세요.
   (단, birth_year는 문자열이므로 변환 필요)

4. 두 문자열 "12.5"와 "7.3"을 실수로 변환하여 더한 후,
   정수로 변환하여 출력하세요.

5. price = 15000 (정수)을 문자열로 변환하고,
   천 단위 구분자를 넣어 "15,000원" 형식으로 출력하세요.

아래에 코드를 작성하세요:
""")
    
    print("\n")


def section4_practice_solution():
    """섹션 4 연습 문제 해설"""
    print("=" * 60)
    print("섹션 4 연습 문제 해설")
    print("=" * 60)
    
    # 1. 문자열을 정수로 변환하여 계산
    str_num = "456"
    result = int(str_num) + 100
    print(f"1. {str_num} + 100 = {result}")
    
    # 2. 문자열을 실수로 변환하고 반올림
    str_pi = "3.14159"
    pi = float(str_pi)
    rounded_pi = round(pi, 2)
    print(f"\n2. '{str_pi}' → {pi} → {rounded_pi}")
    
    # 3. 나이 계산
    birth_year = "2000"
    current_year = 2024
    age = current_year - int(birth_year)
    print(f"\n3. 당신은 {age}세입니다.")
    
    # 4. 실수 변환 후 정수로
    num1 = "12.5"
    num2 = "7.3"
    sum_float = float(num1) + float(num2)
    sum_int = int(sum_float)
    print(f"\n4. {num1} + {num2} = {sum_float} → {sum_int}")
    
    # 5. 정수를 문자열로 변환하고 포맷팅
    price = 15000
    price_str = str(price)
    formatted = f"{price:,}원"
    print(f"\n5. {price} → '{price_str}' → '{formatted}'")
    
    print("\n")


def section5_builtin_functions():
    """
    섹션 5: 내장 함수와 type() 활용
    예상 시간: 25분 (이론 15분 + 실습 10분)
    """
    print("=" * 60)
    print("섹션 5: 내장 함수와 type() 활용")
    print("=" * 60)
    
    # 예제 1: type() 함수
    print("\n[예제 1] type() 함수")
    num = 100
    pi = 3.14
    name = "Python"
    flag = True
    
    print(f"num = {num}, type: {type(num)}")
    print(f"pi = {pi}, type: {type(pi)}")
    print(f"name = '{name}', type: {type(name)}")
    print(f"flag = {flag}, type: {type(flag)}")
    
    # 예제 2: isinstance() 함수
    print("\n[예제 2] isinstance() 함수")
    num = 100
    print(f"num은 int인가? {isinstance(num, int)}")
    print(f"num은 str인가? {isinstance(num, str)}")
    print(f"num은 (int, float) 중 하나인가? {isinstance(num, (int, float))}")
    
    # 예제 3: len() 함수
    print("\n[예제 3] len() 함수")
    text = "Python"
    numbers = [1, 2, 3, 4, 5]
    print(f"'{text}'의 길이: {len(text)}")
    print(f"{numbers}의 길이: {len(numbers)}")
    
    # 예제 4: range() 함수
    print("\n[예제 4] range() 함수")
    print("range(5):", list(range(5)))
    print("range(1, 6):", list(range(1, 6)))
    print("range(0, 10, 2):", list(range(0, 10, 2)))
    
    # 예제 5: id() 함수
    print("\n[예제 5] id() 함수")
    x = 100
    y = 100
    z = 200
    print(f"x의 id: {id(x)}")
    print(f"y의 id: {id(y)}")
    print(f"z의 id: {id(z)}")
    print(f"x와 y는 같은 객체인가? {id(x) == id(y)}")
    
    # 예제 6: 실용적인 함수들
    print("\n[예제 6] 실용적인 함수들")
    numbers = [10, 5, 8, 3, 15]
    print(f"리스트: {numbers}")
    print(f"최댓값: {max(numbers)}")
    print(f"최솟값: {min(numbers)}")
    print(f"합계: {sum(numbers)}")
    print(f"정렬: {sorted(numbers)}")
    
    # 예제 7: 자료형 검증 함수
    print("\n[예제 7] 자료형 검증 실전 활용")
    
    def process_number(value):
        """숫자만 처리하는 함수"""
        if isinstance(value, (int, float)):
            return f"결과: {value * 2}"
        else:
            return f"오류: '{value}'는 숫자가 아닙니다."
    
    print(process_number(10))
    print(process_number(3.14))
    print(process_number("hello"))
    
    print("\n" + "-" * 60)
    print("연습 문제 5 - 내장 함수")
    print("-" * 60)
    
    print("""
연습 문제:
1. 여러 개의 변수를 만들고 type()으로 자료형을 확인하세요:
   - 정수, 실수, 문자열, 불린 각각 하나씩

2. 숫자 리스트 [3, 7, 1, 9, 4, 6]에서
   - 최댓값, 최솟값, 합계를 구하세요.
   - 정렬된 리스트를 출력하세요.

3. 문자열 "Hello Python World"에서
   - 길이를 구하세요.
   - 공백으로 분리한 단어의 개수를 구하세요.

4. isinstance()를 사용하여 변수 x=100이
   int, float, str 중 어떤 타입인지 확인하세요.

5. range()를 사용하여 1부터 20까지의 홀수를 리스트로 만드세요.

아래에 코드를 작성하세요:
""")
    
    print("\n")


def section5_practice_solution():
    """섹션 5 연습 문제 해설"""
    print("=" * 60)
    print("섹션 5 연습 문제 해설")
    print("=" * 60)
    
    # 1. 자료형 확인
    integer = 100
    floating = 3.14
    string = "Hello"
    boolean = True
    
    print("1. 자료형 확인:")
    print(f"   {integer} → {type(integer)}")
    print(f"   {floating} → {type(floating)}")
    print(f"   '{string}' → {type(string)}")
    print(f"   {boolean} → {type(boolean)}")
    
    # 2. 리스트 통계
    numbers = [3, 7, 1, 9, 4, 6]
    print(f"\n2. 리스트: {numbers}")
    print(f"   최댓값: {max(numbers)}")
    print(f"   최솟값: {min(numbers)}")
    print(f"   합계: {sum(numbers)}")
    print(f"   정렬: {sorted(numbers)}")
    
    # 3. 문자열 분석
    text = "Hello Python World"
    words = text.split()
    print(f"\n3. 문자열: '{text}'")
    print(f"   길이: {len(text)}")
    print(f"   단어 수: {len(words)}")
    
    # 4. isinstance() 확인
    x = 100
    print(f"\n4. x = {x}")
    print(f"   int인가? {isinstance(x, int)}")
    print(f"   float인가? {isinstance(x, float)}")
    print(f"   str인가? {isinstance(x, str)}")
    
    # 5. range()로 홀수 생성
    odd_numbers = list(range(1, 21, 2))
    print(f"\n5. 1부터 20까지의 홀수: {odd_numbers}")
    
    print("\n")


def section6_comments():
    """
    섹션 6: 주석 활용법
    예상 시간: 20분 (이론 10분 + 실습 10분)
    """
    print("=" * 60)
    print("섹션 6: 주석 활용법")
    print("=" * 60)
    
    # 예제 1: 한 줄 주석
    print("\n[예제 1] 한 줄 주석")
    # 이것은 주석입니다
    name = "홍길동"  # 학생 이름
    age = 25         # 학생 나이
    print(f"이름: {name}, 나이: {age}")
    
    # 예제 2: 여러 줄 주석
    print("\n[예제 2] 여러 줄 주석")
    """
    이것은 여러 줄 주석입니다.
    함수나 클래스의 설명을 작성할 때 사용합니다.
    """
    score = 95
    print(f"점수: {score}")
    
    # 예제 3: 코드 설명 주석
    print("\n[예제 3] 좋은 주석 vs 나쁜 주석")
    
    # 나쁜 예 - 코드를 그대로 반복
    # x에 10을 할당
    x = 10
    
    # 좋은 예 - 코드의 의도를 설명
    # 사용자 나이가 19세 이상인지 확인 (성인 인증)
    is_adult = age >= 19
    print(f"성인 여부: {is_adult}")
    
    # 예제 4: TODO와 FIXME
    print("\n[예제 4] TODO와 FIXME 주석")
    
    # TODO: 나중에 데이터베이스 연동으로 변경
    users = ["홍길동", "김철수", "이영희"]
    
    # FIXME: 음수 입력 시 오류 발생, 예외 처리 필요
    price = 10000
    
    print(f"사용자 목록: {users}")
    print(f"가격: {price}원")
    
    # 예제 5: docstring
    print("\n[예제 5] Docstring (함수 문서화)")
    
    def calculate_area(width, height):
        """
        직사각형의 넓이를 계산하는 함수
        
        Parameters:
            width (float): 가로 길이
            height (float): 세로 길이
        
        Returns:
            float: 넓이 (가로 × 세로)
        
        Examples:
            >>> calculate_area(5, 10)
            50
        """
        return width * height
    
    result = calculate_area(5, 10)
    print(f"넓이: {result}")
    
    # docstring 확인
    print(f"\n함수 설명:\n{calculate_area.__doc__}")
    
    print("\n" + "-" * 60)
    print("연습 문제 6 - 주석")
    print("-" * 60)
    
    print("""
연습 문제:
1. 다음 코드에 적절한 주석을 추가하세요:
   temperature = 25
   if temperature > 30:
       print("더워요")
   else:
       print("시원해요")

2. 원의 넓이를 계산하는 함수를 만들고 docstring을 작성하세요.
   함수명: calculate_circle_area
   매개변수: radius (반지름)
   반환값: 넓이 (π × r²)

3. 할 일 목록(TODO)을 3개 주석으로 작성하세요.

4. 아래 코드에서 불필요한 주석을 제거하고 필요한 주석만 남기세요:
   # 변수 x 선언
   x = 10
   # 변수 y 선언  
   y = 20
   # x와 y를 더함
   result = x + y
   # 결과 출력
   print(result)

아래에 코드를 작성하세요:
""")
    
    print("\n")


def section6_practice_solution():
    """섹션 6 연습 문제 해설"""
    print("=" * 60)
    print("섹션 6 연습 문제 해설")
    print("=" * 60)
    
    # 1. 적절한 주석 추가
    print("1. 주석 추가:")
    # 현재 온도 설정
    temperature = 25
    
    # 30도를 기준으로 날씨 판단
    if temperature > 30:
        print("   더워요")
    else:
        print("   시원해요")
    
    # 2. docstring이 있는 함수
    print("\n2. Docstring 예제:")
    import math
    
    def calculate_circle_area(radius):
        """
        원의 넓이를 계산하는 함수
        
        Parameters:
            radius (float): 원의 반지름
        
        Returns:
            float: 원의 넓이 (π × r²)
        
        Examples:
            >>> calculate_circle_area(5)
            78.53981633974483
        """
        return math.pi * radius ** 2
    
    area = calculate_circle_area(5)
    print(f"   반지름 5인 원의 넓이: {area:.2f}")
    
    # 3. TODO 목록
    print("\n3. TODO 목록:")
    print("   # TODO: 사용자 입력 검증 기능 추가")
    print("   # TODO: 데이터베이스 연동 구현")
    print("   # TODO: 에러 처리 로직 개선")
    
    # 4. 불필요한 주석 제거
    print("\n4. 개선된 코드:")
    print("""
   # 개선 전:
   # 변수 x 선언
   x = 10
   # 변수 y 선언
   y = 20
   # x와 y를 더함
   result = x + y
   # 결과 출력
   print(result)
   
   # 개선 후:
   x = 10
   y = 20
   result = x + y
   print(result)
   
   # 또는 의미 있는 주석만:
   # 두 숫자의 합 계산
   x = 10
   y = 20
   result = x + y
   print(result)
   """)
    
    print("\n")


def comprehensive_practice():
    """
    종합 실습 문제
    예상 시간: 30분
    """
    print("=" * 60)
    print("종합 실습 문제")
    print("=" * 60)
    
    print("""
지금까지 배운 내용을 종합하여 다음 프로그램을 작성하세요:

【문제 1】학생 정보 관리 프로그램
다음 정보를 변수에 저장하고 출력하세요:
- 학생 이름: "김철수"
- 학번: 20230001  
- 학년: 2
- 학점: 4.3
- 전공: "컴퓨터공학"

출력 형식:
=== 학생 정보 ===
이름: 김철수
학번: 20230001
학년: 2학년
학점: 4.3/4.5
전공: 컴퓨터공학
==================

【문제 2】계산기 프로그램
두 숫자를 변수에 저장하고 사칙연산 결과를 모두 출력하세요.
- num1 = 45
- num2 = 7

출력 형식:
=== 계산 결과 ===
45 + 7 = 52
45 - 7 = 38
45 * 7 = 315
45 / 7 = 6.43
45 // 7 = 6 (몫)
45 % 7 = 3 (나머지)
==================

【문제 3】문자열 처리 프로그램
email = "student.name@university.ac.kr"에서
1) @를 기준으로 사용자명과 도메인 분리
2) 사용자명을 대문자로 변환
3) 도메인에서 '.'을 기준으로 분리
4) 전체 결과를 보기 좋게 출력

【문제 4】온도 변환 프로그램
섭씨 온도를 입력받아 화씨와 켈빈으로 변환하세요.
- 화씨 = 섭씨 × 9/5 + 32
- 켈빈 = 섭씨 + 273.15

입력: celsius = 25
출력:
섭씨: 25°C
화씨: 77.0°F
켈빈: 298.15K

【문제 5】문자열 분석 프로그램
sentence = "Python is a powerful programming language"
1) 전체 길이 출력
2) 'a' 문자가 몇 개 있는지 출력
3) 모든 단어를 대문자로 변환
4) 단어를 리스트로 분리하여 출력
5) 단어 개수 출력

아래에 코드를 작성하세요:
""")
    
    print("\n")


def comprehensive_practice_solution():
    """종합 실습 문제 해설"""
    print("=" * 60)
    print("종합 실습 문제 해설")
    print("=" * 60)
    
    # 문제 1: 학생 정보 관리
    print("\n【문제 1 해설】학생 정보 관리 프로그램")
    student_name = "김철수"
    student_id = 20230001
    grade = 2
    gpa = 4.3
    major = "컴퓨터공학"
    
    print("=== 학생 정보 ===")
    print(f"이름: {student_name}")
    print(f"학번: {student_id}")
    print(f"학년: {grade}학년")
    print(f"학점: {gpa}/4.5")
    print(f"전공: {major}")
    print("==================")
    
    # 문제 2: 계산기
    print("\n【문제 2 해설】계산기 프로그램")
    num1 = 45
    num2 = 7
    
    print("=== 계산 결과 ===")
    print(f"{num1} + {num2} = {num1 + num2}")
    print(f"{num1} - {num2} = {num1 - num2}")
    print(f"{num1} * {num2} = {num1 * num2}")
    print(f"{num1} / {num2} = {num1 / num2:.2f}")
    print(f"{num1} // {num2} = {num1 // num2} (몫)")
    print(f"{num1} % {num2} = {num1 % num2} (나머지)")
    print("==================")
    
    # 문제 3: 문자열 처리
    print("\n【문제 3 해설】문자열 처리 프로그램")
    email = "student.name@university.ac.kr"
    
    # 1) 사용자명과 도메인 분리
    username, domain = email.split("@")
    
    # 2) 사용자명 대문자 변환
    username_upper = username.upper()
    
    # 3) 도메인 분리
    domain_parts = domain.split(".")
    
    # 4) 결과 출력
    print(f"원본 이메일: {email}")
    print(f"사용자명: {username} → 대문자: {username_upper}")
    print(f"도메인: {domain}")
    print(f"도메인 분리: {domain_parts}")
    
    # 문제 4: 온도 변환
    print("\n【문제 4 해설】온도 변환 프로그램")
    celsius = 25
    fahrenheit = celsius * 9/5 + 32
    kelvin = celsius + 273.15
    
    print(f"섭씨: {celsius}°C")
    print(f"화씨: {fahrenheit}°F")
    print(f"켈빈: {kelvin}K")
    
    # 문제 5: 문자열 분석
    print("\n【문제 5 해설】문자열 분석 프로그램")
    sentence = "Python is a powerful programming language"
    
    # 1) 전체 길이
    length = len(sentence)
    print(f"1. 전체 길이: {length}")
    
    # 2) 'a' 개수
    a_count = sentence.count('a')
    print(f"2. 'a' 개수: {a_count}")
    
    # 3) 대문자 변환
    upper_sentence = sentence.upper()
    print(f"3. 대문자: {upper_sentence}")
    
    # 4) 단어 분리
    words = sentence.split()
    print(f"4. 단어 리스트: {words}")
    
    # 5) 단어 개수
    word_count = len(words)
    print(f"5. 단어 개수: {word_count}")
    
    print("\n")


def run_all_examples():
    """모든 예제 실행"""
    print("\n" + "=" * 60)
    print("Python 기초 - 전체 예제 및 실습")
    print("=" * 60)
    print("\n각 섹션을 순서대로 학습합니다.\n")
    
    # 섹션 1
    section1_variables()
    input("계속하려면 Enter를 누르세요...")
    section1_practice_solution()
    input("계속하려면 Enter를 누르세요...")
    
    # 섹션 2
    section2_numbers()
    input("계속하려면 Enter를 누르세요...")
    section2_practice_solution()
    input("계속하려면 Enter를 누르세요...")
    
    # 섹션 3
    section3_strings()
    input("계속하려면 Enter를 누르세요...")
    section3_practice_solution()
    input("계속하려면 Enter를 누르세요...")
    
    # 섹션 4
    section4_type_conversion()
    input("계속하려면 Enter를 누르세요...")
    section4_practice_solution()
    input("계속하려면 Enter를 누르세요...")
    
    # 섹션 5
    section5_builtin_functions()
    input("계속하려면 Enter를 누르세요...")
    section5_practice_solution()
    input("계속하려면 Enter를 누르세요...")
    
    # 섹션 6
    section6_comments()
    input("계속하려면 Enter를 누르세요...")
    section6_practice_solution()
    input("계속하려면 Enter를 누르세요...")
    
    # 종합 실습
    comprehensive_practice()
    input("계속하려면 Enter를 누르세요...")
    comprehensive_practice_solution()
    
    print("\n" + "=" * 60)
    print("모든 학습을 완료했습니다! 수고하셨습니다!")
    print("=" * 60)


def show_menu():
    """메뉴 표시"""
    print("\n" + "=" * 60)
    print("Python 기초 - 변수와 자료형 실습")
    print("=" * 60)
    print("\n학습할 섹션을 선택하세요:")
    print("\n[이론 + 예제]")
    print("1. 섹션 1 - 변수 선언 기초")
    print("2. 섹션 2 - 숫자형 자료형")
    print("3. 섹션 3 - 문자열 자료형")
    print("4. 섹션 4 - 자료형 변환")
    print("5. 섹션 5 - 내장 함수")
    print("6. 섹션 6 - 주석")
    print("\n[연습 문제 해설]")
    print("11. 섹션 1 해설")
    print("12. 섹션 2 해설")
    print("13. 섹션 3 해설")
    print("14. 섹션 4 해설")
    print("15. 섹션 5 해설")
    print("16. 섹션 6 해설")
    print("\n[종합 실습]")
    print("20. 종합 실습 문제")
    print("21. 종합 실습 해설")
    print("\n[전체 실행]")
    print("0. 모든 섹션 순서대로 실행")
    print("q. 종료")
    print("=" * 60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Python 기초 - 변수 선언 규칙과 기본 자료형 실습         ║
║                                                              ║
║     예상 학습 시간: 3시간 30분                               ║
║     난이도: 초급                                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        show_menu()
        choice = input("\n선택: ").strip()
        
        if choice == '0':
            run_all_examples()
        elif choice == '1':
            section1_variables()
        elif choice == '2':
            section2_numbers()
        elif choice == '3':
            section3_strings()
        elif choice == '4':
            section4_type_conversion()
        elif choice == '5':
            section5_builtin_functions()
        elif choice == '6':
            section6_comments()
        elif choice == '11':
            section1_practice_solution()
        elif choice == '12':
            section2_practice_solution()
        elif choice == '13':
            section3_practice_solution()
        elif choice == '14':
            section4_practice_solution()
        elif choice == '15':
            section5_practice_solution()
        elif choice == '16':
            section6_practice_solution()
        elif choice == '20':
            comprehensive_practice()
        elif choice == '21':
            comprehensive_practice_solution()
        elif choice.lower() == 'q':
            print("\n학습을 종료합니다. 수고하셨습니다!")
            break
        else:
            print("\n잘못된 선택입니다. 다시 선택해주세요.")
        
        input("\n메뉴로 돌아가려면 Enter를 누르세요...")
