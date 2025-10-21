"""
Python 기초 - 종합 퀴즈

이 파일은 Python 기초 학습 내용을 종합적으로 평가하는 퀴즈입니다.
총 10문제로 구성되어 있으며, 각 문제는 배운 내용을 확인합니다.

실행 방법:
    python quiz.py
"""


def display_header():
    """퀴즈 헤더 출력"""
    print("=" * 70)
    print("     Python 기초 - 종합 평가 퀴즈 (총 10문제)     ")
    print("=" * 70)
    print("\n각 문제를 읽고 정답을 작성하세요.")
    print("답안 제출 후 자동으로 채점됩니다.\n")
    print("-" * 70)


def quiz_question_1():
    """문제 1: 변수 네이밍"""
    print("\n【문제 1】 변수 네이밍 (객관식)")
    print("-" * 70)
    print("다음 중 올바른 Python 변수명이 아닌 것은?")
    print("① user_name")
    print("② _private_var")
    print("③ 2nd_place")
    print("④ totalPrice")
    
    answer = input("\n정답 (①, ②, ③, ④): ").strip()
    
    if answer in ["③", "3"]:
        print("✓ 정답입니다!")
        print("해설: 변수명은 숫자로 시작할 수 없습니다.")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: ③")
        print("해설: 변수명은 숫자로 시작할 수 없습니다. 2nd_place는 잘못된 변수명입니다.")
        return False


def quiz_question_2():
    """문제 2: 산술 연산"""
    print("\n【문제 2】 산술 연산 (단답형)")
    print("-" * 70)
    print("다음 코드의 결과를 작성하세요:")
    print("```python")
    print("result = 17 // 5")
    print("print(result)")
    print("```")
    
    answer = input("\n결과값: ").strip()
    
    if answer == "3":
        print("✓ 정답입니다!")
        print("해설: // 연산자는 몫을 반환합니다. 17 ÷ 5 = 3 (나머지 2)")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: 3")
        print("해설: // 연산자는 정수 나눗셈의 몫을 반환합니다.")
        return False


def quiz_question_3():
    """문제 3: 문자열 인덱싱"""
    print("\n【문제 3】 문자열 인덱싱 (단답형)")
    print("-" * 70)
    print("다음 코드의 결과를 작성하세요:")
    print("```python")
    print('text = "Programming"')
    print("print(text[4])")
    print("```")
    
    answer = input("\n결과값: ").strip()
    
    if answer == "r":
        print("✓ 정답입니다!")
        print("해설: 인덱스는 0부터 시작하므로 text[4]는 'r'입니다.")
        print("      P(0) r(1) o(2) g(3) r(4) a(5) m(6) m(7) i(8) n(9) g(10)")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: r")
        print("해설: 인덱스 4는 5번째 문자인 'r'을 가리킵니다.")
        return False


def quiz_question_4():
    """문제 4: 문자열 슬라이싱"""
    print("\n【문제 4】 문자열 슬라이싱 (단답형)")
    print("-" * 70)
    print("다음 코드의 결과를 작성하세요:")
    print("```python")
    print('word = "Python"')
    print("print(word[1:4])")
    print("```")
    
    answer = input("\n결과값: ").strip()
    
    if answer == "yth":
        print("✓ 정답입니다!")
        print("해설: 슬라이싱 [1:4]는 인덱스 1, 2, 3의 문자를 가져옵니다.")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: yth")
        print("해설: word[1:4]는 인덱스 1부터 3까지(4 미포함)의 문자를 반환합니다.")
        return False


def quiz_question_5():
    """문제 5: 자료형 변환"""
    print("\n【문제 5】 자료형 변환 (객관식)")
    print("-" * 70)
    print("다음 코드의 결과는?")
    print("```python")
    print('result = int(float("3.8"))')
    print("```")
    print("① 3")
    print("② 4")
    print("③ 3.8")
    print("④ 오류 발생")
    
    answer = input("\n정답 (①, ②, ③, ④): ").strip()
    
    if answer in ["①", "1"]:
        print("✓ 정답입니다!")
        print("해설: float('3.8')은 3.8이 되고, int(3.8)은 소수점을 버려서 3이 됩니다.")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: ①")
        print("해설: int()는 소수점을 버립니다(반올림 X). 3.8 → 3")
        return False


def quiz_question_6():
    """문제 6: f-string"""
    print("\n【문제 6】 f-string 포맷팅 (단답형)")
    print("-" * 70)
    print("다음 코드의 출력 결과를 작성하세요:")
    print("```python")
    print('name = "김철수"')
    print("age = 25")
    print('print(f"{name}님은 {age}세입니다.")')
    print("```")
    
    answer = input("\n출력값: ").strip()
    
    if answer == "김철수님은 25세입니다.":
        print("✓ 정답입니다!")
        print("해설: f-string은 중괄호 안의 변수 값을 문자열에 삽입합니다.")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: 김철수님은 25세입니다.")
        print("해설: f-string의 {}안에 있는 변수가 해당 값으로 치환됩니다.")
        return False


def quiz_question_7():
    """문제 7: 내장 함수"""
    print("\n【문제 7】 내장 함수 (객관식)")
    print("-" * 70)
    print("다음 중 문자열의 길이를 반환하는 함수는?")
    print("① size()")
    print("② length()")
    print("③ len()")
    print("④ count()")
    
    answer = input("\n정답 (①, ②, ③, ④): ").strip()
    
    if answer in ["③", "3"]:
        print("✓ 정답입니다!")
        print("해설: len() 함수는 문자열, 리스트 등의 길이를 반환합니다.")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: ③")
        print("해설: Python에서는 len() 함수를 사용하여 길이를 구합니다.")
        return False


def quiz_question_8():
    """문제 8: 복합 연산"""
    print("\n【문제 8】 복합 문제 (코드 작성)")
    print("-" * 70)
    print("다음 조건을 만족하는 코드를 작성하세요:")
    print("1. 변수 price에 1500 저장")
    print("2. 10% 할인 적용 (price의 90%)")
    print("3. 결과를 정수로 변환하여 출력")
    print("\n(실제로 코드를 실행하지 않고, 결과값만 입력하세요)")
    
    answer = input("\n최종 결과값: ").strip()
    
    if answer == "1350":
        print("✓ 정답입니다!")
        print("해설: 1500 × 0.9 = 1350.0 → int(1350.0) = 1350")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: 1350")
        print("해설:")
        print("  price = 1500")
        print("  discounted = price * 0.9  # 1350.0")
        print("  result = int(discounted)   # 1350")
        return False


def quiz_question_9():
    """문제 9: 문자열 메서드"""
    print("\n【문제 9】 문자열 메서드 (객관식)")
    print("-" * 70)
    print("다음 코드의 결과는?")
    print("```python")
    print('text = "hello world"')
    print("result = text.upper()")
    print("```")
    print("① hello world")
    print("② HELLO WORLD")
    print("③ Hello World")
    print("④ Hello world")
    
    answer = input("\n정답 (①, ②, ③, ④): ").strip()
    
    if answer in ["②", "2"]:
        print("✓ 정답입니다!")
        print("해설: upper() 메서드는 모든 문자를 대문자로 변환합니다.")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: ②")
        print("해설: upper()는 문자열의 모든 알파벳을 대문자로 변환합니다.")
        return False


def quiz_question_10():
    """문제 10: 종합 이해"""
    print("\n【문제 10】 종합 문제 (객관식)")
    print("-" * 70)
    print("다음 코드에서 오류가 발생하는 줄은?")
    print("```python")
    print("1: age = '25'")
    print("2: next_year = age + 1")
    print("3: print(next_year)")
    print("```")
    print("① 1번 줄")
    print("② 2번 줄")
    print("③ 3번 줄")
    print("④ 오류 없음")
    
    answer = input("\n정답 (①, ②, ③, ④): ").strip()
    
    if answer in ["②", "2"]:
        print("✓ 정답입니다!")
        print("해설: age는 문자열이므로 정수 1과 더할 수 없습니다.")
        print("     올바른 코드: next_year = int(age) + 1")
        return True
    else:
        print("✗ 오답입니다.")
        print("정답: ②")
        print("해설: 문자열과 정수는 직접 더할 수 없습니다. TypeError가 발생합니다.")
        return False


def run_quiz():
    """전체 퀴즈 실행"""
    display_header()
    
    questions = [
        quiz_question_1,
        quiz_question_2,
        quiz_question_3,
        quiz_question_4,
        quiz_question_5,
        quiz_question_6,
        quiz_question_7,
        quiz_question_8,
        quiz_question_9,
        quiz_question_10
    ]
    
    score = 0
    total = len(questions)
    
    for i, question_func in enumerate(questions, 1):
        if question_func():
            score += 1
        
        if i < total:
            input("\n다음 문제로 넘어가려면 Enter를 누르세요...")
    
    # 최종 결과
    print("\n" + "=" * 70)
    print("               퀴즈 결과               ")
    print("=" * 70)
    print(f"\n총 {total}문제 중 {score}문제 정답")
    print(f"정답률: {score/total*100:.1f}%")
    
    # 등급 평가
    percentage = score / total * 100
    if percentage >= 90:
        grade = "우수"
        message = "완벽합니다! Python 기초를 확실히 이해하셨네요. 🎉"
    elif percentage >= 70:
        grade = "양호"
        message = "잘하셨습니다! 몇 가지 개념을 복습하면 완벽할 것 같습니다. 👍"
    elif percentage >= 50:
        grade = "보통"
        message = "기본은 이해하셨어요. 오답 부분을 다시 학습해보세요. 📚"
    else:
        grade = "부족"
        message = "theory.md를 다시 읽고 practice.py로 연습해보세요. 💪"
    
    print(f"\n등급: {grade}")
    print(f"평가: {message}")
    
    # 복습 권장사항
    if score < total:
        print("\n" + "-" * 70)
        print("📌 복습 권장 섹션:")
        wrong_topics = []
        
        if not questions[0]() and not questions[5]():  # 문제 1, 6 관련
            wrong_topics.append("• 섹션 1: 변수 선언과 네이밍 규칙")
        if not questions[1]():  # 문제 2 관련
            wrong_topics.append("• 섹션 2: 숫자형 자료형과 산술 연산")
        if not questions[2]() or not questions[3]() or not questions[8]():  # 문제 3, 4, 9 관련
            wrong_topics.append("• 섹션 3: 문자열 자료형")
        if not questions[4]() or not questions[9]():  # 문제 5, 10 관련
            wrong_topics.append("• 섹션 4: 자료형 변환")
        if not questions[6]():  # 문제 7 관련
            wrong_topics.append("• 섹션 5: 내장 함수")
        
        if wrong_topics:
            for topic in wrong_topics:
                print(topic)
        else:
            print("• 전체적으로 복습하시면 좋겠습니다.")
    
    print("\n" + "=" * 70)
    print("수고하셨습니다! 🎓")
    print("=" * 70)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         Python 기초 - 종합 평가 퀴즈                         ║
║                                                              ║
║         총 10문제 | 예상 소요 시간: 15-20분                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    choice = input("퀴즈를 시작하시겠습니까? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n퀴즈를 시작합니다!\n")
        run_quiz()
    else:
        print("\n다음에 다시 도전해보세요!")
