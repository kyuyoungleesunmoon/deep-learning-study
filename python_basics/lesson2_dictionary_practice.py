"""
Python 기초 - 2교시: 딕셔너리(Dictionary) 실습

이 파일은 lesson2_dictionary_theory.md와 함께 사용하는 실습 코드입니다.
딕셔너리의 기본 사용법과 실무 예제를 포함합니다.

실행 방법:
    python lesson2_dictionary_practice.py
"""


def intro():
    """2교시 소개"""
    print("=" * 70)
    print("  🎓 Python 기초 - 2교시: 딕셔너리(Dictionary)")
    print("=" * 70)
    print("\n안녕하세요! 2교시에 오신 것을 환영합니다.")
    print("오늘은 Key-Value 구조로 데이터를 관리하는 방법을 배웁니다.\n")
    print("📚 학습 내용:")
    print("  1. 딕셔너리(Dictionary) - Key로 값을 찾는 자료구조")
    print("  2. 리스트 + 딕셔너리 조합")
    print("  3. 실무 예제 - 제품 정보 관리 시스템")
    print("\n" + "=" * 70 + "\n")


def section1_what_is_dictionary():
    """섹션 1: 딕셔너리란 무엇인가?"""
    print("\n" + "=" * 70)
    print("섹션 1: 딕셔너리(Dictionary)란 무엇인가?")
    print("=" * 70)
    
    print("\n💡 딕셔너리는 '단어와 뜻을 연결한 사전'입니다!")
    print("   일상생활 비유:")
    print("   - 영한 사전: 'apple' → '사과'")
    print("   - 전화번호부: '홍길동' → '010-1234-5678'")
    print("   - 상품 바코드: '8801234567890' → '삼다수 2L'\n")
    
    # 예제 1: 딕셔너리 생성
    print("[예제 1] 딕셔너리 생성하기")
    print("-" * 70)
    phone_book = {
        "홍길동": "010-1234-5678",
        "김철수": "010-9876-5432",
        "이영희": "010-5555-6666"
    }
    print(f"전화번호부: {phone_book}")
    print()
    
    # 예제 2: 리스트 vs 딕셔너리
    print("[예제 2] 왜 딕셔너리를 사용할까?")
    print("-" * 70)
    
    print("❌ 리스트만 사용 (비효율적):")
    student_list = ["홍길동", 25, "컴퓨터공학"]
    print(f"   학생 정보: {student_list}")
    print(f"   이름: {student_list[0]} (0번 인덱스라는 걸 기억해야 함)")
    print(f"   나이: {student_list[1]} (1번 인덱스라는 걸 기억해야 함)")
    
    print("\n✅ 딕셔너리 사용 (명확하고 효율적):")
    student_dict = {"name": "홍길동", "age": 25, "major": "컴퓨터공학"}
    print(f"   학생 정보: {student_dict}")
    print(f"   이름: {student_dict['name']} (의미가 명확!)")
    print(f"   나이: {student_dict['age']} (의미가 명확!)")
    print()
    
    # 예제 3: 다양한 딕셔너리
    print("[예제 3] 다양한 종류의 딕셔너리")
    print("-" * 70)
    
    person = {"name": "홍길동", "age": 25, "city": "서울"}
    product = {"name": "노트북", "price": 1200000, "stock": 5}
    empty_dict = {}
    
    print(f"사람 정보: {person}")
    print(f"제품 정보: {product}")
    print(f"빈 딕셔너리: {empty_dict}")
    print()


def section2_basic_operations():
    """섹션 2: 딕셔너리 기본 조작"""
    print("\n" + "=" * 70)
    print("섹션 2: 딕셔너리 기본 조작 (조회/추가/수정/삭제)")
    print("=" * 70)
    
    # 예제 1: 값 조회
    print("\n[예제 1] 값 조회하기")
    print("-" * 70)
    product = {
        "name": "노트북",
        "price": 1200000,
        "stock": 5
    }
    print(f"제품 정보: {product}")
    print()
    
    print("방법 1: 대괄호 사용")
    print(f"  제품명: {product['name']}")
    print(f"  가격: {product['price']:,}원")
    
    print("\n방법 2: get() 메서드 (안전)")
    print(f"  제품명: {product.get('name')}")
    print(f"  색상: {product.get('color', '정보 없음')} ← Key가 없어도 에러 안 남!")
    print()
    
    # 예제 2: 값 추가
    print("[예제 2] 값 추가하기")
    print("-" * 70)
    product = {"name": "노트북", "price": 1200000}
    print(f"초기 제품: {product}")
    
    product["stock"] = 5
    print(f"재고 추가 후: {product}")
    
    product["brand"] = "Apple"
    print(f"브랜드 추가 후: {product}")
    print()
    
    # 예제 3: 값 수정
    print("[예제 3] 값 수정하기")
    print("-" * 70)
    product = {"name": "노트북", "price": 1200000, "stock": 5}
    print(f"원래 제품: {product}")
    
    product["price"] = 1100000
    print(f"가격 인하 후: {product}")
    
    product["stock"] = 3
    print(f"재고 감소 후: {product}")
    print()
    
    # 예제 4: 값 삭제
    print("[예제 4] 값 삭제하기")
    print("-" * 70)
    product = {"name": "노트북", "price": 1200000, "stock": 5, "discontinued": True}
    print(f"초기 제품: {product}")
    
    del product["discontinued"]
    print(f"단종 정보 삭제 후: {product}")
    
    removed_stock = product.pop("stock")
    print(f"재고 삭제 후: {product} (삭제된 재고: {removed_stock})")
    print()


def section3_methods():
    """섹션 3: 딕셔너리 메서드"""
    print("\n" + "=" * 70)
    print("섹션 3: 딕셔너리 메서드 (keys, values, items)")
    print("=" * 70)
    
    product = {
        "name": "노트북",
        "price": 1200000,
        "stock": 5
    }
    
    # 예제 1: keys()
    print("\n[예제 1] keys() - 모든 Key 가져오기")
    print("-" * 70)
    print(f"제품 정보: {product}")
    print(f"모든 Key: {list(product.keys())}")
    print()
    
    # 예제 2: values()
    print("[예제 2] values() - 모든 Value 가져오기")
    print("-" * 70)
    scores = {"수학": 85, "영어": 90, "과학": 88}
    print(f"과목별 점수: {scores}")
    print(f"모든 점수: {list(scores.values())}")
    print(f"총점: {sum(scores.values())}점")
    print(f"평균: {sum(scores.values()) / len(scores):.1f}점")
    print()
    
    # 예제 3: items()
    print("[예제 3] items() - Key-Value 쌍 가져오기")
    print("-" * 70)
    print(f"제품 정보: {product}")
    print("\nKey-Value 쌍으로 출력:")
    for key, value in product.items():
        print(f"  {key}: {value}")
    print()
    
    # 예제 4: update()
    print("[예제 4] update() - 여러 항목 추가/수정")
    print("-" * 70)
    product = {"name": "노트북", "price": 1200000}
    print(f"초기 제품: {product}")
    
    product.update({"stock": 5, "brand": "Apple", "price": 1100000})
    print(f"업데이트 후: {product}")
    print()


def section4_nested_structures():
    """섹션 4: 리스트 + 딕셔너리 조합"""
    print("\n" + "=" * 70)
    print("섹션 4: 리스트 + 딕셔너리 조합")
    print("=" * 70)
    
    # 예제 1: 딕셔너리 안에 리스트
    print("\n[예제 1] 딕셔너리 안에 리스트")
    print("-" * 70)
    student = {
        "name": "홍길동",
        "age": 20,
        "scores": [85, 90, 88, 92]
    }
    print(f"학생 정보: {student}")
    print(f"이름: {student['name']}")
    print(f"점수: {student['scores']}")
    print(f"평균: {sum(student['scores']) / len(student['scores']):.1f}점")
    print()
    
    # 예제 2: 리스트 안에 딕셔너리 (가장 많이 사용!)
    print("[예제 2] 리스트 안에 딕셔너리 (실무에서 가장 많이 사용!)")
    print("-" * 70)
    products = [
        {"name": "노트북", "price": 1200000, "stock": 5},
        {"name": "마우스", "price": 30000, "stock": 20},
        {"name": "키보드", "price": 80000, "stock": 15}
    ]
    
    print("제품 목록:")
    for i, product in enumerate(products, 1):
        print(f"  {i}. {product['name']}: {product['price']:,}원 (재고: {product['stock']}개)")
    
    total_value = sum(p['price'] * p['stock'] for p in products)
    print(f"\n총 재고 가치: {total_value:,}원")
    print()
    
    # 예제 3: 딕셔너리 안에 딕셔너리
    print("[예제 3] 딕셔너리 안에 딕셔너리")
    print("-" * 70)
    students = {
        "홍길동": {"수학": 85, "영어": 90, "과학": 88},
        "김철수": {"수학": 92, "영어": 88, "과학": 95},
        "이영희": {"수학": 78, "영어": 85, "과학": 80}
    }
    
    print("학생별 과목 점수:")
    for name, scores in students.items():
        average = sum(scores.values()) / len(scores)
        print(f"  {name}: {scores} → 평균 {average:.1f}점")
    print()


def section5_comparison():
    """섹션 5: 자료구조 비교"""
    print("\n" + "=" * 70)
    print("섹션 5: 리스트 vs 튜플 vs 딕셔너리 비교")
    print("=" * 70)
    
    print("""
┌─────────────┬─────────────┬──────────────┬───────────────────┐
│   구분      │ 리스트      │ 튜플         │ 딕셔너리          │
├─────────────┼─────────────┼──────────────┼───────────────────┤
│ 표기법      │ [1, 2, 3]   │ (1, 2, 3)    │ {"a": 1, "b": 2}  │
│ 순서        │ ✅ 있음     │ ✅ 있음      │ ❌ 없음           │
│ 변경 가능   │ ✅ 가능     │ ❌ 불가능    │ ✅ 가능           │
│ 중복 허용   │ ✅ 허용     │ ✅ 허용      │ ❌ Key는 고유     │
│ 접근 방법   │ list[0]     │ tuple[0]     │ dict["key"]       │
│ 용도        │ 순서있는    │ 고정         │ Key-Value 매핑    │
│             │ 데이터      │ 데이터       │                   │
└─────────────┴─────────────┴──────────────┴───────────────────┘
    """)
    
    print("\n[실제 사용 예시 비교]")
    print("-" * 70)
    
    print("\n📝 리스트: 순서가 중요한 데이터")
    shopping_list = ["사과", "우유", "빵"]
    print(f"   장보기 목록: {shopping_list}")
    
    print("\n🔒 튜플: 변경하면 안 되는 데이터")
    rgb_red = (255, 0, 0)
    print(f"   빨간색 RGB: {rgb_red}")
    
    print("\n🔍 딕셔너리: 이름으로 찾는 데이터")
    person = {"name": "홍길동", "age": 25, "city": "서울"}
    print(f"   사람 정보: {person}")
    print()


def section6_product_system():
    """섹션 6: 실무 예제 - 제품 관리 시스템"""
    print("\n" + "=" * 70)
    print("섹션 6: 실무 예제 - 제품 관리 시스템")
    print("=" * 70)
    
    print("\n📦 상황: 쇼핑몰 제품 관리 시스템을 만들어봅시다!\n")
    
    # 제품 데이터베이스
    products = [
        {"product_name": "노트북", "price": 1200000, "stock": 5},
        {"product_name": "마우스", "price": 30000, "stock": 20},
        {"product_name": "키보드", "price": 80000, "stock": 15},
        {"product_name": "모니터", "price": 300000, "stock": 8}
    ]
    
    # 예제 1: 전체 제품 목록 출력
    print("[예제 1] 전체 제품 목록")
    print("=" * 70)
    print(f"{'상품명':<12} {'가격':>12} {'재고':>8}")
    print("-" * 70)
    
    for product in products:
        print(f"{product['product_name']:<12} {product['price']:>10,}원 {product['stock']:>6}개")
    
    total_value = sum(p['price'] * p['stock'] for p in products)
    print("-" * 70)
    print(f"총 재고 가치: {total_value:,}원")
    
    # 예제 2: 제품 검색
    print("\n[예제 2] 제품 검색 기능")
    print("=" * 70)
    
    search_name = "노트북"
    found = None
    for product in products:
        if product["product_name"] == search_name:
            found = product
            break
    
    if found:
        print(f"✅ '{search_name}' 제품을 찾았습니다!")
        print(f"   상품명: {found['product_name']}")
        print(f"   가격: {found['price']:,}원")
        print(f"   재고: {found['stock']}개")
    else:
        print(f"❌ '{search_name}' 제품을 찾을 수 없습니다.")
    
    # 예제 3: 제품 판매
    print("\n[예제 3] 제품 판매 시뮬레이션")
    print("=" * 70)
    
    sell_name = "마우스"
    sell_quantity = 3
    
    print(f"판매 요청: {sell_name} {sell_quantity}개")
    
    for product in products:
        if product["product_name"] == sell_name:
            if product["stock"] >= sell_quantity:
                product["stock"] -= sell_quantity
                total_price = product["price"] * sell_quantity
                print(f"✅ 판매 완료!")
                print(f"   총액: {total_price:,}원")
                print(f"   남은 재고: {product['stock']}개")
            else:
                print(f"❌ 재고 부족! (현재 재고: {product['stock']}개)")
            break
    
    # 예제 4: 재고 부족 제품 찾기
    print("\n[예제 4] 재고 부족 제품 (10개 미만)")
    print("=" * 70)
    
    low_stock_products = []
    for product in products:
        if product["stock"] < 10:
            low_stock_products.append(product)
    
    if low_stock_products:
        print("⚠️  재고 부족 제품:")
        for product in low_stock_products:
            print(f"   - {product['product_name']}: {product['stock']}개")
    else:
        print("✅ 모든 제품의 재고가 충분합니다!")
    print()


def practice_problems():
    """연습 문제"""
    print("\n" + "=" * 70)
    print("📝 연습 문제")
    print("=" * 70)
    
    print("""
다음 문제들을 직접 풀어보세요!

【문제 1】 학생 정보 딕셔너리 만들기
이름, 나이, 전공을 포함하는 학생 딕셔너리를 만들고,
나이를 1 증가시키고, 학년 정보를 추가하세요.

【문제 2】 제품 조회
다음 제품 리스트에서 "키보드"를 찾아 가격과 재고를 출력하세요.
products = [
    {"name": "노트북", "price": 1200000, "stock": 5},
    {"name": "마우스", "price": 30000, "stock": 20},
    {"name": "키보드", "price": 80000, "stock": 15}
]

【문제 3】 과목별 점수 관리
scores = {"수학": 85, "영어": 90, "과학": 88}
1) 모든 과목과 점수를 출력하세요
2) 평균 점수를 구하세요
3) "국어": 92 를 추가하세요

【문제 4】 전화번호부
빈 딕셔너리를 만들고, 3명의 친구 이름과 전화번호를 추가한 후,
특정 친구의 전화번호를 검색하고 출력하세요.

【문제 5】 제품 판매 총액
products = [
    {"name": "사과", "price": 2000, "quantity": 10},
    {"name": "바나나", "price": 1500, "quantity": 15},
    {"name": "오렌지", "price": 3000, "quantity": 8}
]
각 제품의 총 판매액(price × quantity)을 계산하고,
모든 제품의 총 판매액을 구하세요.
    """)


def practice_solutions():
    """연습 문제 해설"""
    print("\n" + "=" * 70)
    print("✅ 연습 문제 해설")
    print("=" * 70)
    
    print("\n【문제 1 해설】 학생 정보 딕셔너리 만들기")
    print("-" * 70)
    student = {"name": "홍길동", "age": 20, "major": "컴퓨터공학"}
    print(f"1. 초기 학생: {student}")
    
    student["age"] += 1
    print(f"2. 나이 증가 후: {student}")
    
    student["grade"] = 2
    print(f"3. 학년 추가 후: {student}")
    
    print("\n【문제 2 해설】 제품 조회")
    print("-" * 70)
    products = [
        {"name": "노트북", "price": 1200000, "stock": 5},
        {"name": "마우스", "price": 30000, "stock": 20},
        {"name": "키보드", "price": 80000, "stock": 15}
    ]
    
    for product in products:
        if product["name"] == "키보드":
            print(f"제품명: {product['name']}")
            print(f"가격: {product['price']:,}원")
            print(f"재고: {product['stock']}개")
            break
    
    print("\n【문제 3 해설】 과목별 점수 관리")
    print("-" * 70)
    scores = {"수학": 85, "영어": 90, "과학": 88}
    
    print("1. 모든 과목과 점수:")
    for subject, score in scores.items():
        print(f"   {subject}: {score}점")
    
    average = sum(scores.values()) / len(scores)
    print(f"\n2. 평균 점수: {average:.1f}점")
    
    scores["국어"] = 92
    print(f"\n3. 국어 추가 후: {scores}")
    
    print("\n【문제 4 해설】 전화번호부")
    print("-" * 70)
    phone_book = {}
    phone_book["홍길동"] = "010-1234-5678"
    phone_book["김철수"] = "010-9876-5432"
    phone_book["이영희"] = "010-5555-6666"
    
    print(f"전화번호부: {phone_book}")
    
    search_name = "김철수"
    if search_name in phone_book:
        print(f"{search_name}의 전화번호: {phone_book[search_name]}")
    
    print("\n【문제 5 해설】 제품 판매 총액")
    print("-" * 70)
    products = [
        {"name": "사과", "price": 2000, "quantity": 10},
        {"name": "바나나", "price": 1500, "quantity": 15},
        {"name": "오렌지", "price": 3000, "quantity": 8}
    ]
    
    total_sales = 0
    for product in products:
        sales = product["price"] * product["quantity"]
        print(f"{product['name']}: {sales:,}원")
        total_sales += sales
    
    print(f"\n총 판매액: {total_sales:,}원")
    print()


def quiz():
    """5문항 퀴즈"""
    print("\n" + "=" * 70)
    print("📝 2교시 퀴즈 (5문항)")
    print("=" * 70)
    print("\n딕셔너리에 대해 배운 내용을 확인해봅시다!\n")
    
    score = 0
    total = 5
    
    # 문제 1
    print("【문제 1】")
    print("딕셔너리를 만드는 올바른 방법은? (객관식)")
    print("① [1, 2, 3]")
    print("② (1, 2, 3)")
    print("③ {'a': 1, 'b': 2}")
    print("④ <'a': 1, 'b': 2>")
    answer = input("\n정답: ").strip()
    if answer in ["③", "3"]:
        print("✅ 정답입니다!")
        score += 1
    else:
        print("❌ 오답입니다. 정답은 ③입니다.")
        print("   딕셔너리는 중괄호 {}를 사용하고 Key:Value 형태입니다.")
    
    # 문제 2
    print("\n" + "-" * 70)
    print("【문제 2】")
    print("다음 코드의 결과는? (단답형)")
    print("person = {'name': '홍길동', 'age': 25}")
    print("print(person['name'])")
    answer = input("\n정답: ").strip()
    if answer == "홍길동":
        print("✅ 정답입니다!")
        score += 1
    else:
        print("❌ 오답입니다. 정답은 '홍길동'입니다.")
        print("   딕셔너리에서 Key로 Value를 조회할 수 있습니다.")
    
    # 문제 3
    print("\n" + "-" * 70)
    print("【문제 3】")
    print("딕셔너리의 Key로 사용할 수 없는 것은? (객관식)")
    print("① 문자열")
    print("② 숫자")
    print("③ 튜플")
    print("④ 리스트")
    answer = input("\n정답: ").strip()
    if answer in ["④", "4"]:
        print("✅ 정답입니다!")
        score += 1
    else:
        print("❌ 오답입니다. 정답은 ④입니다.")
        print("   리스트는 변경 가능한 자료형이므로 Key로 사용할 수 없습니다.")
    
    # 문제 4
    print("\n" + "-" * 70)
    print("【문제 4】")
    print("딕셔너리에 새로운 Key-Value를 추가하는 방법은? (객관식)")
    print("① dict.add(key, value)")
    print("② dict.append(key, value)")
    print("③ dict[key] = value")
    print("④ dict.insert(key, value)")
    answer = input("\n정답: ").strip()
    if answer in ["③", "3"]:
        print("✅ 정답입니다!")
        score += 1
    else:
        print("❌ 오답입니다. 정답은 ③입니다.")
        print("   dict[key] = value로 추가 또는 수정합니다.")
    
    # 문제 5
    print("\n" + "-" * 70)
    print("【문제 5】")
    print("리스트와 딕셔너리의 주요 차이점은? (객관식)")
    print("① 리스트는 인덱스로, 딕셔너리는 Key로 접근")
    print("② 리스트는 변경 가능, 딕셔너리는 변경 불가능")
    print("③ 리스트는 중복 불가, 딕셔너리는 중복 가능")
    print("④ 리스트는 순서 없음, 딕셔너리는 순서 있음")
    answer = input("\n정답: ").strip()
    if answer in ["①", "1"]:
        print("✅ 정답입니다!")
        score += 1
    else:
        print("❌ 오답입니다. 정답은 ①입니다.")
        print("   리스트는 인덱스(숫자)로, 딕셔너리는 Key(이름)로 접근합니다.")
    
    # 결과 출력
    print("\n" + "=" * 70)
    print(f"📊 퀴즈 결과: {score}/{total}점 ({score/total*100:.0f}%)")
    print("=" * 70)
    
    if score == total:
        print("🎉 완벽합니다! 딕셔너리를 완전히 이해하셨네요!")
    elif score >= 3:
        print("👍 잘하셨습니다! 조금만 더 복습하면 완벽해질 거예요!")
    else:
        print("💪 다시 한 번 이론을 복습해보세요!")
    print()


def summary():
    """정리 및 마무리"""
    print("\n" + "=" * 70)
    print("🎓 2교시 정리")
    print("=" * 70)
    
    print("""
오늘 배운 내용:

1️⃣ 딕셔너리(Dictionary)
   • Key-Value 구조로 데이터 저장
   • 중괄호 {}로 표현: {"name": "홍길동"}
   • Key로 빠르게 값을 찾을 수 있음
   • person["name"] 또는 person.get("name")으로 조회

2️⃣ 딕셔너리 조작
   • 추가: dict[key] = value
   • 수정: dict[key] = new_value
   • 삭제: del dict[key] 또는 dict.pop(key)
   • 조회: dict[key] 또는 dict.get(key, 기본값)

3️⃣ 딕셔너리 메서드
   • keys(): 모든 Key 가져오기
   • values(): 모든 Value 가져오기
   • items(): Key-Value 쌍 가져오기
   • update(): 여러 항목 추가/수정

4️⃣ 복합 자료구조
   • 리스트 안에 딕셔너리 (가장 많이 사용!)
   • 딕셔너리 안에 리스트
   • 딕셔너리 안에 딕셔너리

5️⃣ 자료구조 비교
   • 리스트: 순서가 있고 인덱스로 접근
   • 튜플: 순서가 있지만 변경 불가
   • 딕셔너리: 순서 없고 Key로 접근

💡 핵심 코드 패턴:
    # 딕셔너리 생성
    person = {"name": "홍길동", "age": 25}
    
    # 값 조회 (안전)
    name = person.get("name", "이름 없음")
    
    # 값 추가/수정
    person["city"] = "서울"
    
    # 반복문 (가장 유용!)
    for key, value in person.items():
        print(f"{key}: {value}")
    
    # 리스트 + 딕셔너리 (실무에서 필수!)
    products = [
        {"name": "상품1", "price": 1000},
        {"name": "상품2", "price": 2000}
    ]

다음 시간 예고:
3교시에서는 조건문(if)과 반복문(for, while)을 배웁니다!
지금까지 배운 리스트, 튜플, 딕셔너리를 자동으로 처리하는
강력한 도구를 배워봐요!

수고하셨습니다! 🎉
    """)


def main_menu():
    """메인 메뉴"""
    while True:
        print("\n" + "=" * 70)
        print("📚 2교시: 딕셔너리 - 학습 메뉴")
        print("=" * 70)
        print("\n[이론 + 예제]")
        print("  1. 섹션 1 - 딕셔너리란 무엇인가?")
        print("  2. 섹션 2 - 딕셔너리 기본 조작")
        print("  3. 섹션 3 - 딕셔너리 메서드")
        print("  4. 섹션 4 - 리스트 + 딕셔너리 조합")
        print("  5. 섹션 5 - 자료구조 비교")
        print("  6. 섹션 6 - 실무 예제 (제품 관리)")
        
        print("\n[연습 및 평가]")
        print("  7. 연습 문제")
        print("  8. 연습 문제 해설")
        print("  9. 퀴즈 (5문항)")
        print(" 10. 정리 및 마무리")
        
        print("\n[전체 실행]")
        print("  0. 모든 섹션 순서대로 실행")
        print("  q. 종료")
        
        choice = input("\n선택하세요: ").strip()
        
        if choice == "0":
            intro()
            section1_what_is_dictionary()
            input("\nEnter를 눌러 계속...")
            section2_basic_operations()
            input("\nEnter를 눌러 계속...")
            section3_methods()
            input("\nEnter를 눌러 계속...")
            section4_nested_structures()
            input("\nEnter를 눌러 계속...")
            section5_comparison()
            input("\nEnter를 눌러 계속...")
            section6_product_system()
            input("\nEnter를 눌러 계속...")
            practice_problems()
            input("\nEnter를 눌러 계속...")
            practice_solutions()
            input("\nEnter를 눌러 계속...")
            quiz()
            summary()
            print("\n전체 학습을 완료했습니다! 🎉")
            break
        elif choice == "1":
            section1_what_is_dictionary()
        elif choice == "2":
            section2_basic_operations()
        elif choice == "3":
            section3_methods()
        elif choice == "4":
            section4_nested_structures()
        elif choice == "5":
            section5_comparison()
        elif choice == "6":
            section6_product_system()
        elif choice == "7":
            practice_problems()
        elif choice == "8":
            practice_solutions()
        elif choice == "9":
            quiz()
        elif choice == "10":
            summary()
        elif choice.lower() == "q":
            print("\n학습을 종료합니다. 수고하셨습니다! 👋")
            break
        else:
            print("\n❌ 올바른 번호를 선택해주세요.")
        
        if choice != "0":
            input("\nEnter를 눌러 메뉴로 돌아가기...")


if __name__ == "__main__":
    intro()
    main_menu()
