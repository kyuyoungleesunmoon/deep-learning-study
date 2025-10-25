"""
웹 스크레이핑 입문 - 1교시: Requests 라이브러리 실습

대상: 파이썬을 처음 배우는 직장인
학습 시간: 30분
"""

import requests
import time
from urllib.robotparser import RobotFileParser


# ============================================================================
# 섹션 1: 기본 HTML 가져오기
# ============================================================================

def section1_basic_fetch():
    """섹션 1: 기본적인 웹페이지 HTML 가져오기"""
    print("\n" + "="*70)
    print("섹션 1: 기본 HTML 가져오기")
    print("="*70)
    
    print("\n[이론]")
    print("requests.get(url)을 사용하면 웹페이지의 HTML을 가져올 수 있습니다.")
    print("response.text에 HTML 내용이 문자열로 저장됩니다.")
    
    print("\n[예제 1-1] Example.com HTML 가져오기")
    print("-" * 50)
    
    url = 'http://example.com'
    
    try:
        # 웹페이지 요청
        response = requests.get(url)
        
        # 상태 코드 확인
        print(f"✅ 상태 코드: {response.status_code}")
        
        # HTML 길이 출력
        print(f"📄 HTML 길이: {len(response.text)} 문자")
        
        # HTML 일부 출력 (처음 300자)
        print(f"\n📝 HTML 미리보기 (첫 300자):")
        print("-" * 50)
        print(response.text[:300])
        print("...")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    print("\n[설명]")
    print("1. requests.get(url) - URL에 접속하여 응답 받기")
    print("2. response.status_code - 요청 성공 여부 확인 (200 = 성공)")
    print("3. response.text - HTML 내용을 문자열로 반환")
    print("4. len(response.text) - HTML 문자열의 길이")


# ============================================================================
# 섹션 2: 상태 코드와 예외 처리
# ============================================================================

def section2_status_codes():
    """섹션 2: HTTP 상태 코드 이해하기"""
    print("\n" + "="*70)
    print("섹션 2: 상태 코드와 예외 처리")
    print("="*70)
    
    print("\n[이론]")
    print("HTTP 상태 코드는 요청이 성공했는지 실패했는지 알려줍니다.")
    print("- 200: 성공")
    print("- 404: 페이지를 찾을 수 없음")
    print("- 403: 접근 권한 없음")
    print("- 500: 서버 오류")
    
    print("\n[예제 2-1] 다양한 URL 테스트")
    print("-" * 50)
    
    # 테스트할 URL 목록
    test_urls = [
        ('http://example.com', '정상 페이지'),
        ('http://httpstat.us/404', '404 에러 테스트'),
        ('http://httpstat.us/500', '500 에러 테스트')
    ]
    
    for url, description in test_urls:
        print(f"\n🔍 테스트: {description}")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, timeout=5)
            print(f"   ✅ 상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ 성공! HTML 길이: {len(response.text)}")
            elif response.status_code == 404:
                print(f"   ⚠️ 페이지를 찾을 수 없습니다")
            else:
                print(f"   ⚠️ 예상치 못한 상태 코드")
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️ 타임아웃: 응답이 너무 느립니다")
        except requests.exceptions.ConnectionError:
            print(f"   🔌 연결 오류: 서버에 접속할 수 없습니다")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
    
    print("\n[설명]")
    print("1. try-except로 오류를 안전하게 처리합니다")
    print("2. timeout 파라미터로 최대 대기 시간을 지정합니다")
    print("3. 상태 코드에 따라 적절한 처리를 합니다")


# ============================================================================
# 섹션 3: 함수로 재사용 가능하게 만들기 (최신 스타일)
# ============================================================================

def section3_reusable_function():
    """섹션 3: 재사용 가능한 함수 만들기"""
    print("\n" + "="*70)
    print("섹션 3: 재사용 가능한 함수 만들기")
    print("="*70)
    
    print("\n[이론]")
    print("실무에서는 웹페이지를 가져오는 코드를 함수로 만들어 재사용합니다.")
    print("예외 처리를 포함하면 더 안전한 코드가 됩니다.")
    
    def fetch_webpage(url):
        """
        웹페이지의 HTML을 가져오는 함수
        
        Args:
            url (str): 가져올 웹페이지 URL
        
        Returns:
            str: HTML 내용 또는 None (실패 시)
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # 4xx, 5xx 에러 시 예외 발생
            return response.text
            
        except requests.exceptions.Timeout:
            print(f"⏱️ 타임아웃: {url}")
            return None
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP 오류: {e}")
            return None
            
        except requests.exceptions.ConnectionError:
            print(f"🔌 연결 오류: {url}")
            return None
            
        except Exception as e:
            print(f"⚠️ 알 수 없는 오류: {e}")
            return None
    
    print("\n[예제 3-1] 함수 사용하기")
    print("-" * 50)
    
    url = 'http://example.com'
    html = fetch_webpage(url)
    
    if html:
        print(f"✅ 성공! HTML 길이: {len(html)} 문자")
        print(f"\n첫 200자:")
        print(html[:200])
    else:
        print("❌ HTML을 가져오는 데 실패했습니다")
    
    print("\n[설명]")
    print("1. 함수는 한 번 정의하면 여러 번 재사용 가능합니다")
    print("2. docstring(\"\"\" \"\"\")으로 함수 설명을 작성합니다")
    print("3. raise_for_status()는 4xx, 5xx 에러를 자동으로 감지합니다")
    print("4. 각 예외 타입별로 적절한 메시지를 출력합니다")


# ============================================================================
# 섹션 4: User-Agent 설정 (실무 필수)
# ============================================================================

def section4_user_agent():
    """섹션 4: User-Agent 설정하기"""
    print("\n" + "="*70)
    print("섹션 4: User-Agent 설정")
    print("="*70)
    
    print("\n[이론]")
    print("일부 웹사이트는 프로그램의 접근을 차단합니다.")
    print("User-Agent를 설정하면 일반 브라우저처럼 보이게 할 수 있습니다.")
    
    print("\n[예제 4-1] User-Agent 없이 요청")
    print("-" * 50)
    
    url = 'http://example.com'
    
    # User-Agent 없이 요청
    response1 = requests.get(url)
    print(f"User-Agent 없음:")
    print(f"  요청 헤더: {response1.request.headers.get('User-Agent', '없음')}")
    print(f"  상태 코드: {response1.status_code}")
    
    print("\n[예제 4-2] User-Agent 설정하여 요청")
    print("-" * 50)
    
    # User-Agent 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    response2 = requests.get(url, headers=headers)
    print(f"User-Agent 설정:")
    print(f"  요청 헤더: {response2.request.headers.get('User-Agent')[:50]}...")
    print(f"  상태 코드: {response2.status_code}")
    
    print("\n[예제 4-3] 실무에서 자주 사용하는 헤더")
    print("-" * 50)
    
    # 더 완전한 헤더 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    
    response3 = requests.get(url, headers=headers)
    print(f"완전한 헤더 설정:")
    print(f"  상태 코드: {response3.status_code}")
    print(f"  응답 시간: {response3.elapsed.total_seconds():.2f}초")
    print(f"  인코딩: {response3.encoding}")
    
    print("\n[설명]")
    print("1. User-Agent는 어떤 브라우저/프로그램인지 알려주는 정보입니다")
    print("2. headers 딕셔너리에 User-Agent를 설정합니다")
    print("3. 실무에서는 Chrome, Firefox 등의 User-Agent를 사용합니다")
    print("4. Accept, Accept-Language 등 추가 헤더도 설정할 수 있습니다")


# ============================================================================
# 섹션 5: Session 사용 (효율적인 방법)
# ============================================================================

def section5_session():
    """섹션 5: Session으로 효율적으로 요청하기"""
    print("\n" + "="*70)
    print("섹션 5: Session 사용하기")
    print("="*70)
    
    print("\n[이론]")
    print("여러 페이지를 연속으로 요청할 때는 Session을 사용하는 것이 효율적입니다.")
    print("Session은 연결을 재사용하여 속도를 향상시킵니다.")
    
    print("\n[예제 5-1] 일반 requests.get() 사용")
    print("-" * 50)
    
    start_time = time.time()
    
    urls = ['http://example.com'] * 3  # 같은 URL을 3번 요청
    
    for i, url in enumerate(urls, 1):
        response = requests.get(url)
        print(f"{i}. 상태 코드: {response.status_code}, 길이: {len(response.text)}")
    
    elapsed1 = time.time() - start_time
    print(f"⏱️ 소요 시간: {elapsed1:.2f}초")
    
    print("\n[예제 5-2] Session 사용")
    print("-" * 50)
    
    start_time = time.time()
    
    with requests.Session() as session:
        # Session에 헤더 설정 (모든 요청에 적용됨)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        
        for i, url in enumerate(urls, 1):
            response = session.get(url)
            print(f"{i}. 상태 코드: {response.status_code}, 길이: {len(response.text)}")
    
    elapsed2 = time.time() - start_time
    print(f"⏱️ 소요 시간: {elapsed2:.2f}초")
    
    print(f"\n💡 Session이 {elapsed1 - elapsed2:.2f}초 더 빠릅니다!")
    
    print("\n[설명]")
    print("1. with requests.Session()으로 세션을 시작합니다")
    print("2. session.headers.update()로 모든 요청에 적용될 헤더를 설정합니다")
    print("3. session.get()으로 요청합니다 (일반 requests.get()과 동일)")
    print("4. with문이 끝나면 자동으로 세션이 종료됩니다")


# ============================================================================
# 섹션 6: robots.txt 확인하기 (윤리적 스크레이핑)
# ============================================================================

def section6_robots_txt():
    """섹션 6: robots.txt 확인하기"""
    print("\n" + "="*70)
    print("섹션 6: robots.txt 확인")
    print("="*70)
    
    print("\n[이론]")
    print("robots.txt는 웹사이트가 허용/금지하는 스크레이핑 규칙을 담은 파일입니다.")
    print("윤리적인 스크레이핑을 위해 반드시 확인해야 합니다.")
    
    print("\n[예제 6-1] robots.txt 직접 확인하기")
    print("-" * 50)
    
    url = 'https://www.google.com/robots.txt'
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ robots.txt 내용 (처음 500자):")
            print("-" * 50)
            print(response.text[:500])
            print("...")
        else:
            print(f"❌ robots.txt를 가져올 수 없습니다 (상태 코드: {response.status_code})")
            
    except Exception as e:
        print(f"⚠️ 오류: {e}")
    
    print("\n[예제 6-2] robotparser로 자동 확인")
    print("-" * 50)
    
    # RobotFileParser 사용
    rp = RobotFileParser()
    rp.set_url('https://www.google.com/robots.txt')
    
    try:
        rp.read()
        
        # 테스트할 URL들
        test_paths = [
            'https://www.google.com/search',
            'https://www.google.com/maps',
        ]
        
        print("스크레이핑 허용 여부 확인:")
        for path in test_paths:
            can_fetch = rp.can_fetch('*', path)
            status = "✅ 허용" if can_fetch else "❌ 금지"
            print(f"  {path}: {status}")
            
    except Exception as e:
        print(f"⚠️ 오류: {e}")
    
    print("\n[설명]")
    print("1. robots.txt는 URL + '/robots.txt'로 확인합니다")
    print("2. User-agent, Disallow, Allow 등의 규칙이 있습니다")
    print("3. RobotFileParser를 사용하면 자동으로 규칙을 확인할 수 있습니다")
    print("4. can_fetch()로 특정 경로의 스크레이핑 허용 여부를 확인합니다")


# ============================================================================
# 실습 미션
# ============================================================================

def practice_missions():
    """실습 미션 - 직접 해보기"""
    print("\n" + "="*70)
    print("💪 실습 미션")
    print("="*70)
    
    print("\n[미션 1] 여러 웹사이트의 상태 확인하기")
    print("-" * 50)
    print("다음 웹사이트들의 상태 코드와 응답 시간을 확인하는 프로그램을 작성하세요.")
    print("힌트: requests.get(), response.status_code, response.elapsed.total_seconds()")
    print()
    print("확인할 사이트:")
    print("- http://example.com")
    print("- https://www.python.org")
    print("- http://httpstat.us/200")
    
    print("\n[미션 2] 안전한 웹페이지 가져오기 함수 만들기")
    print("-" * 50)
    print("다음 기능을 가진 함수를 작성하세요:")
    print("1. URL을 받아서 HTML을 반환")
    print("2. User-Agent 헤더 포함")
    print("3. 타임아웃 5초 설정")
    print("4. 예외 처리 포함")
    print("5. 성공/실패 메시지 출력")
    
    print("\n💡 힌트는 각 섹션의 예제 코드를 참고하세요!")


# ============================================================================
# 퀴즈
# ============================================================================

def quiz():
    """퀴즈 - 학습 내용 확인"""
    print("\n" + "="*70)
    print("📝 퀴즈")
    print("="*70)
    
    questions = [
        {
            "question": "1. 웹페이지의 HTML을 가져오기 위해 사용하는 requests의 메서드는?",
            "options": ["a) requests.post()", "b) requests.get()", "c) requests.put()", "d) requests.fetch()"],
            "answer": "b"
        },
        {
            "question": "2. HTTP 상태 코드 200은 무엇을 의미하나요?",
            "options": ["a) 페이지를 찾을 수 없음", "b) 성공", "c) 서버 오류", "d) 권한 없음"],
            "answer": "b"
        },
        {
            "question": "3. 웹사이트가 봇 접근을 막는 것을 우회하기 위해 설정하는 것은?",
            "options": ["a) Cookie", "b) Session", "c) User-Agent", "d) Token"],
            "answer": "c"
        }
    ]
    
    score = 0
    
    for i, q in enumerate(questions, 1):
        print(f"\n{q['question']}")
        for option in q['options']:
            print(f"  {option}")
        
        # 자동 채점 모드 (실제로는 input()으로 받아야 함)
        print(f"\n(정답은 하단의 '정답 및 해설'에서 확인하세요)")
    
    print("\n" + "="*70)


# ============================================================================
# 정답 및 해설
# ============================================================================

def show_answers():
    """정답 및 해설"""
    print("\n" + "="*70)
    print("✅ 정답 및 해설")
    print("="*70)
    
    print("""
<details>
<summary>정답 보기</summary>

1. 정답: b) requests.get()
   해설: GET 메서드는 서버에서 데이터를 가져올 때 사용합니다.
         웹 스크레이핑에서 가장 많이 사용하는 메서드입니다.

2. 정답: b) 성공
   해설: 200은 요청이 성공적으로 처리되었다는 의미입니다.
         404는 페이지 없음, 500은 서버 오류입니다.

3. 정답: c) User-Agent
   해설: User-Agent는 어떤 브라우저/프로그램인지 알려주는 정보입니다.
         일부 웹사이트는 User-Agent가 없으면 봇으로 간주하여 차단합니다.

</details>
    """)


# ============================================================================
# 종합 실습 예제
# ============================================================================

def comprehensive_example():
    """종합 실습: 실무 스타일의 완전한 예제"""
    print("\n" + "="*70)
    print("🚀 종합 실습: 실무 스타일 웹 스크레이핑")
    print("="*70)
    
    print("\n[실무에서 사용하는 완전한 예제]")
    print("-" * 50)
    
    def fetch_webpage_pro(url, timeout=5):
        """
        프로페셔널한 웹페이지 가져오기 함수
        
        Args:
            url (str): 가져올 URL
            timeout (int): 타임아웃 시간(초)
        
        Returns:
            dict: {
                'success': bool,
                'html': str or None,
                'status_code': int or None,
                'error': str or None
            }
        """
        # User-Agent 설정
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9',
        }
        
        try:
            # 요청 시작 시간 기록
            start_time = time.time()
            
            # 요청 보내기
            response = requests.get(url, headers=headers, timeout=timeout)
            
            # 응답 시간 계산
            elapsed = time.time() - start_time
            
            # 상태 확인
            response.raise_for_status()
            
            # 성공 로그
            print(f"✅ 성공: {url}")
            print(f"   상태 코드: {response.status_code}")
            print(f"   응답 시간: {elapsed:.2f}초")
            print(f"   HTML 크기: {len(response.text)} 문자")
            print(f"   인코딩: {response.encoding}")
            
            return {
                'success': True,
                'html': response.text,
                'status_code': response.status_code,
                'error': None
            }
            
        except requests.exceptions.Timeout:
            error_msg = f"타임아웃: {url} (>{timeout}초)"
            print(f"⏱️ {error_msg}")
            return {
                'success': False,
                'html': None,
                'status_code': None,
                'error': error_msg
            }
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP 오류: {e}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'html': None,
                'status_code': response.status_code if 'response' in locals() else None,
                'error': error_msg
            }
            
        except requests.exceptions.ConnectionError:
            error_msg = f"연결 오류: {url}"
            print(f"🔌 {error_msg}")
            return {
                'success': False,
                'html': None,
                'status_code': None,
                'error': error_msg
            }
            
        except Exception as e:
            error_msg = f"알 수 없는 오류: {e}"
            print(f"⚠️ {error_msg}")
            return {
                'success': False,
                'html': None,
                'status_code': None,
                'error': error_msg
            }
    
    # 실행 예시
    result = fetch_webpage_pro('http://example.com')
    
    if result['success']:
        print(f"\n📄 HTML 미리보기 (첫 300자):")
        print("-" * 50)
        print(result['html'][:300])
        print("...")
    else:
        print(f"\n❌ 실패: {result['error']}")
    
    print("\n[설명]")
    print("이 함수는 실무에서 바로 사용할 수 있는 수준입니다:")
    print("1. 명확한 docstring (함수 설명)")
    print("2. 완전한 헤더 설정")
    print("3. 모든 예외 처리")
    print("4. 상세한 로그 출력")
    print("5. 딕셔너리로 결과 반환 (성공/실패, 데이터, 오류 정보)")


# ============================================================================
# 메인 메뉴
# ============================================================================

def show_menu():
    """메뉴 표시"""
    print("\n" + "="*70)
    print("웹 스크레이핑 입문 - 1교시: Requests 라이브러리")
    print("="*70)
    print("\n[학습 섹션]")
    print("1. 기본 HTML 가져오기")
    print("2. 상태 코드와 예외 처리")
    print("3. 재사용 가능한 함수 만들기")
    print("4. User-Agent 설정")
    print("5. Session 사용하기")
    print("6. robots.txt 확인")
    print("\n[실습 및 평가]")
    print("7. 실습 미션")
    print("8. 퀴즈")
    print("9. 정답 및 해설")
    print("10. 종합 실습 (실무 스타일)")
    print("\n[전체 실행]")
    print("0. 모든 섹션 순서대로 실행")
    print("q. 종료")
    print("="*70)


def run_all_sections():
    """모든 섹션 순서대로 실행"""
    sections = [
        section1_basic_fetch,
        section2_status_codes,
        section3_reusable_function,
        section4_user_agent,
        section5_session,
        section6_robots_txt,
        practice_missions,
        quiz,
        show_answers,
        comprehensive_example
    ]
    
    for section in sections:
        section()
        print("\n" + "-"*70)
        input("⏎ Enter를 눌러 계속...")


def main():
    """메인 함수"""
    while True:
        show_menu()
        choice = input("\n선택하세요 (0-10, q): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 학습을 종료합니다. 수고하셨습니다!")
            break
        elif choice == '0':
            run_all_sections()
        elif choice == '1':
            section1_basic_fetch()
        elif choice == '2':
            section2_status_codes()
        elif choice == '3':
            section3_reusable_function()
        elif choice == '4':
            section4_user_agent()
        elif choice == '5':
            section5_session()
        elif choice == '6':
            section6_robots_txt()
        elif choice == '7':
            practice_missions()
        elif choice == '8':
            quiz()
        elif choice == '9':
            show_answers()
        elif choice == '10':
            comprehensive_example()
        else:
            print("⚠️ 올바른 번호를 입력하세요.")
        
        if choice != '0' and choice != 'q':
            input("\n⏎ Enter를 눌러 메뉴로 돌아가기...")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║         웹 스크레이핑 입문 - 1교시: Requests 라이브러리          ║
    ║                                                                  ║
    ║                  파이썬으로 웹페이지 가져오기                     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("📚 학습 목표:")
    print("  1. 웹 스크레이핑의 개념 이해")
    print("  2. Requests 라이브러리로 HTML 가져오기")
    print("  3. 예외 처리와 User-Agent 설정")
    print("  4. 윤리적 스크레이핑 (robots.txt)")
    
    main()
