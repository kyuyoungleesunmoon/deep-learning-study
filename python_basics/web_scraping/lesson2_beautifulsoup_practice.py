"""
웹 스크레이핑 입문 - 2교시: BeautifulSoup 기본 사용법 실습

대상: 파이썬을 처음 배우는 직장인
학습 시간: 30분
"""

import requests
from bs4 import BeautifulSoup


# ============================================================================
# 섹션 1: BeautifulSoup 기본 - HTML 파싱하기
# ============================================================================

def section1_basic_parsing():
    """섹션 1: 기본 HTML 파싱"""
    print("\n" + "="*70)
    print("섹션 1: BeautifulSoup 기본 - HTML 파싱하기")
    print("="*70)
    
    print("\n[이론]")
    print("BeautifulSoup은 HTML을 파싱(구조적으로 해석)하여")
    print("원하는 데이터를 쉽게 추출할 수 있게 해주는 라이브러리입니다.")
    
    print("\n[예제 1-1] 간단한 HTML 파싱")
    print("-" * 50)
    
    # HTML 문자열
    html = """
    <html>
        <head>
            <title>내 첫 웹페이지</title>
        </head>
        <body>
            <h1>안녕하세요!</h1>
            <p>BeautifulSoup를 배웁니다.</p>
        </body>
    </html>
    """
    
    # 파싱
    soup = BeautifulSoup(html, 'html.parser')
    
    # 예쁘게 출력
    print("📄 파싱된 HTML:")
    print(soup.prettify())
    
    print("\n[예제 1-2] 특정 태그 찾기")
    print("-" * 50)
    
    # title 태그 찾기
    title_tag = soup.find('title')
    print(f"제목 태그: {title_tag}")
    print(f"제목 텍스트: {title_tag.text}")
    
    # h1 태그 찾기
    h1_tag = soup.find('h1')
    print(f"\nH1 태그: {h1_tag}")
    print(f"H1 텍스트: {h1_tag.text}")
    
    # p 태그 찾기
    p_tag = soup.find('p')
    print(f"\nP 태그: {p_tag}")
    print(f"P 텍스트: {p_tag.text}")
    
    print("\n[설명]")
    print("1. BeautifulSoup(html, 'html.parser')로 HTML을 파싱합니다")
    print("2. soup.find('태그명')으로 원하는 태그를 찾습니다")
    print("3. tag.text로 태그의 텍스트 내용을 추출합니다")
    print("4. soup.prettify()로 HTML을 보기 좋게 출력할 수 있습니다")


# ============================================================================
# 섹션 2: find()와 find_all() - 태그 찾기
# ============================================================================

def section2_find_methods():
    """섹션 2: find()와 find_all() 사용법"""
    print("\n" + "="*70)
    print("섹션 2: find()와 find_all() - 태그 찾기")
    print("="*70)
    
    print("\n[이론]")
    print("find() - 첫 번째 태그 하나만 반환")
    print("find_all() - 조건에 맞는 모든 태그를 리스트로 반환")
    
    html = """
    <div class="news">
        <h2>첫 번째 뉴스</h2>
        <p class="content">첫 번째 뉴스 내용</p>
        
        <h2>두 번째 뉴스</h2>
        <p class="content">두 번째 뉴스 내용</p>
        
        <h2>세 번째 뉴스</h2>
        <p class="content">세 번째 뉴스 내용</p>
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    
    print("\n[예제 2-1] find() - 첫 번째 것만")
    print("-" * 50)
    
    # find()는 첫 번째 것만 반환
    first_h2 = soup.find('h2')
    print(f"첫 번째 h2: {first_h2.text}")
    print(f"타입: {type(first_h2)}")  # bs4.element.Tag
    
    print("\n[예제 2-2] find_all() - 모든 것")
    print("-" * 50)
    
    # find_all()은 모든 것을 리스트로 반환
    all_h2 = soup.find_all('h2')
    print(f"찾은 h2 개수: {len(all_h2)}")
    print(f"타입: {type(all_h2)}")  # list
    
    print("\n모든 h2 태그:")
    for i, h2 in enumerate(all_h2, 1):
        print(f"{i}. {h2.text}")
    
    print("\n[예제 2-3] 클래스로 찾기")
    print("-" * 50)
    
    # class 속성으로 찾기 (주의: class_ 사용)
    content_tags = soup.find_all('p', class_='content')
    print(f"class='content'인 p 태그 개수: {len(content_tags)}")
    
    for i, p in enumerate(content_tags, 1):
        print(f"{i}. {p.text}")
    
    print("\n[예제 2-4] 여러 조건으로 찾기")
    print("-" * 50)
    
    html2 = """
    <a href="/news/1" class="news-link">뉴스 1</a>
    <a href="/news/2" class="news-link">뉴스 2</a>
    <a href="/notice/1" class="notice-link">공지 1</a>
    """
    
    soup2 = BeautifulSoup(html2, 'html.parser')
    
    # 태그 + 클래스
    news_links = soup2.find_all('a', class_='news-link')
    print(f"news-link 클래스를 가진 a 태그: {len(news_links)}개")
    
    for link in news_links:
        print(f"  - {link.text}")
    
    print("\n[설명]")
    print("1. find('태그')는 첫 번째 것만 반환 (Tag 객체)")
    print("2. find_all('태그')는 모든 것을 반환 (리스트)")
    print("3. class로 찾을 때는 class_ (언더스코어) 사용")
    print("4. 여러 조건을 조합하여 정확하게 찾을 수 있습니다")


# ============================================================================
# 섹션 3: 데이터 추출 - 텍스트와 속성
# ============================================================================

def section3_extract_data():
    """섹션 3: 텍스트와 속성 추출하기"""
    print("\n" + "="*70)
    print("섹션 3: 데이터 추출 - 텍스트와 속성")
    print("="*70)
    
    print("\n[이론]")
    print("태그에서 추출할 수 있는 것:")
    print("1. 텍스트: tag.text 또는 tag.get_text()")
    print("2. 속성: tag['속성명'] 또는 tag.get('속성명')")
    
    print("\n[예제 3-1] 텍스트 추출")
    print("-" * 50)
    
    html = """
    <div class="article">
        <h2>   제목입니다   </h2>
        <p>본문 내용입니다.</p>
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    
    h2 = soup.find('h2')
    print(f"원본 텍스트: '{h2.text}'")
    print(f"공백 제거: '{h2.text.strip()}'")
    
    # get_text() 사용
    print(f"\nget_text(): '{h2.get_text()}'")
    print(f"get_text(strip=True): '{h2.get_text(strip=True)}'")
    
    print("\n[예제 3-2] 링크(href) 추출")
    print("-" * 50)
    
    html_links = """
    <div>
        <a href="https://www.python.org">Python 공식 사이트</a>
        <a href="https://www.google.com">구글</a>
        <a href="/news/123">뉴스 기사</a>
    </div>
    """
    
    soup = BeautifulSoup(html_links, 'html.parser')
    
    links = soup.find_all('a')
    print(f"찾은 링크 개수: {len(links)}\n")
    
    for i, link in enumerate(links, 1):
        text = link.text
        url = link['href']  # 속성 추출
        print(f"{i}. 텍스트: {text}")
        print(f"   URL: {url}\n")
    
    print("[예제 3-3] 이미지(src, alt) 추출")
    print("-" * 50)
    
    html_imgs = """
    <div>
        <img src="/images/logo.png" alt="로고">
        <img src="/images/banner.jpg" alt="배너">
    </div>
    """
    
    soup = BeautifulSoup(html_imgs, 'html.parser')
    
    images = soup.find_all('img')
    
    for i, img in enumerate(images, 1):
        src = img['src']
        alt = img['alt']
        print(f"{i}. 이미지 경로: {src}")
        print(f"   대체 텍스트: {alt}\n")
    
    print("[예제 3-4] 안전하게 속성 추출 (get 메서드)")
    print("-" * 50)
    
    html_safe = '<a>링크인데 href가 없음</a>'
    soup = BeautifulSoup(html_safe, 'html.parser')
    
    link = soup.find('a')
    
    # 위험: KeyError 발생 가능
    # href = link['href']  # ❌
    
    # 안전: get() 사용
    href = link.get('href')
    if href:
        print(f"URL: {href}")
    else:
        print("✅ href 속성이 없습니다 (안전하게 처리)")
    
    # 기본값 설정
    href = link.get('href', '링크 없음')
    print(f"기본값 사용: {href}")
    
    print("\n[설명]")
    print("1. tag.text 또는 tag.get_text()로 텍스트를 추출합니다")
    print("2. strip=True로 앞뒤 공백을 제거할 수 있습니다")
    print("3. tag['속성명']으로 속성 값을 추출합니다")
    print("4. tag.get('속성명')을 사용하면 속성이 없어도 안전합니다")


# ============================================================================
# 섹션 4: HTML 구조 탐색 - 부모, 자식, 형제
# ============================================================================

def section4_navigate_tree():
    """섹션 4: HTML 구조 탐색하기"""
    print("\n" + "="*70)
    print("섹션 4: HTML 구조 탐색 - 부모, 자식, 형제")
    print("="*70)
    
    print("\n[이론]")
    print("HTML은 트리 구조입니다. 태그 간의 관계:")
    print("- 부모(parent): 상위 태그")
    print("- 자식(children): 하위 태그")
    print("- 형제(siblings): 같은 레벨의 태그")
    
    print("\n[예제 4-1] 부모 요소 접근")
    print("-" * 50)
    
    html = """
    <div class="container">
        <h1 id="title">제목</h1>
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    
    h1 = soup.find('h1')
    print(f"현재 태그: {h1.name}")
    print(f"현재 태그 id: {h1.get('id')}")
    
    # 부모 접근
    parent = h1.parent
    print(f"\n부모 태그: {parent.name}")
    print(f"부모 클래스: {parent.get('class')}")
    
    print("\n[예제 4-2] 자식 요소 접근")
    print("-" * 50)
    
    html2 = """
    <ul class="menu">
        <li>메뉴 1</li>
        <li>메뉴 2</li>
        <li>메뉴 3</li>
    </ul>
    """
    
    soup = BeautifulSoup(html2, 'html.parser')
    
    ul = soup.find('ul')
    
    # 모든 li 자식 찾기
    children = ul.find_all('li')
    print(f"자식(li) 개수: {len(children)}\n")
    
    for i, child in enumerate(children, 1):
        print(f"{i}. {child.text}")
    
    print("\n[예제 4-3] 형제 요소 접근")
    print("-" * 50)
    
    html3 = """
    <div>
        <h1>제목</h1>
        <p>첫 번째 문단</p>
        <p>두 번째 문단</p>
        <p>세 번째 문단</p>
    </div>
    """
    
    soup = BeautifulSoup(html3, 'html.parser')
    
    h1 = soup.find('h1')
    print(f"현재 위치: {h1.text}\n")
    
    # 다음 형제
    next_sib = h1.find_next_sibling()
    print(f"다음 형제: {next_sib.text}")
    
    # 모든 다음 형제들
    all_next = h1.find_next_siblings()
    print(f"\n모든 다음 형제 ({len(all_next)}개):")
    for i, sib in enumerate(all_next, 1):
        print(f"{i}. {sib.text}")
    
    print("\n[설명]")
    print("1. tag.parent로 부모 요소에 접근합니다")
    print("2. tag.find_all()이나 tag.children으로 자식에 접근합니다")
    print("3. tag.find_next_sibling()으로 다음 형제에 접근합니다")
    print("4. tag.find_next_siblings()로 모든 다음 형제를 찾습니다")


# ============================================================================
# 섹션 5: 실전 예제 - 웹페이지에서 데이터 추출
# ============================================================================

def section5_real_example():
    """섹션 5: 실전 - 모든 링크 추출하기"""
    print("\n" + "="*70)
    print("섹션 5: 실전 예제 - 웹페이지에서 모든 링크 추출")
    print("="*70)
    
    print("\n[이론]")
    print("실제 웹페이지에서 데이터를 추출하는 전체 프로세스:")
    print("1. requests로 HTML 가져오기")
    print("2. BeautifulSoup으로 파싱")
    print("3. find/find_all로 원하는 요소 찾기")
    print("4. 텍스트와 속성 추출")
    
    print("\n[예제 5-1] 완전한 실전 예제")
    print("-" * 50)
    
    def extract_all_links(url):
        """웹페이지의 모든 링크 추출"""
        try:
            print(f"📡 {url}에 접속 중...")
            
            # 1. HTML 가져오기
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            print(f"✅ 성공! (상태 코드: {response.status_code})")
            
            # 2. 파싱
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 3. 모든 a 태그 찾기
            links = soup.find_all('a')
            print(f"📊 찾은 링크 개수: {len(links)}\n")
            
            # 4. 데이터 추출 및 정리
            results = []
            for link in links:
                href = link.get('href')
                text = link.text.strip()
                
                if href:  # href가 있는 경우만
                    results.append({
                        'text': text if text else '(텍스트 없음)',
                        'url': href
                    })
            
            return results
            
        except requests.exceptions.Timeout:
            print(f"⏱️ 타임아웃: {url}")
            return []
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP 오류: {e}")
            return []
        except Exception as e:
            print(f"⚠️ 오류: {e}")
            return []
    
    # 테스트 (인터넷 연결 필요)
    url = 'http://example.com'
    links = extract_all_links(url)
    
    if links:
        print(f"추출 결과 (상위 5개):")
        print("-" * 50)
        for i, link in enumerate(links[:5], 1):
            print(f"{i}. {link['text']}")
            print(f"   → {link['url']}\n")
    
    print("\n[설명]")
    print("이 함수는 실무에서 바로 사용할 수 있습니다:")
    print("1. User-Agent 설정으로 안전하게 접근")
    print("2. 예외 처리로 오류에 대비")
    print("3. 딕셔너리로 구조화된 데이터 반환")
    print("4. 빈 텍스트나 없는 href 처리")


# ============================================================================
# 실습 미션
# ============================================================================

def practice_missions():
    """실습 미션 - 직접 해보기"""
    print("\n" + "="*70)
    print("💪 실습 미션")
    print("="*70)
    
    print("\n[미션 1] HTML에서 특정 태그의 텍스트만 추출하기")
    print("-" * 50)
    print("다음 HTML에서 모든 <h2> 태그의 텍스트를 추출하세요.")
    print()
    print("HTML:")
    print("""
    <div>
        <h2>제목 1</h2>
        <p>내용 1</p>
        <h2>제목 2</h2>
        <p>내용 2</p>
        <h2>제목 3</h2>
        <p>내용 3</p>
    </div>
    """)
    print("\n힌트: find_all('h2')와 반복문 사용")
    
    print("\n[미션 2] 링크와 텍스트를 딕셔너리로 저장하기")
    print("-" * 50)
    print("다음 HTML에서 모든 링크를 추출하여")
    print("{'text': '텍스트', 'url': 'URL'} 형태의 리스트로 만드세요.")
    print()
    print("HTML:")
    print("""
    <nav>
        <a href="/home">홈</a>
        <a href="/about">소개</a>
        <a href="/contact">연락처</a>
    </nav>
    """)
    print("\n힌트: find_all('a'), link.text, link['href']")
    
    print("\n💡 직접 코드를 작성해보세요!")


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
            "question": "1. BeautifulSoup으로 HTML을 파싱할 때 사용하는 기본 파서는?",
            "options": ["a) lxml", "b) html.parser", "c) xml.parser", "d) json.parser"],
            "answer": "b"
        },
        {
            "question": "2. 첫 번째 h1 태그를 찾는 올바른 코드는?",
            "options": ["a) soup.find('h1')", "b) soup.find_all('h1')[0]", "c) soup.h1", "d) 모두 가능"],
            "answer": "d"
        },
        {
            "question": "3. 태그의 href 속성을 안전하게 추출하는 방법은?",
            "options": ["a) tag['href']", "b) tag.href", "c) tag.get('href')", "d) tag.attribute('href')"],
            "answer": "c"
        }
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n{q['question']}")
        for option in q['options']:
            print(f"  {option}")
    
    print("\n(정답은 하단의 '정답 및 해설'에서 확인하세요)")
    print("="*70)


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

1. 정답: b) html.parser
   해설: html.parser는 Python 기본 제공 파서로 별도 설치가 필요 없습니다.
         초보자에게 가장 권장되는 파서입니다.
         lxml은 더 빠르지만 별도 설치가 필요합니다.

2. 정답: d) 모두 가능
   해설: soup.find('h1'), soup.find_all('h1')[0], soup.h1 
         모두 첫 번째 h1 태그를 찾습니다.
         하지만 find()가 가장 명시적이고 읽기 쉽습니다.

3. 정답: c) tag.get('href')
   해설: get() 메서드는 속성이 없어도 None을 반환하므로 안전합니다.
         tag['href']는 속성이 없으면 KeyError가 발생합니다.
         get('속성', '기본값')으로 기본값을 지정할 수도 있습니다.

</details>
    """)


# ============================================================================
# 종합 실습 예제
# ============================================================================

def comprehensive_example():
    """종합 실습: 뉴스 제목과 링크 추출"""
    print("\n" + "="*70)
    print("🚀 종합 실습: 뉴스 제목과 링크 추출")
    print("="*70)
    
    print("\n[실무 스타일의 완전한 예제]")
    print("-" * 50)
    
    # 샘플 HTML (실제 뉴스 사이트 구조 모방)
    sample_html = """
    <html>
    <body>
        <div class="news-list">
            <div class="news-item">
                <a href="/news/1" class="news-title">
                    파이썬으로 업무 자동화 성공 사례
                </a>
                <span class="news-date">2024-01-15</span>
                <span class="news-press">테크뉴스</span>
            </div>
            <div class="news-item">
                <a href="/news/2" class="news-title">
                    웹 스크레이핑으로 데이터 수집하기
                </a>
                <span class="news-date">2024-01-14</span>
                <span class="news-press">데이터저널</span>
            </div>
            <div class="news-item">
                <a href="/news/3" class="news-title">
                    BeautifulSoup 완벽 가이드
                </a>
                <span class="news-date">2024-01-13</span>
                <span class="news-press">코딩매거진</span>
            </div>
        </div>
    </body>
    </html>
    """
    
    def extract_news(html):
        """뉴스 제목, 링크, 날짜, 언론사 추출"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # 모든 뉴스 항목 찾기
        news_items = soup.find_all('div', class_='news-item')
        print(f"📊 찾은 뉴스: {len(news_items)}개\n")
        
        results = []
        
        for item in news_items:
            # 제목과 링크
            title_tag = item.find('a', class_='news-title')
            title = title_tag.text.strip()
            link = title_tag.get('href', '')
            
            # 날짜
            date_tag = item.find('span', class_='news-date')
            date = date_tag.text if date_tag else '날짜 없음'
            
            # 언론사
            press_tag = item.find('span', class_='news-press')
            press = press_tag.text if press_tag else '언론사 없음'
            
            results.append({
                'title': title,
                'link': link,
                'date': date,
                'press': press
            })
        
        return results
    
    # 실행
    news_list = extract_news(sample_html)
    
    print("📰 추출된 뉴스 목록:")
    print("="*70)
    
    for i, news in enumerate(news_list, 1):
        print(f"\n{i}. {news['title']}")
        print(f"   언론사: {news['press']}")
        print(f"   날짜: {news['date']}")
        print(f"   링크: {news['link']}")
    
    print("\n" + "="*70)
    print("\n[설명]")
    print("이 예제는 3교시에서 배울 내용의 미리보기입니다:")
    print("1. 클래스를 사용하여 정확한 요소 찾기")
    print("2. 구조화된 데이터 추출")
    print("3. 딕셔너리로 데이터 정리")
    print("4. 실제 뉴스 사이트에서도 같은 방식으로 적용 가능")


# ============================================================================
# 메인 메뉴
# ============================================================================

def show_menu():
    """메뉴 표시"""
    print("\n" + "="*70)
    print("웹 스크레이핑 입문 - 2교시: BeautifulSoup 기본 사용법")
    print("="*70)
    print("\n[학습 섹션]")
    print("1. BeautifulSoup 기본 - HTML 파싱")
    print("2. find()와 find_all() - 태그 찾기")
    print("3. 데이터 추출 - 텍스트와 속성")
    print("4. HTML 구조 탐색 - 부모, 자식, 형제")
    print("5. 실전 예제 - 모든 링크 추출")
    print("\n[실습 및 평가]")
    print("6. 실습 미션")
    print("7. 퀴즈")
    print("8. 정답 및 해설")
    print("9. 종합 실습 (뉴스 데이터 추출)")
    print("\n[전체 실행]")
    print("0. 모든 섹션 순서대로 실행")
    print("q. 종료")
    print("="*70)


def run_all_sections():
    """모든 섹션 순서대로 실행"""
    sections = [
        section1_basic_parsing,
        section2_find_methods,
        section3_extract_data,
        section4_navigate_tree,
        section5_real_example,
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
        choice = input("\n선택하세요 (0-9, q): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 학습을 종료합니다. 수고하셨습니다!")
            break
        elif choice == '0':
            run_all_sections()
        elif choice == '1':
            section1_basic_parsing()
        elif choice == '2':
            section2_find_methods()
        elif choice == '3':
            section3_extract_data()
        elif choice == '4':
            section4_navigate_tree()
        elif choice == '5':
            section5_real_example()
        elif choice == '6':
            practice_missions()
        elif choice == '7':
            quiz()
        elif choice == '8':
            show_answers()
        elif choice == '9':
            comprehensive_example()
        else:
            print("⚠️ 올바른 번호를 입력하세요.")
        
        if choice != '0' and choice != 'q':
            input("\n⏎ Enter를 눌러 메뉴로 돌아가기...")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║         웹 스크레이핑 입문 - 2교시: BeautifulSoup 기본           ║
    ║                                                                  ║
    ║                  HTML에서 원하는 데이터 추출하기                  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("📚 학습 목표:")
    print("  1. HTML 파싱 개념 이해")
    print("  2. find()와 find_all()로 태그 찾기")
    print("  3. 텍스트와 속성 추출하기")
    print("  4. 실전: 웹페이지에서 링크 추출")
    
    main()
