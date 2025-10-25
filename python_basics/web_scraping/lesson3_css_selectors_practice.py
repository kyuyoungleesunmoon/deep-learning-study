"""
웹 스크레이핑 입문 - 3교시: CSS 선택자를 활용한 정밀 데이터 수집 실습

대상: 파이썬을 처음 배우는 직장인
학습 시간: 50분
"""

import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime


# ============================================================================
# 섹션 1: CSS 선택자 기초
# ============================================================================

def section1_css_basics():
    """섹션 1: CSS 선택자 기초"""
    print("\n" + "="*70)
    print("섹션 1: CSS 선택자 기초")
    print("="*70)
    
    print("\n[이론]")
    print("CSS 선택자는 HTML 요소를 찾기 위한 패턴입니다.")
    print("웹 디자이너가 스타일을 적용하는 방식과 동일합니다.")
    
    html = """
    <div class="container">
        <h1 id="main-title">메인 제목</h1>
        <div class="article">
            <h2>첫 번째 기사</h2>
            <p>내용 1</p>
        </div>
        <div class="article">
            <h2>두 번째 기사</h2>
            <p>내용 2</p>
        </div>
        <div class="notice">
            <h2>공지사항</h2>
        </div>
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    
    print("\n[예제 1-1] 태그 선택자")
    print("-" * 50)
    
    # 모든 h2 태그
    h2_tags = soup.select('h2')
    print(f"모든 h2 태그 ({len(h2_tags)}개):")
    for i, h2 in enumerate(h2_tags, 1):
        print(f"{i}. {h2.text}")
    
    print("\n[예제 1-2] 클래스 선택자 (.)")
    print("-" * 50)
    
    # .article 클래스
    articles = soup.select('.article')
    print(f"article 클래스 ({len(articles)}개):")
    for i, article in enumerate(articles, 1):
        h2 = article.select_one('h2')
        print(f"{i}. {h2.text}")
    
    # div.article (div이면서 article 클래스)
    div_articles = soup.select('div.article')
    print(f"\ndiv.article ({len(div_articles)}개):")
    for article in div_articles:
        print(f"- {article.select_one('h2').text}")
    
    print("\n[예제 1-3] ID 선택자 (#)")
    print("-" * 50)
    
    # #main-title ID
    title = soup.select_one('#main-title')
    print(f"ID 'main-title': {title.text}")
    
    print("\n[설명]")
    print("1. 'tag' - 태그 이름으로 찾기")
    print("2. '.class' - 클래스로 찾기")
    print("3. '#id' - ID로 찾기")
    print("4. 'tag.class' - 태그와 클래스 조합")


# ============================================================================
# 섹션 2: 속성 선택자와 조합
# ============================================================================

def section2_advanced_selectors():
    """섹션 2: 고급 선택자"""
    print("\n" + "="*70)
    print("섹션 2: 속성 선택자와 조합")
    print("="*70)
    
    print("\n[이론]")
    print("속성 선택자로 더 정밀하게 요소를 찾을 수 있습니다.")
    
    html = """
    <div class="links">
        <a href="/news/1">뉴스 1</a>
        <a href="/news/2">뉴스 2</a>
        <a href="/notice/1">공지 1</a>
        <a href="https://external.com">외부 링크</a>
        <img src="image1.jpg" alt="이미지1">
        <img src="image2.png" alt="이미지2">
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    
    print("\n[예제 2-1] 속성 존재 확인 [attr]")
    print("-" * 50)
    
    # href 속성이 있는 a 태그
    links = soup.select('a[href]')
    print(f"href 속성이 있는 링크: {len(links)}개")
    
    print("\n[예제 2-2] 속성 값으로 필터링")
    print("-" * 50)
    
    # href가 "/news"로 시작하는 링크
    news_links = soup.select('a[href^="/news"]')
    print(f"뉴스 링크 ({len(news_links)}개):")
    for link in news_links:
        print(f"- {link.text}: {link['href']}")
    
    # href가 "http"로 시작하는 링크 (외부 링크)
    external = soup.select('a[href^="http"]')
    print(f"\n외부 링크 ({len(external)}개):")
    for link in external:
        print(f"- {link.text}: {link['href']}")
    
    # src가 ".jpg"로 끝나는 이미지
    jpg_images = soup.select('img[src$=".jpg"]')
    print(f"\nJPG 이미지 ({len(jpg_images)}개):")
    for img in jpg_images:
        print(f"- {img['alt']}: {img['src']}")
    
    print("\n[예제 2-3] 속성에 특정 값 포함")
    print("-" * 50)
    
    # href에 "news"가 포함된 링크
    contains_news = soup.select('a[href*="news"]')
    print(f"'news' 포함 링크: {len(contains_news)}개")
    
    print("\n[설명]")
    print("1. [attr] - 속성이 있는 요소")
    print("2. [attr='value'] - 속성 값이 정확히 일치")
    print("3. [attr^='value'] - 속성 값이 value로 시작")
    print("4. [attr$='value'] - 속성 값이 value로 끝남")
    print("5. [attr*='value'] - 속성 값에 value 포함")


# ============================================================================
# 섹션 3: 자손/자식 선택자
# ============================================================================

def section3_descendant_child():
    """섹션 3: 자손과 자식 선택자"""
    print("\n" + "="*70)
    print("섹션 3: 자손/자식 선택자")
    print("="*70)
    
    print("\n[이론]")
    print("공백( ) = 자손 (모든 하위 요소)")
    print(">      = 자식 (직접 하위 요소만)")
    
    html = """
    <div class="container">
        <p>직접 자식 p</p>
        <div class="content">
            <p>손자 p</p>
            <div>
                <p>증손자 p</p>
            </div>
        </div>
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    
    print("\n[예제 3-1] 자손 선택자 (공백)")
    print("-" * 50)
    
    # container 아래의 모든 p (자손)
    all_p = soup.select('.container p')
    print(f"container 아래 모든 p: {len(all_p)}개")
    for i, p in enumerate(all_p, 1):
        print(f"{i}. {p.text}")
    
    print("\n[예제 3-2] 자식 선택자 (>)")
    print("-" * 50)
    
    # container의 직접 자식 p만
    direct_p = soup.select('.container > p')
    print(f"container의 직접 자식 p: {len(direct_p)}개")
    for p in direct_p:
        print(f"- {p.text}")
    
    # content의 직접 자식 p
    content_p = soup.select('.content > p')
    print(f"\ncontent의 직접 자식 p: {len(content_p)}개")
    for p in content_p:
        print(f"- {p.text}")
    
    print("\n[설명]")
    print("1. 'A B' - A 아래의 모든 B (깊이 무관)")
    print("2. 'A > B' - A의 직접 자식 B만")
    print("3. 정밀한 선택이 필요할 때 > 사용")


# ============================================================================
# 섹션 4: 여러 선택자 조합
# ============================================================================

def section4_multiple_selectors():
    """섹션 4: 여러 선택자 조합"""
    print("\n" + "="*70)
    print("섹션 4: 여러 선택자 조합")
    print("="*70)
    
    print("\n[이론]")
    print("쉼표(,)로 여러 선택자를 OR 조건으로 조합할 수 있습니다.")
    
    html = """
    <div>
        <h1>제목 1</h1>
        <h2>제목 2</h2>
        <h3>제목 3</h3>
        <p class="important">중요 문단</p>
        <p class="normal">일반 문단</p>
        <span class="important">중요 스팬</span>
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    
    print("\n[예제 4-1] 여러 태그 선택 (,)")
    print("-" * 50)
    
    # h1, h2, h3 모두 선택
    headings = soup.select('h1, h2, h3')
    print(f"모든 헤딩 태그: {len(headings)}개")
    for h in headings:
        print(f"- {h.name}: {h.text}")
    
    print("\n[예제 4-2] 여러 클래스 선택")
    print("-" * 50)
    
    # important 클래스 (태그 무관)
    important = soup.select('.important')
    print(f"important 클래스: {len(important)}개")
    for elem in important:
        print(f"- {elem.name}: {elem.text}")
    
    print("\n[예제 4-3] 복잡한 조합")
    print("-" * 50)
    
    # h2 또는 p.important
    combined = soup.select('h2, p.important')
    print(f"h2 또는 p.important: {len(combined)}개")
    for elem in combined:
        print(f"- {elem.name}: {elem.text}")
    
    print("\n[설명]")
    print("1. 'A, B' - A 또는 B")
    print("2. 여러 조건을 동시에 만족하는 요소를 찾을 때 유용")
    print("3. 쉼표로 구분하여 무제한 조합 가능")


# ============================================================================
# 섹션 5: 실전 프로젝트 - 뉴스 기사 수집
# ============================================================================

def section5_news_scraping():
    """섹션 5: 실전 - 뉴스 기사 수집"""
    print("\n" + "="*70)
    print("섹션 5: 실전 프로젝트 - 뉴스 기사 수집")
    print("="*70)
    
    print("\n[이론]")
    print("실제 뉴스 사이트 구조를 모방한 HTML에서 데이터를 수집합니다.")
    
    # 실제 뉴스 사이트와 유사한 구조
    sample_html = """
    <html>
    <body>
        <div class="news-list">
            <article class="news-item">
                <a href="/news/1" class="news-link">
                    <h2 class="news-title">파이썬으로 업무 자동화 성공</h2>
                </a>
                <div class="news-meta">
                    <span class="press">테크뉴스</span>
                    <span class="date">2024-01-15</span>
                    <span class="category">기술</span>
                </div>
                <p class="news-summary">
                    파이썬을 활용한 업무 자동화 사례가 증가하고 있습니다.
                </p>
            </article>
            
            <article class="news-item">
                <a href="/news/2" class="news-link">
                    <h2 class="news-title">웹 스크레이핑으로 데이터 수집</h2>
                </a>
                <div class="news-meta">
                    <span class="press">데이터저널</span>
                    <span class="date">2024-01-14</span>
                    <span class="category">데이터</span>
                </div>
                <p class="news-summary">
                    웹 스크레이핑 기술이 데이터 수집의 핵심으로 떠오르고 있습니다.
                </p>
            </article>
            
            <article class="news-item">
                <a href="/news/3" class="news-link">
                    <h2 class="news-title">BeautifulSoup 완벽 가이드</h2>
                </a>
                <div class="news-meta">
                    <span class="press">코딩매거진</span>
                    <span class="date">2024-01-13</span>
                    <span class="category">튜토리얼</span>
                </div>
                <p class="news-summary">
                    BeautifulSoup 라이브러리의 모든 기능을 상세히 알아봅니다.
                </p>
            </article>
        </div>
    </body>
    </html>
    """
    
    soup = BeautifulSoup(sample_html, 'html.parser')
    
    print("\n[예제 5-1] 단계별 데이터 추출")
    print("-" * 50)
    
    # Step 1: 모든 기사 찾기
    articles = soup.select('article.news-item')
    print(f"✅ Step 1: 총 {len(articles)}개의 기사를 찾았습니다.\n")
    
    # Step 2: 각 기사에서 정보 추출
    news_list = []
    
    for i, article in enumerate(articles, 1):
        print(f"[기사 {i}]")
        
        # 제목
        title_tag = article.select_one('h2.news-title')
        title = title_tag.text.strip() if title_tag else '제목 없음'
        print(f"제목: {title}")
        
        # 링크
        link_tag = article.select_one('a.news-link')
        link = link_tag.get('href', '') if link_tag else ''
        print(f"링크: {link}")
        
        # 언론사
        press_tag = article.select_one('.press')
        press = press_tag.text.strip() if press_tag else '언론사 없음'
        print(f"언론사: {press}")
        
        # 날짜
        date_tag = article.select_one('.date')
        date = date_tag.text.strip() if date_tag else '날짜 없음'
        print(f"날짜: {date}")
        
        # 카테고리
        category_tag = article.select_one('.category')
        category = category_tag.text.strip() if category_tag else '카테고리 없음'
        print(f"카테고리: {category}")
        
        # 요약
        summary_tag = article.select_one('.news-summary')
        summary = summary_tag.text.strip() if summary_tag else '요약 없음'
        print(f"요약: {summary[:50]}...")
        
        print()
        
        # 딕셔너리로 저장
        news_list.append({
            'title': title,
            'link': link,
            'press': press,
            'date': date,
            'category': category,
            'summary': summary
        })
    
    print(f"✅ 총 {len(news_list)}개의 기사 정보를 추출했습니다!")
    
    print("\n[설명]")
    print("1. article.news-item으로 각 기사 항목을 찾습니다")
    print("2. select_one()으로 각 기사 안에서 세부 정보를 추출합니다")
    print("3. 딕셔너리로 구조화하여 저장합니다")
    print("4. 이 패턴은 대부분의 목록형 페이지에 적용 가능합니다")
    
    return news_list


# ============================================================================
# 섹션 6: CSV 파일로 저장하기
# ============================================================================

def section6_save_csv():
    """섹션 6: 수집한 데이터를 CSV로 저장"""
    print("\n" + "="*70)
    print("섹션 6: CSV 파일로 저장하기")
    print("="*70)
    
    print("\n[이론]")
    print("수집한 데이터를 CSV 파일로 저장하면:")
    print("- 엑셀에서 열어볼 수 있습니다")
    print("- 데이터 분석에 활용할 수 있습니다")
    print("- 다른 프로그램과 공유할 수 있습니다")
    
    # 샘플 데이터
    sample_data = [
        {
            'title': '파이썬 자동화 성공',
            'press': '테크뉴스',
            'date': '2024-01-15',
            'link': '/news/1'
        },
        {
            'title': '웹 스크레이핑 가이드',
            'press': '데이터저널',
            'date': '2024-01-14',
            'link': '/news/2'
        },
        {
            'title': 'BeautifulSoup 활용',
            'press': '코딩매거진',
            'date': '2024-01-13',
            'link': '/news/3'
        }
    ]
    
    print("\n[예제 6-1] CSV 파일 저장 함수")
    print("-" * 50)
    
    def save_to_csv(data, filename=None):
        """데이터를 CSV 파일로 저장"""
        if not data:
            print("⚠️ 저장할 데이터가 없습니다.")
            return False
        
        # 파일명 생성 (타임스탬프 포함)
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'news_{timestamp}.csv'
        
        try:
            # CSV 파일 쓰기
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                # 헤더 (첫 번째 데이터의 키)
                fieldnames = data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 헤더 쓰기
                writer.writeheader()
                
                # 데이터 쓰기
                writer.writerows(data)
            
            print(f"✅ {len(data)}개의 데이터를 '{filename}'에 저장했습니다!")
            return True
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return False
    
    # 실행
    result = save_to_csv(sample_data, 'sample_news.csv')
    
    if result:
        print("\n📄 저장된 파일:")
        print("- 파일명: sample_news.csv")
        print("- 인코딩: UTF-8 (BOM) - 엑셀에서 한글 깨짐 방지")
        print("- 형식: CSV (쉼표로 구분)")
    
    print("\n[예제 6-2] 저장된 파일 미리보기")
    print("-" * 50)
    
    # 파일 읽어서 확인
    try:
        with open('sample_news.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            print("CSV 파일 내용:")
            for i, row in enumerate(reader, 1):
                print(f"\n{i}. {row['title']}")
                print(f"   언론사: {row['press']}")
                print(f"   날짜: {row['date']}")
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    
    print("\n[설명]")
    print("1. csv.DictWriter로 딕셔너리 데이터를 CSV로 저장합니다")
    print("2. encoding='utf-8-sig'로 엑셀에서 한글이 깨지지 않게 합니다")
    print("3. with문으로 파일을 자동으로 닫습니다")
    print("4. 타임스탬프를 포함한 파일명으로 구분합니다")


# ============================================================================
# 섹션 7: 종합 실습 - 완전한 스크레이핑 프로그램
# ============================================================================

def section7_complete_example():
    """섹션 7: 완전한 웹 스크레이핑 프로그램"""
    print("\n" + "="*70)
    print("섹션 7: 종합 실습 - 완전한 스크레이핑 프로그램")
    print("="*70)
    
    print("\n[이론]")
    print("지금까지 배운 모든 것을 통합한 실무 수준의 프로그램입니다.")
    
    def scrape_and_save_news(html_content=None):
        """
        뉴스 스크레이핑 + CSV 저장 통합 함수
        
        Args:
            html_content: HTML 문자열 (테스트용, 실제로는 requests로 가져옴)
        
        Returns:
            bool: 성공 여부
        """
        print("\n🚀 뉴스 스크레이핑 시작...")
        
        # 샘플 HTML (실제로는 requests.get()으로 가져옴)
        if not html_content:
            html_content = """
            <div class="news-list">
                <article class="news-item">
                    <a href="/news/1" class="news-link">
                        <h2 class="news-title">AI 기술 발전</h2>
                    </a>
                    <div class="news-meta">
                        <span class="press">AI뉴스</span>
                        <span class="date">2024-01-15</span>
                    </div>
                </article>
                <article class="news-item">
                    <a href="/news/2" class="news-link">
                        <h2 class="news-title">데이터 과학 트렌드</h2>
                    </a>
                    <div class="news-meta">
                        <span class="press">데이터타임즈</span>
                        <span class="date">2024-01-14</span>
                    </div>
                </article>
            </div>
            """
        
        try:
            # 1. 파싱
            print("📖 HTML 파싱 중...")
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 2. 데이터 추출
            print("🔍 데이터 추출 중...")
            articles = soup.select('article.news-item')
            
            if not articles:
                print("❌ 기사를 찾을 수 없습니다.")
                return False
            
            print(f"✅ {len(articles)}개의 기사를 찾았습니다.")
            
            news_list = []
            for article in articles:
                title_tag = article.select_one('h2.news-title')
                link_tag = article.select_one('a.news-link')
                press_tag = article.select_one('.press')
                date_tag = article.select_one('.date')
                
                news_list.append({
                    'title': title_tag.text.strip() if title_tag else '',
                    'link': link_tag.get('href', '') if link_tag else '',
                    'press': press_tag.text.strip() if press_tag else '',
                    'date': date_tag.text.strip() if date_tag else ''
                })
            
            # 3. CSV 저장
            print("💾 CSV 파일로 저장 중...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'news_{timestamp}.csv'
            
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                fieldnames = ['title', 'press', 'date', 'link']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(news_list)
            
            # 4. 결과 출력
            print(f"✅ 완료! {filename}에 저장되었습니다.\n")
            
            print("📊 수집된 데이터:")
            print("="*70)
            for i, news in enumerate(news_list, 1):
                print(f"{i}. {news['title']}")
                print(f"   언론사: {news['press']} | 날짜: {news['date']}")
                print(f"   링크: {news['link']}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return False
    
    # 실행
    scrape_and_save_news()
    
    print("\n[설명]")
    print("이 함수는 실무에서 바로 사용할 수 있는 완전한 형태입니다:")
    print("1. HTML 파싱 (BeautifulSoup)")
    print("2. CSS 선택자로 데이터 추출 (select)")
    print("3. 데이터 구조화 (딕셔너리 리스트)")
    print("4. CSV 파일로 저장")
    print("5. 예외 처리 및 로깅")


# ============================================================================
# 실습 미션
# ============================================================================

def practice_missions():
    """실습 미션 - 직접 해보기"""
    print("\n" + "="*70)
    print("💪 실습 미션")
    print("="*70)
    
    print("\n[미션 1] 특정 조건의 링크만 추출하기")
    print("-" * 50)
    print("다음 HTML에서 '/news'로 시작하는 링크만 추출하세요.")
    print()
    print("HTML:")
    print("""
    <div>
        <a href="/news/1">뉴스 1</a>
        <a href="/notice/1">공지 1</a>
        <a href="/news/2">뉴스 2</a>
        <a href="https://external.com">외부</a>
    </div>
    """)
    print("\n힌트: select('a[href^=\"/news\"]')")
    
    print("\n[미션 2] 복잡한 구조에서 데이터 추출")
    print("-" * 50)
    print("다음 HTML에서 각 제품의 이름과 가격을 추출하여")
    print("딕셔너리 리스트로 만드세요.")
    print()
    print("HTML:")
    print("""
    <div class="products">
        <div class="product">
            <h3 class="name">상품 A</h3>
            <span class="price">10,000원</span>
        </div>
        <div class="product">
            <h3 class="name">상품 B</h3>
            <span class="price">20,000원</span>
        </div>
    </div>
    """)
    print("\n힌트: select('div.product'), select_one('.name'), select_one('.price')")


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
            "question": "1. class='news'인 모든 div를 선택하는 CSS 선택자는?",
            "options": ["a) div.news", "b) div#news", "c) div[news]", "d) .news.div"],
            "answer": "a"
        },
        {
            "question": "2. href가 'http'로 시작하는 링크를 찾는 선택자는?",
            "options": ["a) a[href='http']", "b) a[href^='http']", "c) a[href*='http']", "d) a[href$='http']"],
            "answer": "b"
        },
        {
            "question": "3. 자손 선택자(공백)와 자식 선택자(>)의 차이는?",
            "options": [
                "a) 없다, 같은 의미이다",
                "b) 공백은 모든 하위 요소, >는 직접 자식만",
                "c) >는 모든 하위 요소, 공백은 직접 자식만",
                "d) 둘 다 사용하지 않는다"
            ],
            "answer": "b"
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

1. 정답: a) div.news
   해설: 'tag.class' 형식으로 태그와 클래스를 조합합니다.
         div.news는 "div 태그이면서 news 클래스를 가진 요소"를 의미합니다.
         div#news는 ID를 의미하고, div[news]는 news 속성을 의미합니다.

2. 정답: b) a[href^='http']
   해설: ^는 "~로 시작"을 의미하는 속성 선택자입니다.
         = 는 정확히 일치, * 는 포함, $ 는 끝남을 의미합니다.
         외부 링크(http로 시작)와 내부 링크(/로 시작)를 구분할 때 유용합니다.

3. 정답: b) 공백은 모든 하위 요소, >는 직접 자식만
   해설: 'A B'는 A의 모든 자손 B를 찾습니다 (깊이 무관).
         'A > B'는 A의 직접 자식 B만 찾습니다 (바로 아래 1단계만).
         정밀한 선택이 필요할 때 >를 사용합니다.

</details>
    """)


# ============================================================================
# 메인 메뉴
# ============================================================================

def show_menu():
    """메뉴 표시"""
    print("\n" + "="*70)
    print("웹 스크레이핑 입문 - 3교시: CSS 선택자 활용")
    print("="*70)
    print("\n[학습 섹션]")
    print("1. CSS 선택자 기초 (태그, 클래스, ID)")
    print("2. 속성 선택자와 조합")
    print("3. 자손/자식 선택자")
    print("4. 여러 선택자 조합")
    print("5. 실전 프로젝트 - 뉴스 기사 수집")
    print("6. CSV 파일로 저장하기")
    print("7. 종합 실습 - 완전한 프로그램")
    print("\n[실습 및 평가]")
    print("8. 실습 미션")
    print("9. 퀴즈")
    print("10. 정답 및 해설")
    print("\n[전체 실행]")
    print("0. 모든 섹션 순서대로 실행")
    print("q. 종료")
    print("="*70)


def run_all_sections():
    """모든 섹션 순서대로 실행"""
    sections = [
        section1_css_basics,
        section2_advanced_selectors,
        section3_descendant_child,
        section4_multiple_selectors,
        section5_news_scraping,
        section6_save_csv,
        section7_complete_example,
        practice_missions,
        quiz,
        show_answers
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
            print("\n🎉 축하합니다! 웹 스크레이핑 3교시를 모두 완료하셨습니다!")
            print("이제 실무에서 웹 스크레이핑을 활용할 수 있습니다.")
            print("👋 학습을 종료합니다. 수고하셨습니다!")
            break
        elif choice == '0':
            run_all_sections()
        elif choice == '1':
            section1_css_basics()
        elif choice == '2':
            section2_advanced_selectors()
        elif choice == '3':
            section3_descendant_child()
        elif choice == '4':
            section4_multiple_selectors()
        elif choice == '5':
            section5_news_scraping()
        elif choice == '6':
            section6_save_csv()
        elif choice == '7':
            section7_complete_example()
        elif choice == '8':
            practice_missions()
        elif choice == '9':
            quiz()
        elif choice == '10':
            show_answers()
        else:
            print("⚠️ 올바른 번호를 입력하세요.")
        
        if choice != '0' and choice != 'q':
            input("\n⏎ Enter를 눌러 메뉴로 돌아가기...")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║      웹 스크레이핑 입문 - 3교시: CSS 선택자 정밀 데이터 수집      ║
    ║                                                                  ║
    ║               뉴스 기사 자동 수집 + CSV 저장                      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("📚 학습 목표:")
    print("  1. CSS 선택자 마스터하기")
    print("  2. 복잡한 HTML 구조에서 정확한 데이터 추출")
    print("  3. 실전 프로젝트: 뉴스 기사 자동 수집")
    print("  4. CSV 파일로 데이터 저장")
    
    main()
