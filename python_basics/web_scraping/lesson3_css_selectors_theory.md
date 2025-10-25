# 웹 스크레이핑 입문 - 3교시: CSS 선택자를 활용한 정밀 데이터 수집

> **대상**: 파이썬을 처음 배우는 직장인
> 
> **예상 학습 시간**: 1.5시간 (이론 40분 + 실습 50분)

---

## 🎯 학습 목표

이 과정을 마치면 다음을 할 수 있게 됩니다:

1. **CSS 선택자 이해**: CSS 선택자 문법을 이해하고 활용할 수 있습니다
2. **정밀한 데이터 수집**: select()로 복잡한 구조에서도 원하는 데이터를 정확히 추출할 수 있습니다
3. **실전 프로젝트**: 뉴스 기사 목록 페이지에서 모든 기사 제목을 자동 수집할 수 있습니다
4. **데이터 구조화**: 수집한 데이터를 체계적으로 정리하고 CSV 파일로 저장할 수 있습니다
5. **실무 활용**: 실제 업무에서 필요한 데이터를 자동으로 수집하는 프로그램을 만들 수 있습니다

---

## 📚 목차

1. [CSS 선택자란 무엇인가?](#1-css-선택자란-무엇인가)
2. [select()와 select_one() 사용법](#2-select와-select_one-사용법)
3. [기본 CSS 선택자](#3-기본-css-선택자)
4. [고급 CSS 선택자](#4-고급-css-선택자)
5. [실전 프로젝트: 뉴스 기사 수집](#5-실전-프로젝트-뉴스-기사-수집)
6. [데이터 저장하기 (CSV)](#6-데이터-저장하기-csv)

---

## 1. CSS 선택자란 무엇인가?

### 1.1 CSS 선택자의 개념

**CSS 선택자 = HTML 요소를 찾기 위한 패턴(규칙)**

웹 디자이너가 스타일을 적용할 요소를 찾는 방법입니다:

```css
/* CSS에서 사용 예시 */
.news-title {
    color: blue;      /* news-title 클래스를 가진 요소를 파란색으로 */
}

#main-content {
    font-size: 16px;  /* main-content id를 가진 요소의 폰트 크기 */
}
```

**웹 스크레이핑에서도 같은 방식으로 요소를 찾을 수 있습니다!**

### 1.2 find() vs select() 비교

| 구분 | find() / find_all() | select() / select_one() |
|------|---------------------|------------------------|
| **문법** | BeautifulSoup 전용 | CSS 선택자 (표준) |
| **간단한 경우** | 더 간단 | 조금 복잡 |
| **복잡한 경우** | 복잡함 | 훨씬 간단! ✨ |
| **실무 활용** | 기본 작업 | 정밀한 데이터 수집 |

**예시 비교:**

```python
# find_all() 방식
articles = soup.find_all('div', class_='article')
for article in articles:
    title = article.find('h2', class_='title')
    link = article.find('a', class_='link')

# select() 방식 (더 간결!)
titles = soup.select('div.article h2.title')
links = soup.select('div.article a.link')
```

### 1.3 크롬 개발자 도구로 선택자 확인하기

**실무 팁: 크롬에서 CSS 선택자 자동 생성!**

```
1. 크롬에서 웹페이지 열기
2. F12 (개발자 도구)
3. 원하는 요소에 우클릭 → "검사"
4. Elements 탭에서 요소 우클릭
5. Copy → Copy selector
6. 선택자가 클립보드에 복사됨! ✨
```

---

## 2. select()와 select_one() 사용법

### 2.1 기본 문법

```python
from bs4 import BeautifulSoup

html = """
<div class="news">
    <h2 class="title">첫 번째 뉴스</h2>
    <h2 class="title">두 번째 뉴스</h2>
</div>
"""

soup = BeautifulSoup(html, 'html.parser')

# select_one() - 첫 번째 하나만
first = soup.select_one('.title')
print(first.text)  # 첫 번째 뉴스

# select() - 모든 것 (리스트)
all_titles = soup.select('.title')
print(len(all_titles))  # 2
```

### 2.2 find() vs select() 대응표

| find/find_all | select/select_one | 설명 |
|---------------|-------------------|------|
| `find('div')` | `select_one('div')` | div 태그 하나 |
| `find_all('div')` | `select('div')` | 모든 div 태그 |
| `find('div', class_='news')` | `select_one('div.news')` | class가 news인 div |
| `find('div', id='main')` | `select_one('#main')` | id가 main인 div |

---

## 3. 기본 CSS 선택자

### 3.1 태그 선택자

```python
# HTML
html = "<div><p>문단1</p><p>문단2</p></div>"
soup = BeautifulSoup(html, 'html.parser')

# 모든 p 태그
paragraphs = soup.select('p')
# 결과: [<p>문단1</p>, <p>문단2</p>]
```

### 3.2 클래스 선택자 (.)

```python
# HTML
html = """
<div class="article">기사1</div>
<div class="article">기사2</div>
<div class="notice">공지</div>
"""
soup = BeautifulSoup(html, 'html.parser')

# class="article"인 모든 요소
articles = soup.select('.article')
# 결과: 2개

# 태그 + 클래스 조합
div_articles = soup.select('div.article')
# 결과: div 태그이면서 article 클래스인 것
```

### 3.3 ID 선택자 (#)

```python
# HTML
html = """
<div id="header">헤더</div>
<div id="content">콘텐츠</div>
"""
soup = BeautifulSoup(html, 'html.parser')

# id="header"인 요소
header = soup.select_one('#header')
# 결과: <div id="header">헤더</div>
```

### 3.4 속성 선택자 ([])

```python
# HTML
html = """
<a href="/news/1">뉴스1</a>
<a href="/notice/1">공지1</a>
<input type="text" name="username">
<input type="password" name="password">
"""
soup = BeautifulSoup(html, 'html.parser')

# href 속성이 있는 a 태그
links_with_href = soup.select('a[href]')

# href가 "/news"로 시작하는 링크
news_links = soup.select('a[href^="/news"]')

# type이 "text"인 input
text_inputs = soup.select('input[type="text"]')
```

**속성 선택자 패턴:**

| 패턴 | 의미 | 예시 |
|------|------|------|
| `[attr]` | 속성이 있음 | `a[href]` |
| `[attr="value"]` | 속성 값이 정확히 일치 | `input[type="text"]` |
| `[attr^="value"]` | 속성 값이 value로 시작 | `a[href^="http"]` |
| `[attr$="value"]` | 속성 값이 value로 끝남 | `img[src$=".jpg"]` |
| `[attr*="value"]` | 속성 값에 value 포함 | `a[href*="news"]` |

---

## 4. 고급 CSS 선택자

### 4.1 자손 선택자 (공백)

```python
# HTML
html = """
<div class="container">
    <div class="content">
        <h2>제목</h2>
        <p>내용</p>
    </div>
</div>
"""
soup = BeautifulSoup(html, 'html.parser')

# container 아래의 모든 h2 (자손)
h2 = soup.select('.container h2')
# 결과: <h2>제목</h2>

# container 바로 아래의 content 클래스
content = soup.select('.container .content')
```

### 4.2 자식 선택자 (>)

```python
# HTML
html = """
<div class="parent">
    <p>직접 자식</p>
    <div>
        <p>손자</p>
    </div>
</div>
"""
soup = BeautifulSoup(html, 'html.parser')

# parent의 직접 자식인 p만
direct_child = soup.select('.parent > p')
# 결과: [<p>직접 자식</p>]  (손자는 제외)

# 공백을 사용하면 모든 자손
all_descendants = soup.select('.parent p')
# 결과: [<p>직접 자식</p>, <p>손자</p>]
```

### 4.3 여러 선택자 조합 (,)

```python
# HTML
html = """
<h1>제목1</h1>
<h2>제목2</h2>
<p>문단</p>
"""
soup = BeautifulSoup(html, 'html.parser')

# h1 또는 h2
headings = soup.select('h1, h2')
# 결과: [<h1>제목1</h1>, <h2>제목2</h2>]
```

### 4.4 형제 선택자 (+, ~)

```python
# HTML
html = """
<h2>제목</h2>
<p>문단1</p>
<p>문단2</p>
"""
soup = BeautifulSoup(html, 'html.parser')

# h2 바로 다음의 p
next_p = soup.select('h2 + p')
# 결과: [<p>문단1</p>]

# h2 이후의 모든 p
all_next_p = soup.select('h2 ~ p')
# 결과: [<p>문단1</p>, <p>문단2</p>]
```

### 4.5 nth-child (순서로 선택)

```python
# HTML
html = """
<ul>
    <li>항목1</li>
    <li>항목2</li>
    <li>항목3</li>
    <li>항목4</li>
</ul>
"""
soup = BeautifulSoup(html, 'html.parser')

# 첫 번째 li
first = soup.select('li:nth-child(1)')

# 두 번째 li
second = soup.select('li:nth-child(2)')

# 홀수 번째 li
odd = soup.select('li:nth-child(odd)')

# 짝수 번째 li
even = soup.select('li:nth-child(even)')

# 첫 번째
first = soup.select('li:first-child')

# 마지막
last = soup.select('li:last-child')
```

---

## 5. 실전 프로젝트: 뉴스 기사 수집

### 5.1 프로젝트 개요

**목표**: 뉴스 목록 페이지에서 모든 기사의 제목, 링크, 날짜, 언론사를 수집

**HTML 구조 예시** (대부분의 뉴스 사이트와 유사):

```html
<div class="news-list">
    <article class="news-item">
        <a href="/news/1" class="news-link">
            <h2 class="news-title">파이썬 자동화 성공 사례</h2>
        </a>
        <div class="news-meta">
            <span class="press">테크뉴스</span>
            <span class="date">2024.01.15</span>
        </div>
    </article>
    <!-- 더 많은 기사... -->
</div>
```

### 5.2 단계별 데이터 수집

**Step 1: 모든 기사 항목 찾기**

```python
# 모든 news-item 찾기
articles = soup.select('article.news-item')
print(f"총 {len(articles)}개의 기사")
```

**Step 2: 각 기사에서 제목 추출**

```python
for article in articles:
    # 제목 (news-item 안의 h2.news-title)
    title = article.select_one('h2.news-title')
    print(title.text.strip())
```

**Step 3: 링크 추출**

```python
for article in articles:
    # 링크 (news-item 안의 a.news-link)
    link = article.select_one('a.news-link')
    url = link.get('href', '')
    print(url)
```

**Step 4: 메타 정보 추출**

```python
for article in articles:
    # 언론사
    press = article.select_one('.press')
    press_name = press.text.strip() if press else '언론사 없음'
    
    # 날짜
    date = article.select_one('.date')
    date_str = date.text.strip() if date else '날짜 없음'
    
    print(f"{press_name} | {date_str}")
```

**Step 5: 완전한 코드**

```python
import requests
from bs4 import BeautifulSoup

def scrape_news(url):
    """뉴스 목록 페이지에서 기사 정보 수집"""
    try:
        # 1. HTML 가져오기
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 2. 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. 모든 기사 항목 찾기
        articles = soup.select('article.news-item')
        
        # 4. 데이터 추출
        results = []
        for article in articles:
            # 제목
            title_tag = article.select_one('h2.news-title')
            title = title_tag.text.strip() if title_tag else '제목 없음'
            
            # 링크
            link_tag = article.select_one('a.news-link')
            link = link_tag.get('href', '') if link_tag else ''
            
            # 언론사
            press_tag = article.select_one('.press')
            press = press_tag.text.strip() if press_tag else '언론사 없음'
            
            # 날짜
            date_tag = article.select_one('.date')
            date = date_tag.text.strip() if date_tag else '날짜 없음'
            
            results.append({
                'title': title,
                'link': link,
                'press': press,
                'date': date
            })
        
        return results
        
    except Exception as e:
        print(f"오류: {e}")
        return []

# 사용 예시
news_list = scrape_news('https://news.example.com')

for i, news in enumerate(news_list, 1):
    print(f"\n{i}. {news['title']}")
    print(f"   언론사: {news['press']}")
    print(f"   날짜: {news['date']}")
    print(f"   링크: {news['link']}")
```

### 5.3 실무에서 자주 쓰는 CSS 선택자 패턴

```python
# 패턴 1: 클래스 체인
soup.select('div.article h2.title a.link')
# div.article 안의 h2.title 안의 a.link

# 패턴 2: 직접 자식만
soup.select('div.container > div.content')
# container의 직접 자식인 content만

# 패턴 3: 속성 필터링
soup.select('a[href^="http"]')
# http로 시작하는 링크만

# 패턴 4: 복수 클래스
soup.select('.news.featured')
# news와 featured 클래스를 모두 가진 요소

# 패턴 5: 여러 조건 OR
soup.select('.article, .post, .news')
# article 또는 post 또는 news 클래스
```

---

## 6. 데이터 저장하기 (CSV)

### 6.1 CSV 파일로 저장

```python
import csv
from datetime import datetime

def save_to_csv(data, filename=None):
    """수집한 데이터를 CSV 파일로 저장"""
    if not filename:
        # 파일명: news_20240115_143022.csv
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'news_{timestamp}.csv'
    
    # CSV 파일 쓰기
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        if not data:
            print("저장할 데이터가 없습니다.")
            return
        
        # 헤더 (딕셔너리의 키)
        fieldnames = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 헤더 쓰기
        writer.writeheader()
        
        # 데이터 쓰기
        writer.writerows(data)
    
    print(f"✅ {len(data)}개의 데이터를 {filename}에 저장했습니다.")

# 사용 예시
news_list = scrape_news('https://news.example.com')
save_to_csv(news_list, 'news_data.csv')
```

### 6.2 pandas로 저장 (선택사항)

```python
import pandas as pd

def save_to_csv_pandas(data, filename='news_data.csv'):
    """pandas로 CSV 저장"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ {len(data)}개의 데이터를 {filename}에 저장했습니다.")

# 사용 예시
news_list = scrape_news('https://news.example.com')
save_to_csv_pandas(news_list)
```

---

## 💡 핵심 요약

### 배운 내용 정리

1. **CSS 선택자 = 정밀한 요소 찾기**
   - HTML 요소를 찾는 강력한 패턴

2. **기본 선택자**
   ```python
   '.class'     # 클래스
   '#id'        # ID
   'tag'        # 태그
   '[attr]'     # 속성
   ```

3. **조합 선택자**
   ```python
   'div.news'           # div이면서 news 클래스
   '.container h2'      # container 아래의 모든 h2
   '.container > h2'    # container의 직접 자식 h2
   'h1, h2'            # h1 또는 h2
   ```

4. **실전 패턴**
   ```python
   # 1. HTML 가져오기
   response = requests.get(url)
   
   # 2. 파싱
   soup = BeautifulSoup(response.text, 'html.parser')
   
   # 3. CSS 선택자로 데이터 추출
   items = soup.select('div.item')
   for item in items:
       title = item.select_one('h2.title').text
       link = item.select_one('a').get('href')
   
   # 4. CSV 저장
   save_to_csv(data)
   ```

---

## 🎓 실무 활용 시나리오

### 시나리오 1: 경쟁사 가격 모니터링

```python
def monitor_competitor_prices(url):
    """경쟁사 제품 가격 모니터링"""
    soup = BeautifulSoup(get_html(url), 'html.parser')
    
    products = soup.select('div.product-item')
    
    for product in products:
        name = product.select_one('h3.product-name').text.strip()
        price = product.select_one('span.price').text.strip()
        
        # 가격 변동 체크, DB 저장, 알림 등
        print(f"{name}: {price}")
```

### 시나리오 2: 부동산 매물 수집

```python
def scrape_real_estate(url):
    """부동산 매물 정보 수집"""
    soup = BeautifulSoup(get_html(url), 'html.parser')
    
    listings = soup.select('div.listing-item')
    
    for listing in listings:
        address = listing.select_one('.address').text.strip()
        price = listing.select_one('.price').text.strip()
        area = listing.select_one('.area').text.strip()
        
        # CSV 저장, 분석 등
        print(f"{address} | {price} | {area}")
```

### 시나리오 3: 채용 공고 수집

```python
def scrape_job_postings(url):
    """채용 공고 수집"""
    soup = BeautifulSoup(get_html(url), 'html.parser')
    
    jobs = soup.select('div.job-posting')
    
    for job in jobs:
        title = job.select_one('h2.job-title').text.strip()
        company = job.select_one('span.company').text.strip()
        location = job.select_one('span.location').text.strip()
        
        print(f"{title} | {company} | {location}")
```

---

## 🔍 CSS 선택자 치트 시트

```python
# 기본
'.class'              # 클래스
'#id'                 # ID
'tag'                 # 태그

# 조합
'div.class'           # div이면서 class
'div#id'              # div이면서 id

# 자손/자식
'div p'               # div 안의 모든 p (자손)
'div > p'             # div의 직접 자식 p

# 형제
'h2 + p'              # h2 바로 다음의 p
'h2 ~ p'              # h2 이후의 모든 p

# 속성
'[href]'              # href 속성이 있는 요소
'[href="/news"]'      # href가 정확히 일치
'[href^="http"]'      # href가 http로 시작
'[href$=".pdf"]'      # href가 .pdf로 끝남
'[href*="news"]'      # href에 news 포함

# 순서
':first-child'        # 첫 번째 자식
':last-child'         # 마지막 자식
':nth-child(2)'       # 두 번째 자식
':nth-child(odd)'     # 홀수 번째
':nth-child(even)'    # 짝수 번째

# 여러 개
'h1, h2, h3'          # h1 또는 h2 또는 h3
```

---

## 🎉 축하합니다!

웹 스크레이핑 3교시를 모두 완료하셨습니다!

### 배운 것들
✅ 1교시: Requests로 HTML 가져오기  
✅ 2교시: BeautifulSoup로 파싱하기  
✅ 3교시: CSS 선택자로 정밀하게 데이터 수집하기

### 이제 할 수 있는 것들
- 웹페이지에서 자동으로 데이터 수집
- 가격, 뉴스, 제품 정보 등 모니터링
- 수집한 데이터를 CSV로 저장
- 실무에서 바로 활용 가능한 스크립트 작성

---

**작성일**: 2024년  
**난이도**: 초급-중급 (Beginner to Intermediate)  
**예상 학습 시간**: 1.5시간
