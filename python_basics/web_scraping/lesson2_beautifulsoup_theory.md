# 웹 스크레이핑 입문 - 2교시: BeautifulSoup 기본 사용법

> **대상**: 파이썬을 처음 배우는 직장인
> 
> **예상 학습 시간**: 1시간 (이론 30분 + 실습 30분)

---

## 🎯 학습 목표

이 과정을 마치면 다음을 할 수 있게 됩니다:

1. **HTML 파싱 개념 이해**: HTML을 프로그램이 읽을 수 있는 형태로 변환하는 방법을 이해합니다
2. **BeautifulSoup 사용**: Python으로 HTML에서 원하는 데이터를 쉽게 추출할 수 있습니다
3. **태그 찾기**: find()와 find_all()로 원하는 HTML 태그를 찾을 수 있습니다
4. **데이터 추출**: 태그의 텍스트, 속성(href, src 등)을 추출할 수 있습니다
5. **실무 활용**: 웹페이지에서 제목, 링크, 이미지 등을 자동으로 추출할 수 있습니다

---

## 📚 목차

1. [HTML 파싱이란 무엇인가?](#1-html-파싱이란-무엇인가)
2. [BeautifulSoup 소개 및 설치](#2-beautifulsoup-소개-및-설치)
3. [기본 사용법: HTML 파싱하기](#3-기본-사용법-html-파싱하기)
4. [태그 찾기: find()와 find_all()](#4-태그-찾기-find와-find_all)
5. [데이터 추출: 텍스트와 속성](#5-데이터-추출-텍스트와-속성)
6. [HTML 구조 탐색하기](#6-html-구조-탐색하기)

---

## 1. HTML 파싱이란 무엇인가?

### 1.1 파싱(Parsing)의 개념

**파싱 = HTML을 구조적으로 해석해서 원하는 부분만 뽑아내는 과정**

일상생활에서 비유하면:
- 📰 **신문 기사**: 신문 전체 → 제목, 본문, 날짜, 기자명으로 분리
- 📋 **이력서**: 이력서 전체 → 이름, 학력, 경력, 자격증으로 분리
- 🍽️ **레시피**: 레시피 전체 → 재료, 조리법, 조리시간으로 분리

**프로그래밍에서:**
```
HTML 문자열 (덩어리)
    ↓ 파싱 (Parsing)
구조화된 데이터 (나무 구조)
    ↓ 탐색/추출
원하는 정보만!
```

### 1.2 왜 파싱이 필요한가?

**1교시에서 배운 Requests만으로는 부족합니다:**

```python
import requests

response = requests.get('http://example.com')
html = response.text

print(html)  # HTML 전체가 하나의 긴 문자열...
# 어디서 제목을 찾지? 어떻게 링크만 뽑지? 🤔
```

**문제점:**
- HTML이 하나의 긴 문자열이라 처리하기 어려움
- 원하는 부분을 찾으려면 복잡한 문자열 처리 필요
- 태그 구조를 고려해야 함

**해결책 = BeautifulSoup! 🍲**

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, 'html.parser')
title = soup.find('h1').text  # 제목만 간단히 추출! ✨
```

### 1.3 파싱 프로세스

```
1. HTML 문자열 받기 (Requests)
   └─ response.text

2. 파싱 (BeautifulSoup)
   └─ soup = BeautifulSoup(html, 'html.parser')

3. 원하는 요소 찾기
   └─ soup.find('태그이름')
   └─ soup.find_all('태그이름')

4. 데이터 추출
   └─ 텍스트: tag.text
   └─ 속성: tag['href'], tag.get('src')
```

---

## 2. BeautifulSoup 소개 및 설치

### 2.1 BeautifulSoup이란?

**BeautifulSoup = HTML/XML을 쉽게 파싱하고 탐색할 수 있게 해주는 Python 라이브러리**

**이름의 유래:**
- "Beautiful Soup" = 아름다운 수프
- 복잡하게 섞인 HTML을 맛있는 수프처럼 잘 정리해준다는 의미

**특징:**
- ✅ 쉬운 문법 (Python스러움)
- ✅ 강력한 탐색 기능
- ✅ 잘못된 HTML도 처리 가능
- ✅ 다양한 파서 지원
- ✅ 활발한 커뮤니티

### 2.2 설치하기

**명령어:**
```bash
pip install beautifulsoup4
```

**설치 확인:**
```python
from bs4 import BeautifulSoup
print(BeautifulSoup.__version__)  # 예: 4.12.2
```

**주의:** 패키지 이름은 `beautifulsoup4`이지만, import할 때는 `bs4`입니다!

### 2.3 파서(Parser) 종류

BeautifulSoup는 여러 파서를 지원합니다:

| 파서 | 설치 | 속도 | 관대함 | 권장 |
|------|------|------|--------|------|
| **html.parser** | 기본 제공 | 보통 | 보통 | ✅ 권장 (초보자) |
| lxml | `pip install lxml` | 빠름 | 높음 | 고급 사용자 |
| html5lib | `pip install html5lib` | 느림 | 매우 높음 | 특수한 경우 |

**초보자는 `html.parser`를 사용하세요!**
- 별도 설치 불필요
- 대부분의 경우 충분히 빠름
- 사용법이 간단함

---

## 3. 기본 사용법: HTML 파싱하기

### 3.1 최소 코드

```python
from bs4 import BeautifulSoup

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

# 제목 추출
title = soup.find('title')
print(title.text)  # 출력: 내 첫 웹페이지
```

### 3.2 Requests와 함께 사용하기 (실무 패턴)

```python
import requests
from bs4 import BeautifulSoup

# 1단계: HTML 가져오기
url = 'http://example.com'
response = requests.get(url)
html = response.text

# 2단계: 파싱
soup = BeautifulSoup(html, 'html.parser')

# 3단계: 데이터 추출
h1_tag = soup.find('h1')
print(f"제목: {h1_tag.text}")
```

### 3.3 파싱 결과 확인하기

```python
from bs4 import BeautifulSoup

html = "<html><body><h1>제목</h1></body></html>"
soup = BeautifulSoup(html, 'html.parser')

# 1. 원본 HTML (예쁘게 출력)
print(soup.prettify())

# 2. 파싱 결과 타입 확인
print(type(soup))  # <class 'bs4.BeautifulSoup'>

# 3. 특정 태그 찾기
h1_tag = soup.find('h1')
print(type(h1_tag))  # <class 'bs4.element.Tag'>
print(h1_tag.name)   # 태그 이름: h1
print(h1_tag.text)   # 텍스트: 제목
```

---

## 4. 태그 찾기: find()와 find_all()

### 4.1 find() - 첫 번째 태그 하나만

**문법:**
```python
soup.find('태그이름')
soup.find('태그이름', attrs={'속성': '값'})
soup.find('태그이름', class_='클래스명')
```

**예제:**
```python
from bs4 import BeautifulSoup

html = """
<div class="news">
    <h2>첫 번째 뉴스</h2>
    <h2>두 번째 뉴스</h2>
    <h2>세 번째 뉴스</h2>
</div>
"""

soup = BeautifulSoup(html, 'html.parser')

# find()는 첫 번째 것만 반환
first_h2 = soup.find('h2')
print(first_h2.text)  # 출력: 첫 번째 뉴스
```

### 4.2 find_all() - 모든 태그 찾기

**문법:**
```python
soup.find_all('태그이름')
soup.find_all('태그이름', attrs={'속성': '값'})
soup.find_all('태그이름', class_='클래스명')
```

**예제:**
```python
from bs4 import BeautifulSoup

html = """
<div class="news">
    <h2>첫 번째 뉴스</h2>
    <h2>두 번째 뉴스</h2>
    <h2>세 번째 뉴스</h2>
</div>
"""

soup = BeautifulSoup(html, 'html.parser')

# find_all()은 모든 것을 리스트로 반환
all_h2 = soup.find_all('h2')
print(f"찾은 h2 태그 개수: {len(all_h2)}")  # 출력: 3

# 반복문으로 하나씩 처리
for h2 in all_h2:
    print(h2.text)

# 출력:
# 첫 번째 뉴스
# 두 번째 뉴스
# 세 번째 뉴스
```

### 4.3 클래스와 ID로 찾기

**클래스로 찾기:**
```python
html = """
<div class="article">기사 1</div>
<div class="article">기사 2</div>
<div class="notice">공지사항</div>
"""

soup = BeautifulSoup(html, 'html.parser')

# class 속성으로 찾기 (주의: class_로 쓴다)
articles = soup.find_all('div', class_='article')
print(len(articles))  # 출력: 2
```

**ID로 찾기:**
```python
html = """
<div id="main">메인 콘텐츠</div>
<div id="sidebar">사이드바</div>
"""

soup = BeautifulSoup(html, 'html.parser')

# id 속성으로 찾기
main_div = soup.find('div', id='main')
print(main_div.text)  # 출력: 메인 콘텐츠

# 또는 attrs 사용
main_div = soup.find('div', attrs={'id': 'main'})
```

### 4.4 여러 조건으로 찾기

```python
html = """
<a href="/news/1" class="news-link">뉴스 1</a>
<a href="/news/2" class="news-link">뉴스 2</a>
<a href="/notice/1" class="notice-link">공지 1</a>
"""

soup = BeautifulSoup(html, 'html.parser')

# 태그 + 클래스
news_links = soup.find_all('a', class_='news-link')
print(len(news_links))  # 출력: 2

# 태그 + 속성 여러 개
news_link = soup.find('a', attrs={
    'class': 'news-link',
    'href': '/news/1'
})
print(news_link.text)  # 출력: 뉴스 1
```

---

## 5. 데이터 추출: 텍스트와 속성

### 5.1 텍스트 추출

**기본 방법:**
```python
html = "<h1>안녕하세요!</h1>"
soup = BeautifulSoup(html, 'html.parser')

h1 = soup.find('h1')
print(h1.text)        # 출력: 안녕하세요!
print(h1.get_text())  # 출력: 안녕하세요! (같은 결과)
```

**중첩된 태그의 텍스트:**
```python
html = """
<div class="article">
    <h2>제목입니다</h2>
    <p>본문 내용입니다.</p>
</div>
"""

soup = BeautifulSoup(html, 'html.parser')

div = soup.find('div', class_='article')
print(div.text)
# 출력:
# 
#     제목입니다
#     본문 내용입니다.
# 

# 공백 제거하기
print(div.text.strip())
# 출력: 제목입니다 본문 내용입니다.
```

**텍스트만 깔끔하게:**
```python
# 줄바꿈을 공백으로 변환
text = div.get_text(strip=True)
print(text)  # 출력: 제목입니다본문 내용입니다.

# 구분자 추가
text = div.get_text(separator=' ', strip=True)
print(text)  # 출력: 제목입니다 본문 내용입니다.
```

### 5.2 속성(Attribute) 추출

**링크(href) 추출:**
```python
html = '<a href="https://www.python.org">Python 공식 사이트</a>'
soup = BeautifulSoup(html, 'html.parser')

link = soup.find('a')

# 방법 1: 딕셔너리 방식
print(link['href'])  # 출력: https://www.python.org

# 방법 2: get() 메서드 (안전)
print(link.get('href'))  # 출력: https://www.python.org
```

**이미지(src) 추출:**
```python
html = '<img src="/images/logo.png" alt="로고">'
soup = BeautifulSoup(html, 'html.parser')

img = soup.find('img')

print(img['src'])   # 출력: /images/logo.png
print(img['alt'])   # 출력: 로고
```

**여러 링크 추출:**
```python
html = """
<div>
    <a href="/news/1">뉴스 1</a>
    <a href="/news/2">뉴스 2</a>
    <a href="/news/3">뉴스 3</a>
</div>
"""

soup = BeautifulSoup(html, 'html.parser')

links = soup.find_all('a')

for link in links:
    text = link.text
    url = link['href']
    print(f"제목: {text}, URL: {url}")

# 출력:
# 제목: 뉴스 1, URL: /news/1
# 제목: 뉴스 2, URL: /news/2
# 제목: 뉴스 3, URL: /news/3
```

### 5.3 속성이 없을 때 안전하게 처리하기

```python
html = '<a>링크인데 href가 없음</a>'
soup = BeautifulSoup(html, 'html.parser')

link = soup.find('a')

# ❌ 위험: KeyError 발생 가능
# print(link['href'])  

# ✅ 안전: get() 사용
href = link.get('href')
if href:
    print(f"URL: {href}")
else:
    print("href 속성이 없습니다")

# 또는 기본값 설정
href = link.get('href', '링크 없음')
print(href)  # 출력: 링크 없음
```

---

## 6. HTML 구조 탐색하기

### 6.1 부모, 자식, 형제 관계

HTML은 트리 구조입니다:

```
<html>              ← 최상위 부모
  └─ <body>         ← html의 자식, div의 부모
      └─ <div>      ← body의 자식, h1과 p의 부모
          ├─ <h1>   ← div의 자식, p의 형제
          └─ <p>    ← div의 자식, h1의 형제
```

### 6.2 부모 요소 접근

```python
html = """
<div class="container">
    <h1 id="title">제목</h1>
</div>
"""

soup = BeautifulSoup(html, 'html.parser')

h1 = soup.find('h1')

# 부모 태그 접근
parent = h1.parent
print(parent.name)        # 출력: div
print(parent['class'])    # 출력: ['container']
```

### 6.3 자식 요소 접근

```python
html = """
<ul>
    <li>항목 1</li>
    <li>항목 2</li>
    <li>항목 3</li>
</ul>
"""

soup = BeautifulSoup(html, 'html.parser')

ul = soup.find('ul')

# 모든 자식 (리스트)
children = ul.find_all('li')
print(f"자식 개수: {len(children)}")  # 출력: 3

# 자식 순회
for child in ul.children:
    if child.name:  # 텍스트 노드 제외
        print(child.text)
```

### 6.4 형제 요소 접근

```python
html = """
<div>
    <h1>제목</h1>
    <p>첫 번째 문단</p>
    <p>두 번째 문단</p>
</div>
"""

soup = BeautifulSoup(html, 'html.parser')

h1 = soup.find('h1')

# 다음 형제
next_sibling = h1.find_next_sibling()
print(next_sibling.text)  # 출력: 첫 번째 문단

# 모든 다음 형제들
next_siblings = h1.find_next_siblings()
print(f"다음 형제 개수: {len(next_siblings)}")  # 출력: 2
```

---

## 💡 핵심 요약

### 배운 내용 정리

1. **HTML 파싱 = HTML을 구조적으로 해석**
   - BeautifulSoup로 쉽게 처리 가능

2. **기본 사용법**
   ```python
   from bs4 import BeautifulSoup
   soup = BeautifulSoup(html, 'html.parser')
   ```

3. **태그 찾기**
   ```python
   soup.find('태그')           # 첫 번째 하나
   soup.find_all('태그')       # 모두
   soup.find('태그', class_='클래스')
   ```

4. **데이터 추출**
   ```python
   tag.text                    # 텍스트
   tag['속성']                 # 속성 값
   tag.get('속성', '기본값')   # 안전하게
   ```

5. **실무 패턴**
   ```python
   # 1. HTML 가져오기 (Requests)
   response = requests.get(url)
   
   # 2. 파싱 (BeautifulSoup)
   soup = BeautifulSoup(response.text, 'html.parser')
   
   # 3. 데이터 추출
   titles = soup.find_all('h1')
   for title in titles:
       print(title.text)
   ```

---

## 🔍 실전 예제

### 예제: 웹페이지에서 모든 링크 추출

```python
import requests
from bs4 import BeautifulSoup

def extract_all_links(url):
    """웹페이지의 모든 링크 추출"""
    try:
        # 1. HTML 가져오기
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # 2. 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. 모든 a 태그 찾기
        links = soup.find_all('a')
        
        # 4. href 추출
        results = []
        for link in links:
            href = link.get('href')
            text = link.text.strip()
            
            if href:  # href가 있는 경우만
                results.append({
                    'text': text,
                    'url': href
                })
        
        return results
        
    except Exception as e:
        print(f"오류: {e}")
        return []

# 사용 예시
url = 'http://example.com'
links = extract_all_links(url)

print(f"총 {len(links)}개의 링크를 찾았습니다.\n")

for i, link in enumerate(links, 1):
    print(f"{i}. {link['text']}")
    print(f"   → {link['url']}\n")
```

---

## 🎓 다음 단계 예고

**3교시에서 배울 내용:**
- CSS 선택자로 더 정밀하게 데이터 수집
- select()와 select_one() 사용법
- 실전 프로젝트: 뉴스 기사 목록 전체 수집
- 데이터 구조화 및 CSV 저장

**준비할 것:**
- 2교시 내용 복습
- 크롬 개발자 도구 사용법 익히기

---

**작성일**: 2024년  
**난이도**: 초급 (Beginner)  
**예상 학습 시간**: 1시간
