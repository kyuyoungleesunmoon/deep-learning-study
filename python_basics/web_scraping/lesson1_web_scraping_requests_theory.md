# 웹 스크레이핑 입문 - 1교시: 웹 스크레이핑 개념 이해 + Requests 라이브러리

> **대상**: 파이썬을 처음 배우는 직장인
> 
> **예상 학습 시간**: 1시간 (이론 30분 + 실습 30분)

---

## 🎯 학습 목표

이 과정을 마치면 다음을 할 수 있게 됩니다:

1. **웹 스크레이핑 개념 이해**: 웹 스크레이핑이 무엇인지, 왜 필요한지 이해합니다
2. **HTTP 기초 이해**: 웹페이지를 어떻게 가져오는지 기본 원리를 이해합니다
3. **Requests 라이브러리 사용**: Python으로 웹페이지의 HTML을 가져올 수 있습니다
4. **실무 활용**: 실제 웹사이트의 HTML 소스코드를 콘솔에 출력할 수 있습니다
5. **법적/윤리적 고려사항**: 웹 스크레이핑 시 지켜야 할 규칙을 이해합니다

---

## 📚 목차

1. [웹 스크레이핑이란 무엇인가?](#1-웹-스크레이핑이란-무엇인가)
2. [웹 스크레이핑의 실무 활용 사례](#2-웹-스크레이핑의-실무-활용-사례)
3. [웹페이지의 구조 이해 (HTML 기초)](#3-웹페이지의-구조-이해-html-기초)
4. [Requests 라이브러리 소개](#4-requests-라이브러리-소개)
5. [실습: 첫 번째 웹 스크레이핑](#5-실습-첫-번째-웹-스크레이핑)
6. [법적/윤리적 고려사항](#6-법적윤리적-고려사항)

---

## 1. 웹 스크레이핑이란 무엇인가?

### 1.1 웹 스크레이핑의 개념 (쉬운 비유)

**웹 스크레이핑 = 웹페이지에서 필요한 정보를 자동으로 수집하는 기술**

일상생활에서 비유하면:
- 📰 **신문 스크랩**: 신문에서 필요한 기사만 오려서 모으는 것
- 📊 **시장 조사**: 여러 마트를 돌아다니며 가격표를 적어서 비교하는 것
- 📝 **정보 수집**: 도서관에서 여러 책을 읽고 중요한 내용만 노트에 정리하는 것

**웹 스크레이핑이 하는 일:**
```
웹 브라우저로 하는 일              →  프로그램으로 자동화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 웹사이트 접속                  →  자동으로 접속
2. 페이지 로딩                    →  자동으로 HTML 다운로드
3. 원하는 정보 찾기 (마우스/눈)    →  코드로 자동 추출
4. 복사-붙여넣기                  →  자동으로 파일 저장
```

### 1.2 웹 크롤링 vs 웹 스크레이핑

많은 사람들이 두 용어를 혼용하지만, 엄밀히 말하면:

| 구분 | 웹 크롤링 (Web Crawling) | 웹 스크레이핑 (Web Scraping) |
|------|-------------------------|----------------------------|
| **목적** | 웹페이지를 체계적으로 탐색 | 특정 데이터를 추출 |
| **범위** | 넓고 광범위 (여러 페이지) | 좁고 구체적 (원하는 데이터만) |
| **비유** | 도서관 전체를 돌아다니기 | 필요한 책에서 필요한 부분만 찾기 |
| **예시** | 구글 검색엔진이 모든 웹페이지 수집 | 쿠팡에서 특정 상품의 가격만 수집 |

**실무에서는**: 두 용어를 거의 같은 의미로 사용합니다!

---

## 2. 웹 스크레이핑의 실무 활용 사례

### 2.1 업무 자동화

**💼 직장인이 실제로 사용하는 예시들:**

#### 사례 1: 경쟁사 가격 모니터링 (마케팅팀)
```
문제: 매일 아침 경쟁사 10개 웹사이트에서 우리 제품과 같은 제품의 가격을 확인해야 함
해결: 웹 스크레이핑으로 자동 수집 → 엑셀 파일로 저장
시간 절약: 수작업 1시간 → 자동화 3분
```

#### 사례 2: 부동산 매물 정보 수집 (부동산 중개사)
```
문제: 여러 부동산 사이트에서 특정 지역의 매물을 일일이 확인
해결: 자동으로 매물 정보 수집 → 가격/평수별로 정리
효과: 고객에게 빠른 정보 제공, 시장 동향 분석 가능
```

#### 사례 3: 뉴스 기사 수집 (홍보팀/기획팀)
```
문제: 우리 회사 관련 뉴스를 매일 검색해서 보고서 작성
해결: 뉴스 사이트에서 키워드로 자동 검색 → 제목/날짜/링크 수집
효과: 일일 모니터링 자동화, 트렌드 분석 가능
```

#### 사례 4: 환율/주가 정보 수집 (재무팀)
```
문제: 매일 특정 시각의 환율과 주가를 확인해서 기록
해결: 자동으로 데이터 수집 → 데이터베이스 저장
효과: 실시간 모니터링, 히스토리 데이터 축적
```

### 2.2 데이터 분석 기초 자료 확보

웹 스크레이핑으로 수집한 데이터로:
- 📊 **트렌드 분석**: 시간에 따른 가격 변동 그래프
- 📈 **통계 분석**: 평균, 최저가, 최고가 계산
- 🤖 **머신러닝**: 가격 예측 모델 학습용 데이터

---

## 3. 웹페이지의 구조 이해 (HTML 기초)

### 3.1 웹페이지는 어떻게 만들어질까?

**웹페이지 = HTML + CSS + JavaScript**

```
HTML (구조)     →  집의 뼈대 (벽, 문, 창문)
CSS (디자인)    →  집의 인테리어 (색깔, 크기, 배치)
JavaScript (동작) →  집의 기능 (자동문, 조명)
```

**웹 스크레이핑이 주로 다루는 것 = HTML**

### 3.2 HTML의 기본 구조

HTML은 **태그(Tag)**로 이루어져 있습니다:

```html
<!-- 기본 형태: <태그이름>내용</태그이름> -->

<h1>안녕하세요</h1>
<!-- h1 = 제목(Heading 1) 태그 -->

<p>이것은 문단입니다.</p>
<!-- p = 문단(Paragraph) 태그 -->

<a href="https://www.google.com">구글로 이동</a>
<!-- a = 링크(Anchor) 태그 -->
```

### 3.3 실제 웹페이지 HTML 예시

네이버 뉴스 기사 제목이 어떻게 저장되어 있을까요?

```html
<div class="news_area">
    <a href="/news/article/001/0012345678" class="news_tit">
        [단독] 파이썬으로 업무 자동화 성공 사례
    </a>
    <span class="press">뉴스타임즈</span>
    <span class="date">2024.01.15</span>
</div>
```

**우리가 추출하고 싶은 것:**
- 제목: "파이썬으로 업무 자동화 성공 사례"
- 언론사: "뉴스타임즈"
- 날짜: "2024.01.15"

### 3.4 브라우저에서 HTML 확인하기

**Chrome/Edge 개발자 도구 사용법:**

```
방법 1: F12 키 누르기
방법 2: 우클릭 → "검사" 클릭
방법 3: Ctrl + Shift + I (Windows) / Cmd + Option + I (Mac)
```

**실습해보기:**
1. 아무 웹사이트 접속 (예: 네이버)
2. F12 눌러서 개발자 도구 열기
3. Elements 탭에서 HTML 구조 확인
4. 원하는 요소에 마우스 올리면 페이지에서 하이라이트됨!

---

## 4. Requests 라이브러리 소개

### 4.1 Requests가 하는 일

**Requests = 파이썬에서 웹페이지를 가져오는 가장 쉬운 방법**

```
사람이 하는 일                프로그램(Requests)이 하는 일
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 브라우저 열기              import requests
2. 주소창에 URL 입력          response = requests.get(url)
3. Enter → 페이지 로딩        (자동)
4. 화면에 표시됨              html = response.text
```

### 4.2 HTTP 요청/응답의 개념 (초간단 버전)

**HTTP = 웹에서 데이터를 주고받는 약속(프로토콜)**

```
[당신의 컴퓨터]  →  "네이버 메인페이지 주세요!"  →  [네이버 서버]
                     (HTTP 요청 - Request)

[당신의 컴퓨터]  ←  "여기 HTML 파일입니다!"   ←  [네이버 서버]
                     (HTTP 응답 - Response)
```

**주요 HTTP 메서드:**
- **GET**: 데이터를 가져올 때 (웹 스크레이핑에서 주로 사용)
- POST: 데이터를 보낼 때 (로그인, 폼 제출 등)
- PUT, DELETE 등: 업데이트, 삭제 (고급 주제)

### 4.3 Requests 설치

**터미널/명령 프롬프트에서 실행:**

```bash
pip install requests
```

**설치 확인:**
```python
import requests
print(requests.__version__)  # 예: 2.31.0
```

### 4.4 Requests의 기본 문법

#### 최소한의 코드
```python
import requests

# URL에 접속해서 HTML 가져오기
response = requests.get('https://www.example.com')

# HTML 내용 출력
print(response.text)
```

#### 주요 속성과 메서드

```python
import requests

url = 'https://www.example.com'
response = requests.get(url)

# 1. 상태 코드 확인 (성공했는지 확인)
print(response.status_code)  # 200 = 성공, 404 = 페이지 없음

# 2. HTML 내용 (문자열)
print(response.text)

# 3. 바이트 형태의 내용
print(response.content)

# 4. 인코딩 확인
print(response.encoding)  # 예: 'utf-8'

# 5. 헤더 정보
print(response.headers)
```

**주요 상태 코드:**
- **200**: 성공 ✅
- **404**: 페이지를 찾을 수 없음 ❌
- **403**: 접근 권한 없음 🚫
- **500**: 서버 오류 💥

---

## 5. 실습: 첫 번째 웹 스크레이핑

### 5.1 예제 1: 기본 HTML 가져오기

```python
import requests

# Example.com은 테스트용 공개 웹사이트입니다
url = 'http://example.com'

# 웹페이지 요청
response = requests.get(url)

# 상태 코드 출력 (200이면 성공!)
print(f"상태 코드: {response.status_code}")

# HTML 전체 내용 출력
print("\n" + "="*50)
print("HTML 내용:")
print("="*50)
print(response.text)
```

**출력 결과:**
```
상태 코드: 200
==================================================
HTML 내용:
==================================================
<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    ...
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples...</p>
</div>
</body>
</html>
```

### 5.2 예제 2: 최신 스타일 - 예외 처리 포함

```python
import requests

def fetch_webpage(url):
    """
    웹페이지의 HTML을 가져오는 함수
    
    Args:
        url (str): 가져올 웹페이지 URL
    
    Returns:
        str: HTML 내용 또는 None (실패 시)
    """
    try:
        # timeout: 5초 내에 응답 없으면 종료
        response = requests.get(url, timeout=5)
        
        # 상태 코드가 200번대가 아니면 예외 발생
        response.raise_for_status()
        
        return response.text
        
    except requests.exceptions.Timeout:
        print(f"⏱️ 타임아웃: {url} 응답이 너무 느립니다.")
        return None
        
    except requests.exceptions.ConnectionError:
        print(f"🔌 연결 오류: {url}에 접속할 수 없습니다.")
        return None
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP 오류: {e}")
        return None
        
    except Exception as e:
        print(f"⚠️ 알 수 없는 오류: {e}")
        return None

# 사용 예시
url = 'http://example.com'
html = fetch_webpage(url)

if html:
    print("✅ 성공적으로 HTML을 가져왔습니다!")
    print(f"HTML 길이: {len(html)} 문자")
    print("\n첫 500자:")
    print(html[:500])
else:
    print("❌ HTML을 가져오는 데 실패했습니다.")
```

### 5.3 예제 3: User-Agent 설정 (실무 팁)

일부 웹사이트는 프로그램의 접근을 차단합니다. User-Agent를 설정하면 일반 브라우저처럼 보이게 할 수 있습니다.

```python
import requests

url = 'http://example.com'

# User-Agent를 크롬 브라우저로 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

response = requests.get(url, headers=headers)

print(f"상태 코드: {response.status_code}")
print(f"인코딩: {response.encoding}")
```

**왜 필요한가요?**
- 일부 웹사이트는 봇(Bot) 접근을 막기 위해 User-Agent를 확인합니다
- User-Agent가 없으면 403 에러가 날 수 있습니다
- 실무에서는 거의 항상 설정합니다!

### 5.4 예제 4: with문 사용 (권장 방식)

```python
import requests

url = 'http://example.com'

# with문을 사용하면 자동으로 연결을 닫아줍니다 (메모리 효율적)
with requests.Session() as session:
    response = session.get(url)
    
    if response.status_code == 200:
        print("✅ 성공!")
        print(f"응답 시간: {response.elapsed.total_seconds():.2f}초")
        print(f"내용 크기: {len(response.content)} 바이트")
    else:
        print(f"❌ 실패: 상태 코드 {response.status_code}")
```

---

## 6. 법적/윤리적 고려사항

### 6.1 웹 스크레이핑, 합법인가요?

**✅ 일반적으로 합법적인 경우:**
- 공개된 웹페이지의 공개 정보 수집
- 개인적 사용 목적
- 웹사이트의 robots.txt를 준수할 때
- 서버에 과부하를 주지 않을 때

**❌ 불법이 될 수 있는 경우:**
- 로그인이 필요한 정보 무단 수집
- 저작권이 있는 콘텐츠 무단 복제
- 개인정보 무단 수집
- 서버 공격 수준의 과도한 요청

### 6.2 robots.txt 확인하기

**robots.txt = 웹사이트가 허용/금지하는 스크레이핑 규칙**

**확인 방법:**
```
웹사이트 주소 + /robots.txt

예시:
https://www.naver.com/robots.txt
https://www.google.com/robots.txt
```

**예시:**
```
User-agent: *
Disallow: /admin/        # /admin/ 경로는 스크래핑 금지
Allow: /news/           # /news/ 경로는 허용
Crawl-delay: 1          # 요청 사이에 1초 대기
```

### 6.3 윤리적인 웹 스크레이핑 가이드

**🌟 지켜야 할 원칙:**

1. **적절한 속도 유지**
```python
import time

for url in url_list:
    response = requests.get(url)
    process(response)
    time.sleep(1)  # 1초 대기 (서버 부담 감소)
```

2. **User-Agent 명시**
```python
headers = {
    'User-Agent': 'MyBot/1.0 (contact@example.com)'
}
```

3. **robots.txt 준수**
```python
# robotparser 사용 예시
from urllib.robotparser import RobotFileParser

rp = RobotFileParser()
rp.set_url('https://www.example.com/robots.txt')
rp.read()

url = 'https://www.example.com/news'
if rp.can_fetch('*', url):
    response = requests.get(url)
```

4. **이용 약관 확인**
- 웹사이트의 "이용 약관" 또는 "Terms of Service" 확인
- API가 제공된다면 API 사용 권장

5. **개인정보 보호**
- 이메일, 전화번호 등 개인정보는 수집하지 않기
- 수집한 데이터의 적절한 관리

### 6.4 실무에서의 Best Practice

**권장:**
- 공식 API가 있다면 API 사용 (예: 네이버 뉴스 API, 카카오 API)
- 작은 규모의 데이터 수집
- 오픈 데이터 우선 활용

**주의:**
- 대규모 자동화 크롤링 자제
- 상업적 목적일 경우 사전 승인 필요
- 저작권 침해 주의

---

## 💡 핵심 요약

### 배운 내용 정리

1. **웹 스크레이핑 = 웹페이지에서 자동으로 데이터 수집**
   - 업무 자동화, 데이터 분석에 유용

2. **HTML = 웹페이지의 구조**
   - 태그로 이루어짐
   - F12로 확인 가능

3. **Requests = Python에서 웹페이지 가져오기**
   ```python
   response = requests.get(url)
   html = response.text
   ```

4. **실무 코딩 스타일**
   - f-string 사용
   - 예외 처리 (try-except)
   - with문 사용
   - User-Agent 설정

5. **법적/윤리적 준수**
   - robots.txt 확인
   - 적절한 속도 유지
   - 개인정보 보호

---

## 🎓 다음 단계 예고

**2교시에서 배울 내용:**
- BeautifulSoup로 HTML 파싱
- 특정 태그의 텍스트 추출
- 실전: 웹페이지에서 제목만 뽑아내기

**준비할 것:**
```bash
pip install beautifulsoup4
```

---

**작성일**: 2024년  
**난이도**: 초급 (Beginner)  
**예상 학습 시간**: 1시간
