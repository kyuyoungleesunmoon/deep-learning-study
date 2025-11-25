# 📖 Chapter 03: Models & Prompts

## 📋 개요

이 챕터에서는 LangChain을 활용한 LLM API 사용법과 프롬프트 엔지니어링을 학습합니다.
- 다양한 LLM API 통합 (OpenAI, Anthropic)
- 프롬프트 템플릿
- Output Parser

## 🔬 핵심 개념

### 1. LangChain의 장점

**직접 API 호출**:
```python
from openai import OpenAI
client = OpenAI(api_key="...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**LangChain 활용**:
```python
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model_name="gpt-4o-mini")
response = chat.invoke("Hello")
```

**장점**:
- 다양한 LLM을 동일한 인터페이스로 사용
- 프롬프트 템플릿, 메모리, 체인 등 추가 기능
- 쉬운 모델 전환 (OpenAI ↔ Anthropic)

### 2. 프롬프트 템플릿

**PromptTemplate**: 기본 문자열 템플릿
```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "요리사로서 {재료}로 만들 수 있는 요리 {개수}개를 추천해줘"
)
prompt = template.format(재료="계란, 양파", 개수=3)
```

**ChatPromptTemplate**: 대화형 템플릿
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name}"),
    ("human", "{question}")
])
messages = template.format_messages(name="Bob", question="What's your name?")
```

### 3. Output Parser

**목적**: LLM 출력을 구조화된 형식으로 변환

| Parser | 출력 형식 | 용도 |
|--------|----------|------|
| `CommaSeparatedListOutputParser` | 리스트 | 목록 생성 |
| `DatetimeOutputParser` | datetime | 날짜 추출 |
| `JsonOutputParser` | JSON/Dict | 구조화 데이터 |
| `PydanticOutputParser` | Pydantic 모델 | 타입 검증 |

### 4. Temperature 파라미터

| 값 | 특성 | 용도 |
|---|------|------|
| 0 | 결정론적, 일관성 | 팩트 기반 답변 |
| 0.5 | 균형 | 일반 대화 |
| 1.0 | 창의적, 다양성 | 창작, 브레인스토밍 |

## 📊 실습 예제

### 예제 1: 기본 LLM 호출

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key"

# 모델 초기화
chat = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# 단순 호출
response = chat.invoke("파이썬의 장점을 한 문장으로 설명해줘")
print(response.content)
```

### 예제 2: 프롬프트 템플릿

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# 시스템 메시지로 역할 부여
template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다. 친절하게 답변해주세요."),
    ("human", "{question}")
])

# 메시지 생성
messages = template.format_messages(
    role="Python 프로그래밍",
    question="리스트와 튜플의 차이가 뭐야?"
)

# LLM 호출
response = chat.invoke(messages)
print(response.content)
```

### 예제 3: Few-shot 프롬프트

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# 예시들
examples = [
    {"input": "아이유", "output": "아: 아이유는\n이: 이렇게\n유: 유명해요"},
    {"input": "방탄", "output": "방: 방금\n탄: 탄생한\n"}
]

# 예시 템플릿
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="입력: {input}\n출력:\n{output}"
)

# Few-shot 템플릿
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="삼행시를 만들어주세요.\n",
    suffix="입력: {word}\n출력:",
    input_variables=["word"]
)

# 사용
final_prompt = prompt.format(word="코딩")
response = chat.invoke(final_prompt)
```

### 예제 4: Output Parser

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

# 파서 생성
parser = CommaSeparatedListOutputParser()

# 파서 지침을 프롬프트에 포함
template = PromptTemplate(
    template="{subject}의 종류 {count}개를 나열해주세요.\n{format_instructions}",
    input_variables=["subject", "count"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 체인 구성
prompt = template.format(subject="프로그래밍 언어", count=5)
response = chat.invoke(prompt)

# 파싱
result = parser.parse(response.content)
print(result)  # ['Python', 'JavaScript', 'Java', 'C++', 'Go']
```

### 예제 5: JSON Output Parser

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 출력 스키마 정의
class Country(BaseModel):
    name: str = Field(description="나라 이름")
    capital: str = Field(description="수도")
    population: int = Field(description="인구수")

parser = JsonOutputParser(pydantic_object=Country)

template = PromptTemplate(
    template="{country}에 대한 정보를 알려주세요.\n{format_instructions}",
    input_variables=["country"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 체인 (LCEL)
chain = template | chat | parser
result = chain.invoke({"country": "대한민국"})
print(result)  # {'name': '대한민국', 'capital': '서울', 'population': 51000000}
```

### 예제 6: 스트리밍

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 스트리밍 출력
for chunk in chat.stream("달에 대한 시를 써줘"):
    print(chunk.content, end="", flush=True)
```

## 🎯 핵심 포인트

1. **LangChain의 추상화**: 다양한 LLM을 동일한 인터페이스로 사용
2. **프롬프트 엔지니어링**: 템플릿으로 재사용 가능한 프롬프트 설계
3. **Output Parser**: 구조화된 출력으로 후처리 용이
4. **LCEL**: `|` 연산자로 직관적인 체인 구성

## 📚 참고 자료

- 원본 코드: https://github.com/Kane0002/Langchain-RAG/tree/main/3장
- LangChain 문서: https://python.langchain.com/
- OpenAI API: https://platform.openai.com/docs
