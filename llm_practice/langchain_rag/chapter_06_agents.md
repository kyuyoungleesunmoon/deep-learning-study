# 📖 Chapter 06: Tools & Agents

## 📋 개요

이 챕터에서는 LLM이 외부 도구를 활용하는 Agent 시스템을 학습합니다.
- Tool의 개념과 정의
- Agent 아키텍처
- 실시간 검색 통합 (Tavily)

## 🔬 핵심 개념

### 1. LLM의 한계와 Tool

**LLM의 한계**:
- 실시간 정보 없음 (학습 데이터 이후 정보)
- 계산 정확도 낮음
- 외부 시스템 접근 불가

**Tool로 해결**:
```
사용자 질문 → LLM → "계산이 필요하군" → Calculator Tool → 결과 → LLM → 최종 답변
```

### 2. Agent 아키텍처

```
사용자 질문
    │
    ▼
┌─────────────┐
│    LLM      │ ◄─── Tool 결과 피드백
│  (추론기)   │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Tool 선택   │
│ (어떤 도구?)│
└─────────────┘
    │
    ├──► Tool 1 (검색)
    ├──► Tool 2 (계산기)
    └──► Tool 3 (DB 쿼리)
```

**ReAct 패턴**:
```
Thought: 무엇을 해야 할까? (추론)
Action: 어떤 도구를 사용할까? (행동)
Observation: 도구 실행 결과 (관찰)
... (반복)
Final Answer: 최종 답변
```

### 3. Tavily Search

**특징**:
- AI 최적화된 웹 검색 API
- 실시간 정보 제공
- 요약된 결과 반환

### 4. Multi-Tool Agent

```python
# 여러 Tool 정의
tools = [
    TavilySearchResults(),  # 웹 검색
    Calculator(),           # 계산
    DatabaseQuery()         # DB 쿼리
]

# Agent가 상황에 맞는 Tool 자동 선택
agent = create_openai_tools_agent(llm, tools, prompt)
```

## 📊 실습 예제

### 예제 1: 기본 Tool 정의

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """수학 표현식을 계산합니다. 예: '2 + 2'"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "계산할 수 없습니다"

@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### 예제 2: Tavily 검색 Agent

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
import os

os.environ["TAVILY_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 도구 설정
tools = [TavilySearchResults(max_results=3)]

# 프롬프트 및 LLM
prompt = hub.pull("hwchase17/openai-tools-agent")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent 생성
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = agent_executor.invoke({"input": "오늘 비트코인 가격은?"})
print(result["output"])
```

### 예제 3: 벡터 DB + 검색 통합

```python
from langchain.tools.retriever import create_retriever_tool

# 벡터 DB에서 Retriever Tool 생성
retriever_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description="회사 문서에서 정보를 검색합니다."
)

# 웹 검색 도구
search_tool = TavilySearchResults()

# 두 도구 결합
tools = [retriever_tool, search_tool]

# Agent 생성
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# 질문에 따라 적절한 도구 선택
result = executor.invoke({"input": "회사 휴가 정책은?"})  # → retriever_tool
result = executor.invoke({"input": "오늘 날씨는?"})      # → search_tool
```

### 예제 4: Streamlit 챗봇

```python
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

st.title("AI 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # AI 응답
    with st.chat_message("assistant"):
        response = llm.invoke(prompt)
        st.write(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
```

### 예제 5: 커스텀 Tool

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """날씨 조회 입력 스키마"""
    city: str = Field(description="도시 이름")

class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "특정 도시의 날씨를 조회합니다"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        # 실제로는 날씨 API 호출
        return f"{city}의 날씨: 맑음, 23°C"

weather_tool = WeatherTool()
```

## 🎯 핵심 포인트

1. **Tool 설명 중요**: LLM이 적절한 도구를 선택하려면 명확한 설명 필요
2. **Verbose 모드**: 디버깅 시 추론 과정 확인
3. **오류 처리**: Tool 실패 시 graceful degradation
4. **토큰 비용**: Agent는 여러 번 LLM 호출 가능

## ⚠️ 주의사항

- API 키 보안: 환경 변수 사용
- 비용 관리: Agent는 토큰 소비 많음
- 무한 루프: max_iterations 설정

## 📚 참고 자료

- 원본 코드: https://github.com/Kane0002/Langchain-RAG/tree/main/6장
- Tavily API: https://tavily.com/
- LangChain Agents: https://python.langchain.com/docs/modules/agents/
