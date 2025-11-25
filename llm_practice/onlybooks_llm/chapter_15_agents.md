# 📖 Chapter 15: LLM 에이전트 (LLM Agents)

## 📋 개요

이 챕터에서는 LLM을 활용한 에이전트 시스템을 학습합니다.
- AutoGen 프레임워크
- 멀티에이전트 시스템
- 코드 실행 에이전트

## 🔬 핵심 개념

### 1. LLM 에이전트란?

**정의**: LLM이 외부 도구를 활용하여 복잡한 태스크를 수행하는 시스템

```
사용자 요청 → LLM (추론) → 도구 선택 → 도구 실행 → 결과 해석 → 응답
```

**핵심 구성요소**:
- **LLM**: 추론 및 계획 수립
- **도구 (Tools)**: 검색, 계산, 코드 실행 등
- **메모리**: 대화 히스토리, 중간 결과 저장
- **플래너**: 복잡한 태스크 분해

### 2. AutoGen 프레임워크

**특징**:
- Microsoft에서 개발한 멀티에이전트 프레임워크
- 에이전트 간 대화를 통한 협업
- 코드 생성 및 실행 자동화

**핵심 에이전트 타입**:

| 에이전트 | 역할 |
|----------|------|
| `AssistantAgent` | 지시에 따라 작업 수행 |
| `UserProxyAgent` | 사용자 역할, 코드 실행 |
| `GroupChatManager` | 여러 에이전트 조율 |

### 3. 에이전트 대화 패턴

**Two-Agent Pattern**:
```
User ↔ Assistant
사용자 요청 → 어시스턴트 응답 → 사용자 피드백 → ...
```

**Group Chat Pattern**:
```
Manager → Agent1 → Agent2 → Agent3 → Manager
각 에이전트가 전문 영역 담당
```

## 📊 실습 예제

### 예제 1: 기본 AutoGen 설정

```python
import autogen

# API 설정
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": "your-api-key"
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0,
}

# Assistant 에이전트
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant."
)

# User Proxy 에이전트
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 자동 실행
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# 대화 시작
user_proxy.initiate_chat(
    assistant,
    message="파이썬으로 피보나치 수열을 계산하는 함수를 작성해줘."
)
```

### 예제 2: 코드 실행 에이전트

```python
import autogen

# 종료 조건 설정
def is_termination_msg(msg):
    content = msg.get("content", "")
    return content and content.rstrip().endswith("TERMINATE")

# Assistant 에이전트
assistant = autogen.AssistantAgent(
    name="code_assistant",
    llm_config=llm_config,
    system_message="""
    You are a Python programming expert.
    When the task is complete, reply with TERMINATE.
    """
)

# User Proxy (코드 실행)
user_proxy = autogen.UserProxyAgent(
    name="executor",
    is_termination_msg=is_termination_msg,
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False
    }
)

# 복잡한 태스크 실행
user_proxy.initiate_chat(
    assistant,
    message="""
    삼성전자의 최근 3개월 주식 가격 데이터를 가져와서
    그래프로 시각화하고 stock_price.png로 저장해줘.
    """
)
```

### 예제 3: 멀티에이전트 협업

```python
import autogen

# 여러 전문 에이전트 정의
researcher = autogen.AssistantAgent(
    name="researcher",
    llm_config=llm_config,
    system_message="""
    You are a research expert.
    Search for information and provide summaries.
    """
)

coder = autogen.AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message="""
    You are a Python programmer.
    Write code based on research findings.
    """
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    llm_config=llm_config,
    system_message="""
    You are a code reviewer.
    Review code for bugs and improvements.
    """
)

# 사용자 프록시
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "workspace"}
)

# 그룹 채팅 설정
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, coder, reviewer],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 복잡한 태스크 시작
user_proxy.initiate_chat(
    manager,
    message="머신러닝 모델로 IRIS 데이터셋을 분류하는 코드를 작성해줘."
)
```

### 예제 4: 도구 사용 에이전트

```python
import autogen

# 커스텀 도구 정의
def search_web(query: str) -> str:
    """웹 검색 시뮬레이션"""
    return f"'{query}'에 대한 검색 결과: ..."

def calculate(expression: str) -> float:
    """수식 계산"""
    return eval(expression)

# Function 등록
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

# 함수를 에이전트에 등록
@assistant.register_for_llm(description="Search the web")
def web_search(query: str) -> str:
    return search_web(query)

@assistant.register_for_llm(description="Calculate math expression")
def calc(expression: str) -> float:
    return calculate(expression)

# User Proxy에도 실행 권한 등록
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER"
)

user_proxy.register_for_execution(name="web_search")(web_search)
user_proxy.register_for_execution(name="calc")(calc)
```

## 🎯 핵심 포인트

1. **에이전트 역할 분리**: 각 에이전트에 명확한 역할 부여
2. **종료 조건**: 무한 루프 방지를 위한 종료 조건 설정
3. **코드 실행 안전성**: Docker 사용 또는 샌드박스 환경
4. **토큰 비용**: 멀티에이전트는 토큰 사용량 증가

## ⚠️ 주의사항

- API 키 노출 주의
- 코드 실행 시 보안 고려
- 토큰 비용 모니터링

## 📚 참고 자료

- 원본 코드: https://github.com/onlybooks/llm/tree/main/15장
- AutoGen 문서: https://microsoft.github.io/autogen/
- AutoGen GitHub: https://github.com/microsoft/autogen
