"""
Chapter 15: LLM 에이전트 실습 코드
==================================

이 파일은 LLM 에이전트의 개념을 실습합니다:
1. 에이전트 패턴 시뮬레이션
2. 도구 사용 개념
3. 멀티에이전트 협업
4. (선택) AutoGen 사용

실행 방법:
    pip install numpy
    python chapter_15_practice.py

    # AutoGen 사용 시:
    pip install pyautogen openai
"""

from typing import List, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import re


# ============================================================
# Part 1: 기본 에이전트 구조
# ============================================================

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """대화 메시지"""
    role: MessageRole
    content: str
    name: str = ""
    tool_calls: List[Dict] = field(default_factory=list)


@dataclass
class Tool:
    """도구 정의"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, str] = field(default_factory=dict)


class SimpleAgent:
    """
    간단한 에이전트 시뮬레이터
    
    실제 LLM 대신 규칙 기반으로 동작합니다.
    """
    
    def __init__(self, name: str, system_message: str = ""):
        self.name = name
        self.system_message = system_message
        self.messages: List[Message] = []
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool):
        """도구 등록"""
        self.tools[tool.name] = tool
    
    def add_message(self, role: MessageRole, content: str):
        """메시지 추가"""
        self.messages.append(Message(role=role, content=content))
    
    def get_response(self, user_input: str) -> str:
        """
        사용자 입력에 응답 생성
        
        실제로는 LLM을 호출하지만, 여기서는 규칙 기반으로 시뮬레이션
        """
        self.add_message(MessageRole.USER, user_input)
        
        # 도구 호출 필요 여부 확인
        tool_call = self._detect_tool_call(user_input)
        
        if tool_call:
            # 도구 실행
            tool_name, args = tool_call
            result = self._execute_tool(tool_name, args)
            response = f"[도구 '{tool_name}' 실행 결과]\n{result}"
        else:
            # 일반 응답
            response = self._generate_response(user_input)
        
        self.add_message(MessageRole.ASSISTANT, response)
        return response
    
    def _detect_tool_call(self, text: str) -> tuple:
        """도구 호출 감지"""
        # 간단한 패턴 매칭
        for tool_name in self.tools:
            if tool_name.lower() in text.lower():
                # 인자 추출 (간단한 버전)
                args = self._extract_args(text, tool_name)
                return (tool_name, args)
        return None
    
    def _extract_args(self, text: str, tool_name: str) -> Dict:
        """인자 추출 (간단한 버전)"""
        # 숫자 추출
        numbers = re.findall(r'\d+', text)
        if numbers:
            return {"value": int(numbers[0])}
        return {}
    
    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """도구 실행"""
        if tool_name not in self.tools:
            return f"도구 '{tool_name}'을 찾을 수 없습니다."
        
        tool = self.tools[tool_name]
        try:
            result = tool.function(**args)
            return str(result)
        except Exception as e:
            return f"도구 실행 오류: {e}"
    
    def _generate_response(self, text: str) -> str:
        """일반 응답 생성"""
        # 간단한 규칙 기반 응답
        if "안녕" in text:
            return "안녕하세요! 무엇을 도와드릴까요?"
        elif "도움" in text or "help" in text.lower():
            tools_list = ", ".join(self.tools.keys()) if self.tools else "없음"
            return f"저는 {self.name}입니다. 사용 가능한 도구: {tools_list}"
        else:
            return f"'{text}'에 대해 처리 중입니다."


def demo_simple_agent():
    """간단한 에이전트 데모"""
    print("\n" + "="*60)
    print("🤖 간단한 에이전트 데모")
    print("="*60)
    
    # 에이전트 생성
    agent = SimpleAgent(
        name="Assistant",
        system_message="You are a helpful assistant."
    )
    
    # 도구 등록
    def calculator(value: int = 0) -> int:
        return value * 2
    
    def search(query: str = "") -> str:
        return f"'{query}'에 대한 검색 결과입니다."
    
    agent.register_tool(Tool(
        name="calculator",
        description="숫자를 2배로 계산",
        function=calculator
    ))
    
    agent.register_tool(Tool(
        name="search",
        description="웹 검색",
        function=search
    ))
    
    # 대화 테스트
    test_inputs = [
        "안녕하세요",
        "도움이 필요해요",
        "calculator로 5를 계산해줘",
        "날씨 정보를 search해줘"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        response = agent.get_response(user_input)
        print(f"🤖 Agent: {response}")


# ============================================================
# Part 2: 코드 실행 에이전트
# ============================================================

class CodeExecutorAgent:
    """
    코드 실행 에이전트
    
    주의: 실제 환경에서는 샌드박스 사용 필수!
    """
    
    def __init__(self, name: str = "CodeExecutor"):
        self.name = name
        self.execution_history: List[Dict] = []
    
    def execute_code(self, code: str) -> Dict:
        """
        Python 코드 실행
        
        보안 주의: 실제로는 Docker나 샌드박스 사용 필요
        """
        result = {
            "code": code,
            "success": False,
            "output": "",
            "error": ""
        }
        
        try:
            # 안전한 네임스페이스에서 실행
            namespace = {"__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "list": list,
                "dict": dict,
                "str": str,
                "int": int,
                "float": float,
            }}
            
            # 출력 캡처
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            exec(code, namespace)
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            result["success"] = True
            result["output"] = output if output else "코드 실행 완료 (출력 없음)"
            
        except Exception as e:
            result["error"] = str(e)
        
        self.execution_history.append(result)
        return result


def demo_code_executor():
    """코드 실행 에이전트 데모"""
    print("\n" + "="*60)
    print("💻 코드 실행 에이전트 데모")
    print("="*60)
    
    executor = CodeExecutorAgent()
    
    # 테스트 코드들
    test_codes = [
        # 성공 케이스
        """
numbers = [1, 2, 3, 4, 5]
result = sum(numbers)
print(f"합계: {result}")
""",
        # 피보나치
        """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
""",
        # 오류 케이스
        """
import os  # 차단됨
os.system("ls")
"""
    ]
    
    for i, code in enumerate(test_codes, 1):
        print(f"\n--- 코드 {i} ---")
        print(code.strip()[:100] + "..." if len(code) > 100 else code.strip())
        
        result = executor.execute_code(code)
        
        if result["success"]:
            print(f"✅ 성공:\n{result['output']}")
        else:
            print(f"❌ 오류: {result['error']}")


# ============================================================
# Part 3: 멀티에이전트 시뮬레이션
# ============================================================

class AgentRole(Enum):
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"


class MultiAgentSystem:
    """
    멀티에이전트 시스템 시뮬레이터
    """
    
    def __init__(self):
        self.agents: Dict[str, SimpleAgent] = {}
        self.conversation: List[Dict] = []
    
    def add_agent(self, name: str, role: AgentRole, system_message: str):
        """에이전트 추가"""
        agent = SimpleAgent(name, system_message)
        self.agents[name] = agent
    
    def run_conversation(self, task: str, max_rounds: int = 3) -> List[Dict]:
        """
        에이전트 간 대화 실행
        """
        self.conversation = []
        
        # 태스크 시작
        self.conversation.append({
            "speaker": "User",
            "message": task
        })
        
        # 라운드별 실행
        agent_order = list(self.agents.keys())
        
        for round_num in range(max_rounds):
            for agent_name in agent_order:
                agent = self.agents[agent_name]
                
                # 이전 대화 컨텍스트
                context = self._get_context()
                
                # 에이전트 응답 생성 (시뮬레이션)
                response = self._simulate_response(agent_name, context, round_num)
                
                self.conversation.append({
                    "speaker": agent_name,
                    "message": response
                })
                
                # 종료 조건 확인
                if "TERMINATE" in response:
                    return self.conversation
        
        return self.conversation
    
    def _get_context(self) -> str:
        """대화 컨텍스트 생성"""
        return "\n".join([
            f"{item['speaker']}: {item['message']}"
            for item in self.conversation[-5:]  # 최근 5개
        ])
    
    def _simulate_response(self, agent_name: str, context: str, round_num: int) -> str:
        """에이전트 응답 시뮬레이션"""
        if "Planner" in agent_name:
            if round_num == 0:
                return "태스크를 분석했습니다. 다음 단계를 수행하겠습니다:\n1. 데이터 수집\n2. 코드 작성\n3. 검토"
            else:
                return "진행 상황을 확인했습니다. Coder에게 전달합니다."
        
        elif "Coder" in agent_name:
            return """코드를 작성했습니다:
```python
def solution():
    return "Hello, World!"
```
Reviewer에게 검토를 요청합니다."""
        
        elif "Reviewer" in agent_name:
            if round_num >= 1:
                return "코드 검토 완료. 문제없습니다. TERMINATE"
            return "코드를 검토 중입니다. 수정이 필요할 수 있습니다."
        
        return "작업을 처리 중입니다."


def demo_multi_agent():
    """멀티에이전트 시스템 데모"""
    print("\n" + "="*60)
    print("👥 멀티에이전트 시스템 데모")
    print("="*60)
    
    system = MultiAgentSystem()
    
    # 에이전트 추가
    system.add_agent(
        "Planner",
        AgentRole.PLANNER,
        "You plan and coordinate tasks."
    )
    system.add_agent(
        "Coder",
        AgentRole.CODER,
        "You write Python code."
    )
    system.add_agent(
        "Reviewer",
        AgentRole.REVIEWER,
        "You review code quality."
    )
    
    # 대화 실행
    task = "간단한 Hello World 프로그램을 작성해주세요."
    conversation = system.run_conversation(task, max_rounds=2)
    
    print(f"\n📋 태스크: {task}\n")
    print("대화 기록:")
    print("-" * 40)
    
    for item in conversation:
        speaker = item['speaker']
        message = item['message']
        icon = {"User": "👤", "Planner": "📋", "Coder": "💻", "Reviewer": "🔍"}.get(speaker, "🤖")
        print(f"\n{icon} [{speaker}]")
        print(f"   {message}")


# ============================================================
# Part 4: AutoGen 사용 (선택적)
# ============================================================

def demo_autogen():
    """AutoGen 사용 데모"""
    try:
        import autogen
        
        print("\n" + "="*60)
        print("🚀 AutoGen 데모")
        print("="*60)
        
        print("""
AutoGen 사용을 위해서는 OpenAI API 키가 필요합니다.

예제 코드:

```python
import autogen

config_list = [{"model": "gpt-4o-mini", "api_key": "your-key"}]
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "workspace"}
)

user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate factorial."
)
```
        """)
        
    except ImportError:
        print("\n⚠️ pyautogen이 설치되지 않았습니다.")
        print("설치: pip install pyautogen")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 15: LLM 에이전트 실습")
    print("="*60)
    
    demo_simple_agent()
    demo_code_executor()
    demo_multi_agent()
    demo_autogen()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
