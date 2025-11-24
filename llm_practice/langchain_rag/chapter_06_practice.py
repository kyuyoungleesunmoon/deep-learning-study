"""
Chapter 06: Tools & Agents 실습 코드
====================================

이 파일은 LangChain의 Tools와 Agents 개념을 실습합니다:
1. Tool 정의
2. Agent 시뮬레이션
3. ReAct 패턴

실행 방법:
    pip install numpy
    python chapter_06_practice.py

    # LangChain 사용 시:
    pip install langchain langchain-openai tavily-python
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
import re
from datetime import datetime


# ============================================================
# Part 1: Tool 정의
# ============================================================

@dataclass
class Tool:
    """Tool 클래스"""
    name: str
    description: str
    func: Callable
    
    def run(self, input_str: str) -> str:
        """Tool 실행"""
        try:
            return self.func(input_str)
        except Exception as e:
            return f"오류: {e}"


def create_calculator_tool() -> Tool:
    """계산기 Tool 생성"""
    def calculate(expression: str) -> str:
        try:
            # 안전한 수식만 허용
            allowed = set("0123456789+-*/.()")
            if not all(c in allowed or c.isspace() for c in expression):
                return "허용되지 않는 문자가 포함되어 있습니다"
            
            result = eval(expression)
            return str(result)
        except:
            return "계산할 수 없습니다"
    
    return Tool(
        name="calculator",
        description="수학 표현식을 계산합니다. 예: '2 + 2', '10 * 5'",
        func=calculate
    )


def create_datetime_tool() -> Tool:
    """날짜/시간 Tool 생성"""
    def get_datetime(query: str) -> str:
        now = datetime.now()
        if "날짜" in query or "date" in query.lower():
            return now.strftime("%Y년 %m월 %d일")
        elif "시간" in query or "time" in query.lower():
            return now.strftime("%H시 %M분 %S초")
        else:
            return now.strftime("%Y년 %m월 %d일 %H시 %M분")
    
    return Tool(
        name="datetime",
        description="현재 날짜와 시간을 알려줍니다",
        func=get_datetime
    )


def create_search_tool() -> Tool:
    """검색 Tool 생성 (시뮬레이션)"""
    def search(query: str) -> str:
        # 시뮬레이션된 검색 결과
        mock_results = {
            "비트코인": "현재 비트코인 가격은 약 $45,000입니다.",
            "날씨": "서울 날씨: 맑음, 기온 23도",
            "python": "Python은 인기있는 프로그래밍 언어입니다.",
            "gpt": "GPT-4는 OpenAI의 최신 대규모 언어 모델입니다."
        }
        
        for key, result in mock_results.items():
            if key.lower() in query.lower():
                return result
        
        return f"'{query}'에 대한 검색 결과: 관련 정보를 찾을 수 없습니다."
    
    return Tool(
        name="search",
        description="웹에서 실시간 정보를 검색합니다",
        func=search
    )


# ============================================================
# Part 2: 간단한 Agent
# ============================================================

class SimpleAgent:
    """간단한 ReAct 스타일 Agent"""
    
    def __init__(self, tools: List[Tool], verbose: bool = True):
        self.tools = {tool.name: tool for tool in tools}
        self.verbose = verbose
    
    def _get_tool_descriptions(self) -> str:
        """Tool 설명 문자열 생성"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _decide_action(self, question: str) -> Optional[tuple]:
        """어떤 Tool을 사용할지 결정 (간단한 규칙 기반)"""
        question_lower = question.lower()
        
        # 계산 필요 여부
        if any(c in question for c in ['+', '-', '*', '/', '계산', 'calculate']):
            # 수식 추출
            numbers = re.findall(r'[\d\+\-\*\/\.\(\)\s]+', question)
            if numbers:
                return ("calculator", numbers[0].strip())
        
        # 시간 관련
        if any(word in question_lower for word in ['시간', '날짜', 'time', 'date', '오늘']):
            return ("datetime", question)
        
        # 검색 필요
        if any(word in question_lower for word in ['가격', '날씨', '뉴스', '최신', '현재']):
            return ("search", question)
        
        return None
    
    def run(self, question: str, max_iterations: int = 3) -> str:
        """Agent 실행"""
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"질문: {question}")
            print(f"{'='*50}")
        
        for i in range(max_iterations):
            if self.verbose:
                print(f"\n[반복 {i+1}]")
            
            # 1. 추론 (Thought)
            action_result = self._decide_action(question)
            
            if action_result is None:
                if self.verbose:
                    print("Thought: 도구 없이 답변 가능")
                return f"'{question}'에 대한 직접 답변입니다."
            
            tool_name, tool_input = action_result
            
            if self.verbose:
                print(f"Thought: '{tool_name}' 도구를 사용해야겠다")
                print(f"Action: {tool_name}")
                print(f"Action Input: {tool_input}")
            
            # 2. 행동 (Action)
            if tool_name not in self.tools:
                if self.verbose:
                    print(f"Observation: 도구 '{tool_name}'을 찾을 수 없음")
                continue
            
            tool = self.tools[tool_name]
            result = tool.run(tool_input)
            
            if self.verbose:
                print(f"Observation: {result}")
            
            # 3. 결과가 유효하면 최종 답변
            if "오류" not in result and "찾을 수 없" not in result:
                if self.verbose:
                    print(f"\nFinal Answer: {result}")
                return result
        
        return "답변을 생성할 수 없습니다."


# ============================================================
# Part 3: 대화형 Agent
# ============================================================

class ConversationalAgent:
    """대화 히스토리를 유지하는 Agent"""
    
    def __init__(self, tools: List[Tool]):
        self.agent = SimpleAgent(tools, verbose=False)
        self.history: List[Dict[str, str]] = []
    
    def chat(self, user_input: str) -> str:
        """대화"""
        self.history.append({"role": "user", "content": user_input})
        
        response = self.agent.run(user_input)
        
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_history(self) -> List[Dict[str, str]]:
        """대화 히스토리 반환"""
        return self.history


# ============================================================
# 데모 함수들
# ============================================================

def demo_tools():
    """Tool 데모"""
    print("\n" + "="*60)
    print("🔧 Tool 데모")
    print("="*60)
    
    # Tool 생성
    calc_tool = create_calculator_tool()
    time_tool = create_datetime_tool()
    search_tool = create_search_tool()
    
    print("\n[사용 가능한 Tools]")
    for tool in [calc_tool, time_tool, search_tool]:
        print(f"  - {tool.name}: {tool.description}")
    
    # Tool 실행 테스트
    print("\n[Tool 실행 테스트]")
    
    print(f"\n계산기: 2 + 3 * 4 = {calc_tool.run('2 + 3 * 4')}")
    print(f"날짜/시간: {time_tool.run('현재 날짜')}")
    print(f"검색 (비트코인): {search_tool.run('비트코인 가격')}")


def demo_agent():
    """Agent 데모"""
    print("\n" + "="*60)
    print("🤖 Agent 데모")
    print("="*60)
    
    # Agent 생성
    tools = [
        create_calculator_tool(),
        create_datetime_tool(),
        create_search_tool()
    ]
    
    agent = SimpleAgent(tools, verbose=True)
    
    # 다양한 질문 테스트
    questions = [
        "15 * 7 + 23을 계산해줘",
        "오늘 날짜가 뭐야?",
        "비트코인 현재 가격 알려줘"
    ]
    
    for q in questions:
        result = agent.run(q)
        print(f"\n최종 결과: {result}")


def demo_conversational():
    """대화형 Agent 데모"""
    print("\n" + "="*60)
    print("💬 대화형 Agent 데모")
    print("="*60)
    
    tools = [
        create_calculator_tool(),
        create_datetime_tool(),
        create_search_tool()
    ]
    
    chat_agent = ConversationalAgent(tools)
    
    conversations = [
        "안녕하세요!",
        "100 / 4는 얼마야?",
        "지금 몇 시야?",
        "비트코인 정보 알려줘"
    ]
    
    for user_msg in conversations:
        print(f"\n👤 User: {user_msg}")
        response = chat_agent.chat(user_msg)
        print(f"🤖 Agent: {response}")


def demo_react_pattern():
    """ReAct 패턴 시뮬레이션"""
    print("\n" + "="*60)
    print("🔄 ReAct 패턴 시뮬레이션")
    print("="*60)
    
    # ReAct 로그 시뮬레이션
    react_log = """
Question: 서울에서 부산까지 거리가 400km이고, 시속 100km로 달리면 몇 시간이 걸리나요?

Thought: 거리를 속도로 나누어 시간을 계산해야 합니다.
Action: calculator
Action Input: 400 / 100
Observation: 4.0

Thought: 계산 결과를 얻었습니다. 4시간이 걸립니다.
Final Answer: 서울에서 부산까지 시속 100km로 달리면 4시간이 걸립니다.
"""
    
    print(react_log)


def demo_langchain_agent():
    """LangChain Agent 실제 사용 (선택적)"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from langchain import hub
        import os
        
        print("\n" + "="*60)
        print("🚀 LangChain Agent 데모")
        print("="*60)
        
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            print("""
예제 코드:

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub

# 도구
tools = [TavilySearchResults()]

# Agent 생성
prompt = hub.pull("hwchase17/openai-tools-agent")
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = executor.invoke({"input": "오늘 날씨는?"})
            """)
            return
        
    except ImportError:
        print("\n⚠️ langchain 패키지가 설치되지 않았습니다.")
        print("설치: pip install langchain langchain-openai tavily-python")


def demo_multi_tool_selection():
    """Multi-Tool 선택 데모"""
    print("\n" + "="*60)
    print("🔀 Multi-Tool 선택 데모")
    print("="*60)
    
    tools = [
        create_calculator_tool(),
        create_datetime_tool(),
        create_search_tool()
    ]
    
    agent = SimpleAgent(tools, verbose=False)
    
    test_cases = [
        ("123 + 456", "calculator"),
        ("오늘이 며칠이야?", "datetime"),
        ("GPT-4에 대해 알려줘", "search"),
        ("안녕하세요", None)
    ]
    
    print("\n[질문별 Tool 선택]")
    print("-" * 50)
    
    for question, expected in test_cases:
        result = agent._decide_action(question)
        selected = result[0] if result else "None"
        status = "✓" if selected == expected else "✗"
        print(f"{status} '{question[:20]}...' → {selected}")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 06: Tools & Agents 실습")
    print("="*60)
    
    demo_tools()
    demo_agent()
    demo_conversational()
    demo_react_pattern()
    demo_multi_tool_selection()
    demo_langchain_agent()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
