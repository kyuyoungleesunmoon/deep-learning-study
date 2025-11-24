"""
Chapter 03: Models & Prompts 실습 코드
======================================

이 파일은 LangChain의 Models와 Prompts를 실습합니다:
1. 프롬프트 템플릿
2. Output Parser
3. 체인 구성

실행 방법:
    pip install langchain langchain-openai
    export OPENAI_API_KEY="your-api-key"
    python chapter_03_practice.py
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass


# ============================================================
# Part 1: 프롬프트 템플릿 시뮬레이션
# ============================================================

class SimplePromptTemplate:
    """간단한 프롬프트 템플릿"""
    
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        """템플릿에 변수 삽입"""
        result = self.template
        for var in self.input_variables:
            if var in kwargs:
                result = result.replace("{" + var + "}", str(kwargs[var]))
        return result
    
    @classmethod
    def from_template(cls, template: str) -> 'SimplePromptTemplate':
        """템플릿 문자열에서 변수 자동 추출"""
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        return cls(template, variables)


class ChatPromptTemplate:
    """채팅 프롬프트 템플릿"""
    
    def __init__(self, messages: List[tuple]):
        self.messages = messages
    
    @classmethod
    def from_messages(cls, messages: List[tuple]) -> 'ChatPromptTemplate':
        return cls(messages)
    
    def format_messages(self, **kwargs) -> List[Dict[str, str]]:
        """메시지 리스트 생성"""
        result = []
        for role, content in self.messages:
            formatted_content = content
            for key, value in kwargs.items():
                formatted_content = formatted_content.replace("{" + key + "}", str(value))
            result.append({"role": role, "content": formatted_content})
        return result


def demo_prompt_template():
    """프롬프트 템플릿 데모"""
    print("\n" + "="*60)
    print("📝 프롬프트 템플릿 데모")
    print("="*60)
    
    # 기본 PromptTemplate
    template = SimplePromptTemplate.from_template(
        "너는 {role} 전문가야. {topic}에 대해 설명해줘."
    )
    
    prompt = template.format(role="요리", topic="파스타 만드는 법")
    print("\n[기본 템플릿]")
    print(f"템플릿: {template.template}")
    print(f"변수: {template.input_variables}")
    print(f"결과: {prompt}")
    
    # ChatPromptTemplate
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "당신은 {name}이라는 이름의 도우미입니다."),
        ("human", "안녕하세요!"),
        ("assistant", "안녕하세요! 저는 {name}입니다."),
        ("human", "{question}")
    ])
    
    messages = chat_template.format_messages(
        name="루시",
        question="오늘 날씨가 어때?"
    )
    
    print("\n[채팅 템플릿]")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")


# ============================================================
# Part 2: Few-shot 프롬프트
# ============================================================

class FewShotPromptTemplate:
    """Few-shot 프롬프트 템플릿"""
    
    def __init__(self, examples: List[Dict], example_template: SimplePromptTemplate,
                 prefix: str = "", suffix: str = "", input_variables: List[str] = None):
        self.examples = examples
        self.example_template = example_template
        self.prefix = prefix
        self.suffix = suffix
        self.input_variables = input_variables or []
    
    def format(self, **kwargs) -> str:
        # 예시들 포맷팅
        formatted_examples = []
        for example in self.examples:
            formatted = self.example_template.format(**example)
            formatted_examples.append(formatted)
        
        # Suffix 포맷팅
        suffix = self.suffix
        for key, value in kwargs.items():
            suffix = suffix.replace("{" + key + "}", str(value))
        
        # 조합
        parts = [self.prefix] + formatted_examples + [suffix]
        return "\n\n".join(parts)


def demo_few_shot():
    """Few-shot 프롬프트 데모"""
    print("\n" + "="*60)
    print("🎯 Few-shot 프롬프트 데모")
    print("="*60)
    
    # 예시들
    examples = [
        {"word": "아이유", "acrostic": "아: 아이유는\n이: 이 세상에서\n유: 유일한 존재"},
        {"word": "코딩", "acrostic": "코: 코드를\n딩: 딩동댕 완성"}
    ]
    
    # 예시 템플릿
    example_template = SimplePromptTemplate(
        template="단어: {word}\n삼행시:\n{acrostic}",
        input_variables=["word", "acrostic"]
    )
    
    # Few-shot 템플릿
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_template=example_template,
        prefix="다음은 삼행시 예시입니다:",
        suffix="단어: {input_word}\n삼행시:",
        input_variables=["input_word"]
    )
    
    result = prompt.format(input_word="파이썬")
    print("\n[Few-shot 프롬프트]")
    print(result)


# ============================================================
# Part 3: Output Parser
# ============================================================

class CommaSeparatedListOutputParser:
    """쉼표로 구분된 리스트 파서"""
    
    def get_format_instructions(self) -> str:
        return "쉼표로 구분된 리스트 형식으로 답변해주세요. 예: item1, item2, item3"
    
    def parse(self, text: str) -> List[str]:
        """텍스트를 리스트로 파싱"""
        # 쉼표로 분리하고 공백 제거
        items = [item.strip() for item in text.split(",")]
        # 빈 항목 제거
        return [item for item in items if item]


class JsonOutputParser:
    """JSON 출력 파서"""
    
    def __init__(self, schema: Dict[str, str]):
        self.schema = schema
    
    def get_format_instructions(self) -> str:
        fields = ", ".join([f'"{k}": <{v}>' for k, v in self.schema.items()])
        return f"다음 JSON 형식으로 답변해주세요: {{{fields}}}"
    
    def parse(self, text: str) -> Dict:
        """텍스트에서 JSON 추출"""
        import json
        import re
        
        # JSON 패턴 찾기
        json_match = re.search(r'\{[^{}]+\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 실패 시 빈 딕셔너리
        return {}


def demo_output_parser():
    """Output Parser 데모"""
    print("\n" + "="*60)
    print("🔧 Output Parser 데모")
    print("="*60)
    
    # 리스트 파서
    list_parser = CommaSeparatedListOutputParser()
    print("\n[리스트 파서]")
    print(f"지침: {list_parser.get_format_instructions()}")
    
    sample_output = "Python, JavaScript, Java, C++, Go"
    parsed = list_parser.parse(sample_output)
    print(f"입력: {sample_output}")
    print(f"파싱 결과: {parsed}")
    
    # JSON 파서
    json_parser = JsonOutputParser({
        "name": "문자열",
        "age": "숫자",
        "city": "문자열"
    })
    print("\n[JSON 파서]")
    print(f"지침: {json_parser.get_format_instructions()}")
    
    sample_json = '답변입니다. {"name": "홍길동", "age": 30, "city": "서울"}'
    parsed = json_parser.parse(sample_json)
    print(f"입력: {sample_json}")
    print(f"파싱 결과: {parsed}")


# ============================================================
# Part 4: 체인 시뮬레이션
# ============================================================

class SimpleLLM:
    """간단한 LLM 시뮬레이터"""
    
    def __init__(self, name: str = "SimpleLLM"):
        self.name = name
    
    def invoke(self, prompt: str) -> str:
        """LLM 호출 시뮬레이션"""
        # 간단한 규칙 기반 응답
        if "리스트" in prompt or "나열" in prompt:
            return "Python, JavaScript, Java, C++, Go"
        elif "JSON" in prompt or "json" in prompt:
            return '{"name": "테스트", "value": 123}'
        else:
            return f"[{self.name}] {prompt[:50]}에 대한 응답입니다."


class SimpleChain:
    """간단한 체인 (prompt | llm | parser)"""
    
    def __init__(self, prompt_template, llm, parser=None):
        self.prompt_template = prompt_template
        self.llm = llm
        self.parser = parser
    
    def invoke(self, inputs: Dict) -> Any:
        # 1. 프롬프트 생성
        prompt = self.prompt_template.format(**inputs)
        
        # 2. LLM 호출
        response = self.llm.invoke(prompt)
        
        # 3. 파싱 (있는 경우)
        if self.parser:
            return self.parser.parse(response)
        
        return response


def demo_chain():
    """체인 데모"""
    print("\n" + "="*60)
    print("⛓️ 체인 데모")
    print("="*60)
    
    # 구성 요소
    template = SimplePromptTemplate.from_template(
        "{subject}의 종류 5개를 리스트로 나열해주세요."
    )
    llm = SimpleLLM()
    parser = CommaSeparatedListOutputParser()
    
    # 체인 구성
    chain = SimpleChain(template, llm, parser)
    
    # 실행
    result = chain.invoke({"subject": "프로그래밍 언어"})
    
    print(f"\n입력: subject='프로그래밍 언어'")
    print(f"결과: {result}")
    print(f"타입: {type(result)}")


# ============================================================
# Part 5: LangChain 실제 사용 (선택적)
# ============================================================

def demo_langchain():
    """LangChain 실제 사용 데모"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.output_parsers import CommaSeparatedListOutputParser
        
        print("\n" + "="*60)
        print("🚀 LangChain 실제 사용 데모")
        print("="*60)
        
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            print("설정: export OPENAI_API_KEY='your-api-key'")
            return
        
        # 모델 초기화
        chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        # 간단한 호출
        response = chat.invoke("파이썬의 장점을 한 문장으로 설명해줘")
        print(f"\n응답: {response.content}")
        
    except ImportError:
        print("\n⚠️ langchain-openai가 설치되지 않았습니다.")
        print("설치: pip install langchain langchain-openai")


# ============================================================
# 메인 함수
# ============================================================

def main():
    """메인 함수"""
    print("="*60)
    print("🤖 Chapter 03: Models & Prompts 실습")
    print("="*60)
    
    demo_prompt_template()
    demo_few_shot()
    demo_output_parser()
    demo_chain()
    demo_langchain()
    
    print("\n" + "="*60)
    print("✅ 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
