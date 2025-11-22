"""
Emotion Agent

환자 발화(utterance)를 입력받아 감정을 추출하고 위험도를 산출하는 Agent
- Primary emotion (기본 감정)
- Secondary emotion + score (보조 감정 + 점수)
- Subtle emotion + score (세밀 감정 + 점수)
- Masked emotion (억압 감정)
- Emotion risk score (최종 감정 위험도)
"""

import os
import json
from typing import TypedDict, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()


class EmotionState(TypedDict):
    """Emotion Agent 상태"""
    utterance: str  # 입력: 환자 발화
    primary_emotion: str  # 기본 감정
    secondary_emotion: Optional[str]  # 보조 감정
    secondary_score: float  # 보조 감정 점수 (0~1)
    subtle_emotion: Optional[str]  # 세밀 감정
    subtle_score: float  # 세밀 감정 점수 (0~1)
    masked_emotion: Optional[str]  # 억압 감정
    emotion_risk_score: float  # 최종 감정 위험도 (0~1)
    reasoning: str  # LLM의 추론 과정
    error: Optional[str]  # 오류 메시지


class EmotionAgent:
    """감정 추출 및 위험도 산출 Agent"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Args:
            model: 사용할 LLM 모델명
            temperature: LLM temperature (0~1)
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 감정 추출 프롬프트
        self.emotion_prompt = PromptTemplate.from_template("""
You are an expert emotion analysis agent for healthcare triage systems. 
Your task is to analyze a patient's utterance and extract emotional signals.

**Input:**
Patient Utterance: "{utterance}"

**Task:**
Analyze the patient's emotional state and extract the following:

1. **Primary Emotion**: The most dominant, clearly expressed emotion
   - Examples: anger, fear, sadness, joy, disgust, surprise, neutral
   
2. **Secondary Emotion**: A secondary emotion that is also present (if any)
   - Provide the emotion name and intensity score (0.0 to 1.0)
   - If no secondary emotion, return null
   
3. **Subtle Emotion**: A subtle, underlying emotion that may not be immediately obvious
   - Provide the emotion name and intensity score (0.0 to 1.0)
   - If no subtle emotion, return null
   
4. **Masked Emotion**: An emotion that the patient might be suppressing or hiding
   - This requires careful analysis of what is NOT said or implied
   - If no masked emotion detected, return null
   
5. **Emotion Risk Score**: Overall emotional risk level (0.0 to 1.0)
   - Consider: intensity, combination of emotions, masked emotions, potential for escalation
   - Higher scores indicate higher emotional risk
   - Provide brief reasoning for your score

**Output Format (JSON only):**
{{
    "primary_emotion": "emotion_name",
    "secondary_emotion": "emotion_name or null",
    "secondary_score": 0.0-1.0 or null,
    "subtle_emotion": "emotion_name or null",
    "subtle_score": 0.0-1.0 or null,
    "masked_emotion": "emotion_name or null",
    "emotion_risk_score": 0.0-1.0,
    "reasoning": "brief explanation of the risk score"
}}

**Important:**
- Be precise and clinical in your analysis
- Consider cultural and linguistic nuances
- Masked emotions require careful inference
- Risk score should reflect potential for emotional escalation or crisis
- Return ONLY valid JSON, no additional text

Now analyze the utterance and return the JSON:
""")
        
        # LangGraph 워크플로우 구축
        self.workflow = self._build_workflow()
    
    def _extract_emotions(self, state: EmotionState) -> EmotionState:
        """LLM을 사용하여 감정 추출"""
        try:
            # 프롬프트 생성
            prompt = self.emotion_prompt.format(utterance=state["utterance"])
            
            # LLM 호출
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # JSON 파싱 (마크다운 코드 블록 제거)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            emotion_data = json.loads(content)
            
            # 상태 업데이트
            state["primary_emotion"] = emotion_data.get("primary_emotion", "neutral")
            state["secondary_emotion"] = emotion_data.get("secondary_emotion")
            state["secondary_score"] = emotion_data.get("secondary_score", 0.0) or 0.0
            state["subtle_emotion"] = emotion_data.get("subtle_emotion")
            state["subtle_score"] = emotion_data.get("subtle_score", 0.0) or 0.0
            state["masked_emotion"] = emotion_data.get("masked_emotion")
            state["emotion_risk_score"] = emotion_data.get("emotion_risk_score", 0.0)
            state["reasoning"] = emotion_data.get("reasoning", "")
            state["error"] = None
            
        except json.JSONDecodeError as e:
            state["error"] = f"JSON 파싱 오류: {e}\n응답 내용: {content[:200]}"
        except Exception as e:
            state["error"] = f"감정 추출 오류: {e}"
        
        return state
    
    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        workflow = StateGraph(EmotionState)
        
        # 노드 추가
        workflow.add_node("extract_emotions", self._extract_emotions)
        
        # 엣지 설정
        workflow.set_entry_point("extract_emotions")
        workflow.add_edge("extract_emotions", END)
        
        return workflow.compile()
    
    def process(self, utterance: str) -> Dict[str, Any]:
        """
        환자 발화를 처리하여 감정 정보 추출
        
        Args:
            utterance: 환자 발화 텍스트
            
        Returns:
            감정 정보가 포함된 딕셔너리
        """
        # 초기 상태
        initial_state: EmotionState = {
            "utterance": utterance,
            "primary_emotion": "",
            "secondary_emotion": None,
            "secondary_score": 0.0,
            "subtle_emotion": None,
            "subtle_score": 0.0,
            "masked_emotion": None,
            "emotion_risk_score": 0.0,
            "reasoning": "",
            "error": None
        }
        
        # 워크플로우 실행
        result = self.workflow.invoke(initial_state)
        
        # 결과 반환 (JSON 직렬화 가능한 형태)
        return {
            "primary_emotion": result["primary_emotion"],
            "secondary_emotion": result["secondary_emotion"],
            "secondary_score": result["secondary_score"],
            "subtle_emotion": result["subtle_emotion"],
            "subtle_score": result["subtle_score"],
            "masked_emotion": result["masked_emotion"],
            "emotion_risk_score": result["emotion_risk_score"],
            "reasoning": result["reasoning"],
            "error": result["error"]
        }
    
    def process_batch(self, utterances: list[str]) -> list[Dict[str, Any]]:
        """
        여러 발화를 배치로 처리
        
        Args:
            utterances: 발화 텍스트 리스트
            
        Returns:
            감정 정보 딕셔너리 리스트
        """
        results = []
        for utterance in utterances:
            result = self.process(utterance)
            results.append(result)
        return results


def main():
    """테스트용 메인 함수"""
    # Emotion Agent 초기화
    agent = EmotionAgent()
    
    # 테스트 발화
    test_utterances = [
        "I'm really worried about this pain in my chest. It's been getting worse.",
        "I feel fine, everything is okay. No problems at all.",
        "I'm so frustrated with this constant headache. Nothing seems to help.",
    ]
    
    print("=" * 60)
    print("Emotion Agent 테스트")
    print("=" * 60)
    
    for i, utterance in enumerate(test_utterances, 1):
        print(f"\n[테스트 {i}]")
        print(f"발화: {utterance}")
        print("-" * 60)
        
        result = agent.process(utterance)
        
        if result["error"]:
            print(f"❌ 오류: {result['error']}")
        else:
            print(f"✓ Primary Emotion: {result['primary_emotion']}")
            if result["secondary_emotion"]:
                print(f"✓ Secondary Emotion: {result['secondary_emotion']} (score: {result['secondary_score']:.2f})")
            if result["subtle_emotion"]:
                print(f"✓ Subtle Emotion: {result['subtle_emotion']} (score: {result['subtle_score']:.2f})")
            if result["masked_emotion"]:
                print(f"✓ Masked Emotion: {result['masked_emotion']}")
            print(f"✓ Emotion Risk Score: {result['emotion_risk_score']:.2f}")
            print(f"✓ Reasoning: {result['reasoning']}")
        
        print()


if __name__ == "__main__":
    main()

